# train.py — Parquet‑shard trainer with full (epoch + batch) resume
# -----------------------------------------------------------------
import os, math, random, json, glob, numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter

import pyarrow.parquet as pq
import pymongo

from shared import (
    build_to_tensor,
    FIGHTERS,
    IDX,
    STACK_LIMITS,
    MERC_CLIP,
    BOARD_W,
    BOARD_H,
    C_IN,
)

# -------------------------------------------------------------------------
# Paths / hyper‑parameters
# -------------------------------------------------------------------------
SHARDS        = sorted(glob.glob("shards/*.parquet"))
CKPT_PATH     = "checkpoints/wavecnn.chkpt"
LOG_DIR       = f"runs/{datetime.now().strftime('%Y%m%d_%H%M')}"

BATCH         = 256
NUM_WORKERS   = 16
PREFETCH      = 8
LR            = 1e-2
WEIGHT_DECAY  = 1e-4
SAVE_EVERY    = 1_000
NUM_EPOCHS    = 10

WAVE_MAX      = 20
NF            = len(FIGHTERS)
DEVICE        = "cuda"

# -------------------------------------------------------------------------
# Dataset‑wide helpers
# -------------------------------------------------------------------------
def _count_rows(paths):
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)

DOCS             = _count_rows(SHARDS)
STEPS_PER_EPOCH  = math.ceil(DOCS / BATCH)
TOTAL_STEPS      = STEPS_PER_EPOCH * NUM_EPOCHS

# -------------------------------------------------------------------------
# Optional manual regression tests
# -------------------------------------------------------------------------
with open("tests.json", "r") as f:
    TEST_CASES = json.load(f)

def eval_manual_tests(model):
    model.eval()
    tot_err, passed = 0.0, 0
    with torch.no_grad(), autocast(device_type="cuda", enabled=True):
        for case in TEST_CASES:
            board = build_to_tensor(case["export"]).unsqueeze(0).to(DEVICE)
            wave  = torch.tensor([[case["wave"] - 1]], dtype=torch.long, device=DEVICE)
            merc_val = case["merc"]
            merc_feat = torch.tensor(
                [[math.log1p(min(merc_val, MERC_CLIP)) / math.log1p(MERC_CLIP)]],
                dtype=torch.float16,
                device=DEVICE,
            )
            pct = torch.sigmoid(model(board, wave, merc_feat)).item() * 100.0
            diff = abs(pct - case["expected_leak"])
            tot_err += diff
            passed  += diff <= 5.0
    model.train()
    return tot_err / len(TEST_CASES), passed / len(TEST_CASES) * 100.0

# -------------------------------------------------------------------------
# Parquet‑sharded dataset
# -------------------------------------------------------------------------
class WaveDataset(IterableDataset):
    """
    Streams each Parquet row exactly once per epoch across all shards.
    Shard and row‑group order are re‑randomised every epoch.
    Workers take every N‑th row where N = num_workers (round‑robin).
    """
    def __init__(self, shard_paths, epoch=0):
        super().__init__()
        self.shard_paths = shard_paths
        self.epoch       = epoch
        cli = pymongo.MongoClient("mongodb://127.0.0.1:27017/", connect=False)
        self.wave_val = {
            int(d["levelNum"]): float(d.get("totalReward", 0.0))
            for d in cli["legiontd2"]["waves"].find({}, {"levelNum": 1, "totalReward": 1})
        }

    def __iter__(self):
        worker      = get_worker_info()
        wid, wnum   = (worker.id, worker.num_workers) if worker else (0, 1)
        rng         = np.random.default_rng(self.epoch + wid)

        for path in rng.permutation(self.shard_paths):
            pf = pq.ParquetFile(path)
            for rg in rng.permutation(pf.num_row_groups):
                tbl = pf.read_row_group(
                    rg, columns=["_id", "totalLeaked", "totalOccurrences"]
                ).to_pydict()

                for loc in range(wid, len(tbl["_id"]), wnum):
                    sample = self._row_to_sample(tbl, loc)
                    if sample is not None:
                        yield sample

    # ---------------------------------------------------------------------
    # row parser
    # ---------------------------------------------------------------------
    def _row_to_sample(self, tbl, loc):
        _id        = tbl["_id"][loc]
        total_leak = float(tbl["totalLeaked"][loc])
        total_occ  = float(tbl["totalOccurrences"][loc])

        part_wave, part_merc, fighters_txt = (_id.split("|") + [""])[:3]
        wave_num    = int(part_wave.split("=")[1])
        merc_bounty = float(part_merc.split("=")[1])

        threat = self.wave_val.get(wave_num, 0.0) + merc_bounty
        if threat <= 0:
            return None
        frac = (total_leak / total_occ) / threat
        if frac > 1.0001:
            print(f"raw fraction is over 1. id:{_id}")
            return None
        if frac == 0.0 and random.random() > 0.1:
            return None

        board = torch.zeros((C_IN, BOARD_H, BOARD_W), dtype=torch.float16)
        if fighters_txt:
            for ent in fighters_txt.split(","):
                uid, xs, ys, st = ent.split(":")
                ch = IDX.get(uid)
                if ch is None:
                    return None
                x = int(float(xs) - 0.5)
                y = int(float(ys) - 0.5)
                if not (0 <= x < BOARD_W - 1 and 0 <= y < BOARD_H - 1):
                    return None
                board[ch, y:y + 2, x:x + 2] = 1
                max_stack = STACK_LIMITS.get(uid, 0)
                if max_stack:
                    norm = min(int(st), max_stack) / max_stack
                    board[NF, y:y + 2, x:x + 2] = norm

        wave_id   = torch.tensor([wave_num - 1], dtype=torch.long)
        merc_feat = torch.tensor(
            [math.log1p(min(merc_bounty, MERC_CLIP)) / math.log1p(MERC_CLIP)],
            dtype=torch.float16,
        )
        target = torch.tensor([frac], dtype=torch.float16)
        return board, wave_id, merc_feat, target

# -------------------------------------------------------------------------
# DataLoader factory
# -------------------------------------------------------------------------
def get_loader(epoch):
    ds = WaveDataset(SHARDS, epoch)
    return DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH,
        persistent_workers=True,
    )

# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(C_IN, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.wave_emb = nn.Embedding(WAVE_MAX, 8)
        self.merc_mlp = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 4))
        self.flat     = 128 * 7 * 4
        self.head     = nn.Sequential(
            nn.Linear(self.flat + 8 + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, board, wave_id, merc_feat):
        z_img  = self.cnn(board).flatten(1)
        z_wave = self.wave_emb(wave_id.squeeze(-1))
        z_merc = self.merc_mlp(merc_feat)
        return self.head(torch.cat([z_img, z_wave, z_merc], 1))

# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------
def save_ckpt(step, model, opt, scaler):
    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "scaler": scaler.state_dict(),
        },
        CKPT_PATH,
    )

def load_ckpt(model, opt, scaler):
    if not os.path.isfile(CKPT_PATH):
        return 0
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optim"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0)

# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------
def train():
    model = ConvNet().to(DEVICE)
    model = torch.compile(model, backend="eager")

    opt    = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    scaler = GradScaler(enabled=DEVICE.startswith("cuda"))

    step          = load_ckpt(model, opt, scaler)
    start_epoch   = step // STEPS_PER_EPOCH
    batches_done  = step %  STEPS_PER_EPOCH

    writer        = SummaryWriter(LOG_DIR)
    loss_fn       = nn.BCEWithLogitsLoss()

    running_loss  = 0.0                       # accumulate 1 000 step loss

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            for batch_idx, (boards, wave_ids, merc_feats, targets) in enumerate(get_loader(epoch)):
                if epoch == start_epoch and batch_idx < batches_done:
                    continue

                boards, wave_ids, merc_feats, targets = (
                    boards.to(DEVICE, non_blocking=True),
                    wave_ids.to(DEVICE, non_blocking=True),
                    merc_feats.to(DEVICE, non_blocking=True),
                    targets.to(DEVICE, non_blocking=True),
                )

                opt.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=True):
                    logits = model(boards, wave_ids, merc_feats).squeeze()
                    loss   = loss_fn(logits, targets.squeeze())

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                step += 1
                running_loss += loss.item()    # record every-step loss

                if step % SAVE_EVERY == 0:     # 1 000-step boundary
                    avg_loss = running_loss / SAVE_EVERY
                    writer.add_scalar("train/loss_avg", avg_loss, step)
                    running_loss = 0.0          # reset window

                    if TEST_CASES:              # manual regression suite
                        err, rate = eval_manual_tests(model)
                        writer.add_scalar("train/validate mean error %", err,  step)
                        writer.add_scalar("train/validate tests passed %",  rate, step)

                    save_ckpt(step, model, opt, scaler)

            # end-of-epoch checkpoint
            save_ckpt(step, model, opt, scaler)
            print(f"epoch complete: {epoch}/{NUM_EPOCHS}")

    except KeyboardInterrupt:
        print("\n[info] Interrupted — saving checkpoint...")
        save_ckpt(step, model, opt, scaler)
        print("done")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    train()

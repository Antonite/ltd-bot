# train.py — Parquet-shard trainer (full-precision, no AMP)
# -------------------------------------------------------------------------
import os, math, random, json, glob, numpy as np, timm
from datetime import datetime
from dataclasses import dataclass
from typing import Optional 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter

import pyarrow.parquet as pq
import pymongo

from shared import (
    build_to_tensor,
    FIGHTERS,
    STACK_LIMITS,
    IDX,
    MERC_CLIP,
    BOARD_W,
    BOARD_H,
    C_IN,
)

# -------------------------------------------------------------------------
# Paths / hyper-parameters
# -------------------------------------------------------------------------
SHARDS        = sorted(glob.glob("shards/*.parquet"))
CKPT_PATH     = "checkpoints/waveconvnext.chkpt"
LOG_DIR       = f"runs/{datetime.now().strftime('%Y%m%d_%H%M')}"

BATCH         = 512
NUM_WORKERS   = 12
PREFETCH      = 8
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
SAVE_EVERY    = 1_000
NUM_EPOCHS    = 10

WAVE_MAX      = 20
NF            = len(FIGHTERS)
DEVICE        = "cuda"

# class-balance weights (shared by dataset and sampler)
POS_W = 4.0
NEG_W = 1.0

# fixed positional grids (share storage with shared.py)
_X_GRID = torch.linspace(0, 1, BOARD_W, dtype=torch.float32).repeat(BOARD_H, 1)
_Y_GRID = torch.linspace(0, 1, BOARD_H, dtype=torch.float32).unsqueeze(1).repeat(1, BOARD_W)

# -------------------------------------------------------------------------
# Dataset-wide helpers
# -------------------------------------------------------------------------
def _count_rows(paths):
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)

DOCS            = _count_rows(SHARDS)
STEPS_PER_EPOCH = math.ceil(DOCS / BATCH)

# -------------------------------------------------------------------------
# Optional manual regression tests
# -------------------------------------------------------------------------
with open("tests.json", "r") as f:
    TEST_CASES = json.load(f)

def eval_manual_tests(model):
    model.eval()
    tot_err, passed = 0.0, 0
    with torch.no_grad():
        for case in TEST_CASES:
            board = build_to_tensor(case["export"]).unsqueeze(0).to(DEVICE)
            wave  = torch.tensor([[case["wave"] - 1]], dtype=torch.long, device=DEVICE)

            merc_feat = torch.tensor(
                [[math.log1p(min(case["merc"], MERC_CLIP)) / math.log1p(MERC_CLIP)]],
                dtype=torch.float32, device=DEVICE,
            )

            logit = model(board, wave, merc_feat).squeeze()
            pct   = torch.sigmoid(logit).item() * 100.0

            diff   = abs(pct - case["expected_leak"])
            tot_err += diff
            passed  += diff <= 5.0
    model.train()
    return tot_err / len(TEST_CASES), passed / len(TEST_CASES) * 100.0

# -------------------------------------------------------------------------
# Parquet-sharded dataset
# -------------------------------------------------------------------------
@dataclass
class WaveSample:
    board:     torch.Tensor   # (C,H,W) float32
    wave_id:   torch.Tensor   # ()  long
    merc_feat: torch.Tensor   # ()  float32 (log-scaled)
    target:    torch.Tensor   # ()  float32

    @property
    def weight(self) -> float:
        return POS_W if self.target.item() > 0 else NEG_W

class WaveDataset(IterableDataset):
    """Streams every Parquet row exactly once per epoch, with augments."""
    def __init__(self, shard_paths, epoch_seed=0):
        super().__init__()
        self.shard_paths = shard_paths
        self.epoch_seed  = epoch_seed
        cli = pymongo.MongoClient("mongodb://127.0.0.1:27017/", connect=False)
        self.wave_val = {
            int(d["levelNum"]): float(d.get("totalReward", 0.0))
            for d in cli["legiontd2"]["waves"].find({}, {"levelNum": 1, "totalReward": 1})
        }

    def __iter__(self):
        worker      = get_worker_info()
        wid, wnum   = (worker.id, worker.num_workers) if worker else (0, 1)
        rng         = np.random.default_rng(self.epoch_seed + wid)

        for path in rng.permutation(self.shard_paths):
            pf = pq.ParquetFile(path)
            for rg in rng.permutation(pf.num_row_groups):
                tbl = pf.read_row_group(
                    rg, columns=["_id", "totalLeaked", "totalOccurrences"]
                ).to_pydict()

                for loc in range(wid, len(tbl["_id"]), wnum):
                    sample = self._row_to_sample(tbl, loc, rng)
                    if sample is not None:
                        yield sample

    # ---------------------------------------------------------------------
    # row → tensors
    # ---------------------------------------------------------------------
    def _row_to_sample(self, tbl, loc, rng) -> Optional[WaveSample]:
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
            return None
        if frac == 0.0 and random.random() > 0.1:   # down-sample perfect holds
            return None

        # ── board tensor ────────────────────────────────────────────
        board = torch.zeros((C_IN, BOARD_H, BOARD_W), dtype=torch.float32)

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

        # positional channels
        board[NF + 1] = _X_GRID
        board[NF + 2] = _Y_GRID

        # 50 % horizontal flip
        if rng.random() < 0.5:
            board = torch.flip(board, dims=[2])     # flip W-axis
            board[NF + 1] = 1.0 - board[NF + 1]

        # scalars (0-D tensors)
        wave_id   = torch.tensor(wave_num - 1, dtype=torch.long)
        merc_feat = torch.tensor(
            math.log1p(min(merc_bounty, MERC_CLIP)) / math.log1p(MERC_CLIP),
            dtype=torch.float32,
        )
        target = torch.tensor(frac, dtype=torch.float32)
        return WaveSample(board, wave_id, merc_feat, target)

# -------------------------------------------------------------------------
# DataLoader factory
# -------------------------------------------------------------------------
class OnTheFlyWeighted(IterableDataset):
    def __init__(self, base_ds, *, pos_w: float, neg_w: float,
                 num_samples: int | None = None, seed: int = 0):
        super().__init__()
        self.base_ds     = base_ds
        self.pos_w, self.neg_w = pos_w, neg_w
        self.num_samples = num_samples
        self.seed        = seed
        self._w_max      = max(pos_w, neg_w)

    def __iter__(self):
        info  = torch.utils.data.get_worker_info()
        rng   = np.random.default_rng(self.seed + (info.id if info else 0))
        count = 0
        for sample in self.base_ds:
            if rng.random() < sample.weight / self._w_max:
                yield (
                    sample.board,
                    sample.wave_id.unsqueeze(0),
                    sample.merc_feat.unsqueeze(0),
                    sample.target.unsqueeze(0),
                )
                count += 1
                if self.num_samples and count >= self.num_samples:
                    break

def get_loader(epoch_seed: int):
    base = WaveDataset(SHARDS, epoch_seed)
    ds   = OnTheFlyWeighted(
        base,
        pos_w=POS_W,
        neg_w=NEG_W,
        num_samples=BATCH * STEPS_PER_EPOCH,
        seed=epoch_seed,
    )
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
class WaveModel(nn.Module):
    """CNN encoder (board) + embeddings (wave, merc features)."""
    def __init__(self):
        super().__init__()

        patch_size = 2
        self.cnn = timm.create_model(
            "convnextv2_nano",
            patch_size=patch_size,
            in_chans=C_IN,
            num_classes=0,
            pretrained=False,
            drop_path_rate=0.1,
        )
        self._divisor = patch_size * 8      # pad so H,W % 16 == 0

        self.wave_emb = nn.Embedding(WAVE_MAX, 8)
        self.merc_mlp = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 4)
        )

        self.head = nn.Sequential(
            nn.Linear(self.cnn.num_features + 8 + 4, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),             # raw logits
        )

    def forward(self, board, wave_id, merc_feat):
        h, w = board.shape[-2:]
        pad_h = (self._divisor - h % self._divisor) % self._divisor
        pad_w = (self._divisor - w % self._divisor) % self._divisor
        if pad_h or pad_w:
            board = F.pad(board, (0, pad_w, 0, pad_h))

        z_img  = self.cnn(board)
        z_wave = self.wave_emb(wave_id.squeeze(-1))
        z_merc = self.merc_mlp(merc_feat)
        return self.head(torch.cat([z_img, z_wave, z_merc], 1))

# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------
def save_ckpt(model, opt):
    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(),
                "optim": opt.state_dict()}, CKPT_PATH)

def load_ckpt(model, opt):
    if not os.path.isfile(CKPT_PATH):
        return
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optim"])

# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------
def train():
    model = WaveModel().to(DEVICE)
    opt = optim.AdamW(
        model.parameters(),
        lr=1e-4,                               
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    load_ckpt(model, opt=None)
    model = torch.compile(model, backend="eager")

    writer        = SummaryWriter(LOG_DIR)

    # IMPORTANT: targets are *continuous leak fractions* (0-1) →
    #            use a *regression* loss, NOT BCE.
    loss_fn       = nn.SmoothL1Loss(beta=0.01)

    step, running_loss, base_seed = 0, 0.0, random.randint(0, 2**31 - 1)

    try:
        for epoch in range(NUM_EPOCHS):
            loader = get_loader(base_seed + epoch)
            for boards, wave_ids, merc_feats, targets in loader:
                boards, wave_ids, merc_feats, targets = (
                    boards.to(DEVICE, non_blocking=True),
                    wave_ids.to(DEVICE, non_blocking=True),
                    merc_feats.to(DEVICE, non_blocking=True),
                    targets.to(DEVICE, non_blocking=True),
                )
                opt.zero_grad(set_to_none=True)

                logits = model(boards, wave_ids, merc_feats).squeeze()
                preds  = torch.sigmoid(logits)          # 0–1 leak fraction
                loss   = loss_fn(preds, targets.squeeze())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

                step += 1
                running_loss += loss.item()

                if step % SAVE_EVERY == 0:
                    writer.add_scalar("train/loss_avg", running_loss / SAVE_EVERY, step)
                    running_loss = 0.0
                    if TEST_CASES:
                        err, rate = eval_manual_tests(model)
                        writer.add_scalar("train/validate mean error %", err,  step)
                        writer.add_scalar("train/validate tests passed %",  rate, step)
                    save_ckpt(model, opt)
        save_ckpt(model, opt)
    except KeyboardInterrupt:
        print("saving...")
        save_ckpt(model, opt)
        print("saved")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    train()

# train.py — Parquet-shard trainer (static LR, no scheduler)
# full-precision, no AMP desired
# parquet files contain about 50 million unique positions
# about 40 % of unique positions do not leak – target is zero
# trained on a Windows machine with an RTX 4090 + 64 GB RAM
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

import logging, sys, faulthandler, traceback, signal
LOG_FILE = f"train_errors_{datetime.now():%Y%m%d}.log"

# uncaught Python exceptions -------------------------------------------
def _log_uncaught(exc_type, exc_value, exc_tb):
    logging.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc_value, exc_tb))
sys.excepthook = _log_uncaught
# native faults (segfault, illegal instr., abort, etc.) -----------------
faulthandler.enable(open(LOG_FILE, "a"), all_threads=True)
# polite / external kills ----------------------------------------------
def _log_signal(sig, frame):
    logging.critical("Received signal %s — dumping stack", sig)
    logging.critical("".join(traceback.format_stack(frame)))
    faulthandler.dump_traceback(file=open(LOG_FILE, "a"))
    os._exit(1)        # hard-exit so trainer exits with non-zero status
for _s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_s, _log_signal)

from shared import (                     # ← keep the same order
    build_to_tensor,
    FIGHTERS,
    STACK_LIMITS,
    IDX,
    MERC_CLIP,
    BOARD_W,
    BOARD_H,
    C_IN,
    STACK_CH, 
    SPELL_CH,
    X_CH,
    Y_CH,
)

from coord_utils import (
    half_tile_coord_to_index
)

# -------------------------------------------------------------------------
# Paths / hyper-parameters
# -------------------------------------------------------------------------
SHARDS        = sorted(glob.glob("shards/*.parquet"))
CKPT_PATH     = "checkpoints/waveconvnext.chkpt"
LOG_DIR       = f"runs/{datetime.now().strftime('%Y%m%d_%H%M')}"

BATCH         = 128
NUM_WORKERS   = 4
PREFETCH      = 8
LR            = 1e-4         
WEIGHT_DECAY  = 1e-4
SAVE_EVERY    = 1_000
TEST_EVERY    = 5_000
NUM_EPOCHS    = 20

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

@torch.no_grad()
def eval_manual_tests(model):
    """
    Return (mean_abs_error %, pass_rate %) on TEST_CASES.

    Works for either a compiled wrapper or a plain nn.Module because we
    never touch .eval() / .train(); we just forward-prop once per case.
    """
    device = next(model.parameters()).device
    total_err, passed = 0.0, 0

    for case in TEST_CASES:
        board = build_to_tensor(case["export"]).unsqueeze(0).to(device)
        wave  = torch.tensor([[case["wave"] - 1]], dtype=torch.long, device=device)
        merc  = min(case["merc"], MERC_CLIP)
        merc_feat = torch.tensor(
            [[math.log1p(merc) / math.log1p(MERC_CLIP)]],
            dtype=torch.float32, device=device,
        )

        pct  = torch.sigmoid(model(board, wave, merc_feat)).item() * 100.0
        diff = abs(pct - case["expected_leak"])
        total_err += diff
        passed   += diff <= 5.0

    n = len(TEST_CASES)
    return total_err / n, passed / n * 100.0

# -------------------------------------------------------------------------
# Parquet-sharded dataset
# -------------------------------------------------------------------------
@dataclass
class WaveSample:
    board:     torch.Tensor   # (C_IN × H × W)
    wave_id:   torch.Tensor   # long  ∈ [0,19]
    merc_feat: torch.Tensor   # scalar ∈ [0,1]
    target:    torch.Tensor   # scalar ∈ [0,1]
    @property
    def weight(self) -> float:
        return POS_W if self.target.item() > 0 else NEG_W


class WaveDataset(IterableDataset):
    """Streams every Parquet row exactly once per epoch, with augments."""

    def __init__(self, shard_paths: list[str], epoch_seed: int = 0):
        super().__init__()
        self.shard_paths = shard_paths
        self.epoch_seed  = epoch_seed        # different seed each epoch

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_parse_id(_id: str):
        """
        Returns (wave_num:int, merc_bounty:float, fighters_txt:str) or None.
        Gracefully handles bad formats instead of raising.
        """
        try:
            wave_part, merc_part, fighters_txt = (_id.split("|") + ["", ""])[:3]
            wave_num    = int(wave_part.split("=")[1])
            merc_bounty = float(merc_part.split("=")[1] or 0.0)
            return wave_num, merc_bounty, fighters_txt
        except (IndexError, ValueError):
            return None

    # ------------------------------------------------------------------
    def __iter__(self):
        worker = get_worker_info()
        wid, wnum = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = np.random.default_rng(self.epoch_seed + wid)

        # wave-reward lookup (inside the worker → safe with fork)
        cli = pymongo.MongoClient("mongodb://127.0.0.1:27017/", connect=False)
        wave_val = {
            int(d["levelNum"]): float(d.get("totalReward", 0.0))
            for d in cli["legiontd2"]["waves"].find({}, {"levelNum": 1, "totalReward": 1})
        }

        # one pinned copy of the positional grids
        grid_x = _X_GRID.to(torch.float32).pin_memory()
        grid_y = _Y_GRID.to(torch.float32).pin_memory()

        for path in rng.permutation(self.shard_paths):
            pf = pq.ParquetFile(path)
            for rg in rng.permutation(pf.num_row_groups):
                tbl = pf.read_row_group(rg).to_pydict()
                n = len(tbl["_id"])
                for loc in rng.permutation(n):
                    samp = self._row_to_sample(tbl, loc, rng, wave_val, grid_x, grid_y)
                    if samp is not None:
                        yield samp

    # ------------------------------------------------------------------
    def _row_to_sample(
        self,
        tbl: dict,
        loc: int,
        rng: np.random.Generator,
        wave_val: dict[int, float],
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
    ) -> Optional["WaveSample"]:
        """Convert one Parquet row to a training sample (or return None)."""

        _id        = tbl["_id"][loc]
        total_leak = float(tbl["totalLeaked"][loc])
        total_occ  = float(tbl["totalOccurrences"][loc])
        if total_occ <= 0:
            print(f"empty occ {_id}")

        parsed = self._safe_parse_id(_id)
        if parsed is None:
            return None
        wave_num, merc_bounty, fighters_txt = parsed

        threat = wave_val.get(wave_num, 0.0) + merc_bounty
        if threat <= 0:
            return None
        frac = (total_leak / total_occ) / threat
        if not (0.0 <= frac <= 1.0001):
            return None
        if frac == 0.0 and rng.random() > 0.1:          # heavy down-sample of easy 0-leaks
            return None

        board = torch.zeros((C_IN, BOARD_H, BOARD_W), dtype=torch.float32)
        if fighters_txt:
            for ent in fighters_txt.split(","):
                parts = ent.split(":")
                uid, xs, ys, st, flag = parts

                ch = IDX.get(uid)
                if ch is None:
                    print(f"couldn't find unit {ch}, ent: {ent}, fighters_txt: {fighters_txt}")
                    continue
                x = half_tile_coord_to_index(float(xs))
                y = half_tile_coord_to_index(float(ys))
                if not (0 <= x < BOARD_W - 1 and 0 <= y < BOARD_H - 1):
                    print(f"outside of board BOARD_W: {BOARD_W}, BOARD_H: {BOARD_H}, fighters_txt: {fighters_txt}")
                    continue

                # fighter presence
                board[ch, y:y+2, x:x+2] = 1.0

                # stack encoding
                max_stack = STACK_LIMITS.get(uid, 2)
                board[STACK_CH, y:y+2, x:x+2] = min(int(st), max_stack) / max_stack

                # spell flag (binary)
                if int(flag) > 0:
                    board[SPELL_CH, y:y+2, x:x+2] = 1.0

        # positional channels
        board[X_CH].copy_(grid_x)
        board[Y_CH].copy_(grid_y)

        wave_id   = torch.tensor(wave_num - 1, dtype=torch.long)
        merc_feat = torch.tensor(
            math.log1p(min(merc_bounty, MERC_CLIP)) / math.log1p(MERC_CLIP),
            dtype=torch.float32,
        )
        target    = torch.tensor(frac, dtype=torch.float32)
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
        multiprocessing_context="spawn"
    )

# -------------------------------------------------------------------------
# Model (drop-in replacement)
# -------------------------------------------------------------------------
class WaveModel(nn.Module):
    """Higher-capacity CNN encoder + embeddings."""
    def __init__(self):
        super().__init__()

        self.cnn = timm.create_model(
            "convnextv2_base",
            patch_size=1,
            in_chans=C_IN,
            num_classes=0,
            pretrained=False,   # train from scratch
            drop_path_rate=0.2,
        )
        self._divisor = 8

        self.wave_emb = nn.Embedding(WAVE_MAX, 16)
        self.merc_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 8)
        )

        self.head = nn.Sequential(
            nn.Linear(self.cnn.num_features + 16 + 8, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
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
def save_ckpt(model, opt, step: int):
    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "step":  step,
        },
        CKPT_PATH,
    )

def load_ckpt(model, opt, *, lr_override: float | None = None) -> int:
    """
    Load checkpoint and (optionally) override the learning-rate of every
    parameter group so it is guaranteed to match `lr_override`.

    Returns the training step stored in the checkpoint (0 if none).
    """
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optim"])

    if lr_override is not None:
        for pg in opt.param_groups:
            pg["lr"] = lr_override

    return ckpt.get("step", 0)

# -------------------------------------------------------------------------
# Training loop (no scheduler)
# -------------------------------------------------------------------------
def train():
    model = WaveModel().to(DEVICE)
    opt = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ------------------------------------------------------------------
    # Resume if a checkpoint exists, otherwise start from scratch
    # ------------------------------------------------------------------
    if os.path.isfile(CKPT_PATH):
        try:
            step = load_ckpt(model, opt, lr_override=LR)
            logging.info("Resumed from %s at step %d", CKPT_PATH, step)
        except Exception:
            logging.exception(
                "Checkpoint %s exists but could not be loaded – "
                "training will start from scratch",
                CKPT_PATH,
            )
            step = 0
    else:
        logging.info("No checkpoint found at %s – training from scratch", CKPT_PATH)
        step = 0

    writer        = SummaryWriter(LOG_DIR)
    loss_fn       = nn.SmoothL1Loss(beta=0)
    running_loss  = 0.0
    base_seed     = random.randint(0, 2**31 - 1)

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
            preds  = torch.sigmoid(logits)
            loss   = loss_fn(preds, targets.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            step += 1
            opt.step()

            running_loss += loss.item()
            if step % SAVE_EVERY == 0:
                writer.add_scalar("train/loss_avg", running_loss / SAVE_EVERY, step)
                writer.add_scalar("train/lr",       opt.param_groups[0]["lr"], step)
                running_loss = 0.0
                save_ckpt(model, opt, step)

            if step % TEST_EVERY == 0 and TEST_CASES:
                err, rate = eval_manual_tests(model)
                writer.add_scalar("train/validate mean error %", err,  step)
                writer.add_scalar("train/validate tests passed %",  rate, step)
                
        print(f"epoch {epoch + 1} complete")

    save_ckpt(model, opt, step)

# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    try:
        train()
    except Exception:
        logging.exception("Training crashed hard")
        raise

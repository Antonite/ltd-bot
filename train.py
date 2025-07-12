# train.py — Parquet-shard trainer (static LR, pure-L1 objective)
# -------------------------------------------------------------------------
# full-precision, no AMP desired
# parquet files contain ~50 million unique positions
# about 40 % of positions do not leak – target is zero
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

# ─────────────────────── safety / debug hooks ──────────────────────────
def _log_uncaught(exc_type, exc_value, exc_tb):
    logging.critical("UNCAUGHT EXCEPTION",
                     exc_info=(exc_type, exc_value, exc_tb))
sys.excepthook = _log_uncaught

faulthandler.enable(open(LOG_FILE, "a"), all_threads=True)

def _log_signal(sig, frame):
    logging.critical("Received signal %s — dumping stack", sig)
    logging.critical("".join(traceback.format_stack(frame)))
    faulthandler.dump_traceback(file=open(LOG_FILE, "a"))
    os._exit(1)

for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, _log_signal)

# ───────────────────────────── project imports ──────────────────────────
from shared import (
    FIGHTERS,
    STACK_LIMITS,
    IDX,
    MERC_CLIP,
    BOARD_W,
    BOARD_H,
    C_IN,
    N_SPELL,
    STACK_CH,
    SPELL_CH,
    X_CH,
    Y_CH,
    NF,
    spell_channel,
)
from coord_utils import half_tile_coord_to_index

# ─────────────────────────── paths / hyper-params ───────────────────────
SHARDS        = sorted(glob.glob("shards/*.parquet"))
VAL_SHARDS    = sorted(glob.glob("shards/val/*.parquet"))
CKPT_PATH     = "checkpoints/waveconvnext.chkpt"
LOG_DIR       = f"runs/{datetime.now():%Y%m%d_%H%M}"

BATCH         = 128
NUM_WORKERS   = 4
PREFETCH      = 8
LR            = 1e-5
WEIGHT_DECAY  = 1e-4
SAVE_EVERY    = 1_000
TEST_EVERY    = 5_000
NUM_EPOCHS    = 20

DEVICE        = "cuda"
POS_W         = 4.0         # still used for sampler weighting
NEG_W         = 1.0

# ───────────────────── dataset-level helpers / stats ────────────────────
def _count_rows(paths):
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)

DOCS            = _count_rows(SHARDS)
STEPS_PER_EPOCH = math.ceil(DOCS / BATCH)

# ────────────────────────────── dataset defs ────────────────────────────
@dataclass
class WaveSample:
    board:     torch.Tensor   # (C_IN, H, W)
    wave_id:   torch.Tensor   # long ∈ [0, 19]
    merc_feat: torch.Tensor   # scalar
    target:    torch.Tensor   # scalar ∈ [0, 1]

    @property
    def weight(self) -> float:
        return POS_W if self.target.item() > 0 else NEG_W


class WaveDataset(IterableDataset):
    """Streams every Parquet row exactly once per epoch, with augments."""

    def __init__(self, shard_paths: list[str], epoch_seed: int = 0):
        super().__init__()
        self.shard_paths = shard_paths
        self.epoch_seed  = epoch_seed                        # epoch ↦ seed

    # −−− helpers −−−
    @staticmethod
    def _safe_parse_id(_id: str):
        """
        Returns (wave:int, merc:float, fighters:str) or None on errors.
        """
        try:
            wave_part, merc_part, fighters_txt = (_id.split("|") + ["", ""])[:3]
            wave_num    = int(wave_part.split("=")[1])
            merc_bounty = float(merc_part.split("=")[1] or 0.0)
            return wave_num, merc_bounty, fighters_txt
        except (IndexError, ValueError):
            return None

    # −−− iterator −−−
    def __iter__(self):
        worker = get_worker_info()
        wid, wnum = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = np.random.default_rng(self.epoch_seed + wid)

        # mongodb client opened inside worker (safe after fork)
        cli = pymongo.MongoClient("mongodb://127.0.0.1:27017",
                                  maxPoolSize=1, connect=False)
        wave_val = {
            int(d["levelNum"]): float(d.get("totalReward", 0.0))
            for d in cli["legiontd2"]["waves"].find({}, {"levelNum": 1, "totalReward": 1})
        }

        # coordinate grids cached per resolution
        grid_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        for shard_path in self.shard_paths:
            pq_file = pq.ParquetFile(shard_path)
            for batch in pq_file.iter_batches(batch_size=2048):
                ids          = batch.column("_id").to_pylist()
                leaks        = batch.column("totalLeaked").to_numpy()
                total_occ    = batch.column("totalOccurrences").to_numpy()

                for _id, total_leak, total_occ in zip(ids, leaks, total_occ):
                    parsed = self._safe_parse_id(_id)
                    if not parsed:
                        print(f"failed to parse id: {_id}") # keep this print line always
                        continue
                    wave_num, merc_bounty, fighters_txt = parsed

                    base_reward = wave_val.get(wave_num, 0.0)
                    if base_reward <= 0.0:
                        print(f"negative reward: {base_reward} id: {_id}") # keep this print line always
                        continue

                    threat = base_reward + merc_bounty
                    if threat <= 0:
                        print(f"negative threat: {threat} id: {_id}") # keep this print line always
                        continue
                    frac = (total_leak / total_occ) / threat
                    if not (0.0 <= frac <= 1.0001):
                        print(f"bad fraction: {frac} id: {_id}") # keep this print line always
                        continue
                    if frac == 0.0 and rng.random() > 0.1:     # heavy down-sample
                        continue

                    board = torch.zeros((C_IN, BOARD_H, BOARD_W),
                                        dtype=torch.float32)

                    # build fighters
                    if fighters_txt:
                        for ent in fighters_txt.split(","):
                            uid, xs, ys, st, flag = ent.split(":")
                            ch = IDX.get(uid)
                            if ch is None:
                                print(f"ch is none: {uid}, fighters_txt: {fighters_txt}") # keep this print line always
                                continue
                            x = half_tile_coord_to_index(float(xs))
                            y = half_tile_coord_to_index(float(ys))
                            if not (0 <= x < BOARD_W-1 and 0 <= y < BOARD_H-1):
                                print(f"bad board coords: {BOARD_W},{BOARD_H} fighters_txt: {fighters_txt}") # keep this print line always
                                continue

                            board[ch, y:y+2, x:x+2] = 1.0          # presence

                            max_stack = STACK_LIMITS.get(uid, 2)
                            board[STACK_CH, y:y+2, x:x+2] = \
                                min(int(st), max_stack) / max_stack

                            sid = int(flag)
                            sch = spell_channel(sid)
                            if sch != -1:
                                board[sch, y:y+2, x:x+2] = 1.0

                    # positional channels
                    key = (BOARD_H, BOARD_W)
                    if key not in grid_cache:
                        xs = torch.linspace(0, 1, BOARD_W)
                        ys = torch.linspace(0, 1, BOARD_H)
                        grid_cache[key] = (xs.expand(BOARD_H, -1),
                                           ys.unsqueeze(1).expand(-1, BOARD_W))
                    gx, gy = grid_cache[key]
                    board[X_CH].copy_(gx)
                    board[Y_CH].copy_(gy)

                    # log-scaled merc-to-wave reward ratio
                    ratio      = merc_bounty / base_reward        # ≥ 0
                    log_ratio  = math.log1p(ratio)                # still ≥ 0

                    yield WaveSample(
                        board,
                        torch.tensor(wave_num - 1, dtype=torch.long),
                        torch.tensor(log_ratio, dtype=torch.float32),
                        torch.tensor(frac,      dtype=torch.float32),
                    )


# ─────────────────────────── Sampler / loaders ──────────────────────────
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


# ─────────────────── validation-only collate helper ────────────────────
def _collate_wave_samples(batch):
    """
    Convert a list of WaveSample objects into the same 4-tuple of tensors
    that OnTheFlyWeighted already produces for the train loader:
        boards      - (B, C_IN, H, W)
        wave_ids    - (B, 1)
        merc_feats  - (B, 1)
        targets     - (B, 1)
    """
    boards     = torch.stack([s.board      for s in batch])
    wave_ids   = torch.stack([s.wave_id    for s in batch]).unsqueeze(1)
    merc_feats = torch.stack([s.merc_feat  for s in batch]).unsqueeze(1)
    targets    = torch.stack([s.target     for s in batch]).unsqueeze(1)
    return boards, wave_ids, merc_feats, targets

def get_val_loader(batch_size=BATCH):
    ds = WaveDataset(VAL_SHARDS, epoch_seed=0)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH,
        persistent_workers=True,
        multiprocessing_context="spawn",
        collate_fn=_collate_wave_samples,   # ← **new line**
    )

def get_loader(epoch_seed: int):
    base = WaveDataset(SHARDS, epoch_seed)
    ds   = OnTheFlyWeighted(
        base,
        pos_w=POS_W,
        neg_w=NEG_W,
        num_samples=BATCH*STEPS_PER_EPOCH,
        seed=epoch_seed,
    )
    return DataLoader(
        ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=PREFETCH, persistent_workers=True,
        multiprocessing_context="spawn",
    )

# ──────────────────────────────── model ─────────────────────────────────
WAVE_MAX = 20

class WaveModel(nn.Module):
    """
    ConvNeXt-V2 encoder + learned rescale of merc-ratio.
    Single head → fractional leak in [0, 1] (after sigmoid).
    """
    def __init__(self):
        super().__init__()

        # image backbone
        self.cnn = timm.create_model(
            "convnextv2_tiny",
            patch_size=1,
            in_chans=C_IN,
            num_classes=0,
            pretrained=False,
            drop_path_rate=0.2,
        )
        self._divisor = 8    # for padding convenience

        # learned merc-ratio scale
        self.merc_proj = nn.Sequential(
            nn.Linear(1, 1),
            nn.Tanh(),          # output ∈ (−1, 1)
        )

        # wave embedding + joint features
        self.wave_emb = nn.Embedding(WAVE_MAX, 8)
        self.joint = nn.Sequential(
            nn.Linear(8 + 1 + 8, 16),   # w | m | w⊙m
            nn.GELU(),
            nn.Linear(16, 16),
        )

        # shared trunk
        self.shared = nn.Sequential(
            nn.Linear(self.cnn.num_features + 16, 512),
            nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.GELU(), nn.Dropout(0.25),
        )
        self.out_head = nn.Linear(128, 1)

    # −−− forward −−−
    def forward(self, board, wave_id, merc_ratio):
        # pad (H, W) to multiple of 8 so ConvNeXt works
        h, w = board.shape[-2:]
        pad_h = (self._divisor - h % self._divisor) % self._divisor
        pad_w = (self._divisor - w % self._divisor) % self._divisor
        if pad_h or pad_w:
            board = F.pad(board, (0, pad_w, 0, pad_h))

        z_img = self.cnn(board)                         # (N, F)

        w = self.wave_emb(wave_id.squeeze(-1))          # (N, 8)
        m = self.merc_proj(merc_ratio)                  # (N, 1)

        z_joint = self.joint(torch.cat([w, m, w*m], dim=1))
        feats   = self.shared(torch.cat([z_img, z_joint], dim=1))
        return self.out_head(feats)                     # raw logit (N, 1)

# ───────────────────────── checkpoint helpers ───────────────────────────
def save_ckpt(model, opt, step: int):
    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)
    torch.save(
        {"model": model.state_dict(),
         "optim": opt.state_dict(),
         "step":  step},
        CKPT_PATH,
    )


def load_ckpt(model, opt, *, lr_override: float | None = None) -> int:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)   # old dual-head OK
    opt.load_state_dict(ckpt["optim"])
    if lr_override is not None:
        for pg in opt.param_groups:
            pg["lr"] = lr_override
    return ckpt.get("step", 0)

# ───────────────────────────── training loop ────────────────────────────
def train():
    model = WaveModel().to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    step = 0
    if os.path.isfile(CKPT_PATH):
        try:
            step = load_ckpt(model, opt, lr_override=LR)
            print(f"Resumed from {CKPT_PATH} at step {step}")
        except Exception:
            print("Checkpoint present but failed to load; restarting")
            step = 0

    writer        = SummaryWriter(LOG_DIR)
    running_loss  = running_mae = 0.0
    base_seed     = random.randint(0, 2**31 - 1)

    for epoch in range(NUM_EPOCHS):
        loader = get_loader(base_seed + epoch)

        for boards, wave_ids, merc_feats, targets in loader:
            boards, wave_ids, merc_feats, targets = (
                boards.to(DEVICE, non_blocking=True),
                wave_ids.to(DEVICE, non_blocking=True),
                merc_feats.to(DEVICE, non_blocking=True),
                targets.to(DEVICE, non_blocking=True).squeeze(),
            )

            opt.zero_grad(set_to_none=True)
            logit = model(boards, wave_ids, merc_feats).squeeze()  # (N,)
            pred_frac = torch.sigmoid(logit)

            loss = F.l1_loss(pred_frac, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            # metrics
            running_loss += loss.item()
            step += 1

            # log / checkpoint
            if step % SAVE_EVERY == 0:
                k = SAVE_EVERY
                writer.add_scalar("train/loss_avg", running_loss / k, step)
                writer.add_scalar("train/lr",      opt.param_groups[0]["lr"], step)
                running_loss = 0.0
                save_ckpt(model, opt, step)

            if step % TEST_EVERY == 0:
                model.eval()
                mae_sum, n = 0.0, 0
                with torch.no_grad():
                    for b, w, m, t in get_val_loader():
                        b = b.to(DEVICE, non_blocking=True)
                        w = w.to(DEVICE, non_blocking=True)
                        m = m.to(DEVICE, non_blocking=True)
                        t = t.to(DEVICE, non_blocking=True).squeeze()

                        fp = torch.sigmoid(model(b, w, m).squeeze())
                        mae_sum += (fp - t).abs().sum().item()
                        n += t.numel()

                writer.add_scalar("train/validate loss %", 100 * mae_sum / n, step)
                model.train()

        print(f"epoch {epoch+1} complete")

    save_ckpt(model, opt, step)

# ────────────────────────────── entry point ─────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    try:
        train()
    except Exception:
        logging.exception("Training crashed hard")
        raise

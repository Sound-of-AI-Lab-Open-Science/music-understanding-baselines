"""MIDI-RAE-JEPA embedding model: wraps our trained hierarchical Swin V2 encoder
(https://github.com/drscotthawley/midi-rae) for BenchMIR.

Converts a raw MIDI file into a tempo-normalized 32nd-note binary piano roll
(matching training-time preprocessing in midi-rae/scripts/midi_to_pianoroll.py),
slides a 128x128 window across it, runs each crop through the frozen
SwinEncoder, and aggregates patch embeddings per hierarchy level.

`level` selects which hierarchy level feeds the output embedding:
    "concat" -> concatenate mean-pooled embeddings from all 6 levels
    0-5      -> just that level's mean-pooled embedding (0=coarsest L0, 5=finest L5,
                matching HierarchicalPatchState.levels ordering: coarsest first)

Song-level (run/run_batch): mean-pool over all sliding-window crops -> one
fixed vector per song. Used for CIPI / EMOPIA / TopMAGD / Humdrum.

Frame-level (run_frames, for POP909 chord/root/key): keeps every patch of the
chosen level as one frame, across every crop, with real start/end times
computed from the song's tempo and the patch's column position within the
crop plus the crop's pixel offset within the song. For level="concat", frames
are taken from the finest level (L5) since coarse levels' patches span
multiple bars and aren't meaningful as time-aligned frames.

Needs the midi_rae package importable (installed editable from the sibling
midi-rae repo checkout) and pretty_midi.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pretty_midi
import torch

from benchmir.eval.feature_extraction.embeddings.embedding_extractor import (
    EmbeddingExtractor,
)
from benchmir.eval.feature_extraction.embeddings.frame_embedding_extractor import (
    FrameEmbeddingExtractor,
)
from benchmir.models.base import Model

from midi_rae.swin import SwinEncoder
from midi_rae.utils import load_checkpoint

STEPS_PER_BEAT = 8  # matches midi-rae/scripts/midi_to_pianoroll.py training-time quantization
MAX_LEN = 4096  # same cap as training preprocessing (~512 beats) -- without this, a long/dense
                # song can produce thousands of crops and take minutes per file on CPU
LEVEL_DIMS = [256, 128, 64, 32, 16, 8]  # L0(coarsest)..L5(finest), matches levels[] order
LEVEL_GRID = [1, 2, 4, 8, 16, 32]  # patches-per-crop-side at each level; LEVEL_DIMS[i]*LEVEL_GRID[i]==256 always


def midi_to_roll(
    path: str | Path,
    steps_per_beat: int = STEPS_PER_BEAT,
    max_len: int = MAX_LEN,
    align_downbeat: bool = False,
):
    """Tempo-normalized binary piano roll + columns-per-second, identical
    recipe to midi-rae's training-time preprocessing (including the same
    max_len truncation, so eval-time embeddings see the same kind of input
    distribution the encoder was actually trained on).

    align_downbeat: if True, trim leading columns so column 0 lands exactly
    on the song's first downbeat (from pretty_midi's beat/meter estimate)
    instead of on the file's raw start time. For a constant-tempo 4/4 song
    this puts every bar boundary on a multiple of 32 columns (8 steps/beat x
    4 beats/bar), and since the finest patch width (4 columns) divides that
    evenly, every patch tile in every sliding-window crop lands on a
    bar-phase-consistent boundary -- crops always start at stride multiples
    of 64 = 2 bars, preserving phase. Tests whether MIDI-RAE-JEPA's
    frame-level POP909 underperformance vs. MuseTok (which tokenizes
    bar-wise) is explained by MIDI-RAE-JEPA's patches having no bar-phase
    awareness at all (this flag), as opposed to being inherent to the
    JEPA/predictive training objective discarding fine local detail
    regardless of alignment. Only meaningful for 4/4, roughly constant-tempo
    songs; a meter change or pickup measure mid-song isn't corrected for."""
    pm = pretty_midi.PrettyMIDI(str(path))
    end_time = pm.get_end_time()
    if end_time <= 0:
        raise ValueError(f"{path}: zero-length MIDI")
    tempi = pm.get_tempo_changes()[1]
    try:
        tempo = float(tempi[0]) if len(tempi) else pm.estimate_tempo()
    except Exception:
        tempo = 120.0
    if not (20 <= tempo <= 300):
        tempo = 120.0
    fs = (tempo / 60.0) * steps_per_beat
    start_time = 0.0
    if align_downbeat:
        try:
            downbeats = pm.get_downbeats()
        except Exception:
            downbeats = np.array([])
        if len(downbeats) and downbeats[0] < end_time:
            start_time = float(downbeats[0])
    roll = pm.get_piano_roll(fs=fs)
    roll = (roll > 0).astype(np.uint8)
    if start_time > 0:
        start_col = int(round(start_time * fs))
        roll = roll[:, start_col:]
    if roll.shape[1] > max_len:
        roll = roll[:, :max_len]
    return roll, fs, start_time


class MidiRaeJepaEmbeddingModel(Model):
    """Song-level or frame-level embeddings from a frozen midi-rae-jepa SwinEncoder.

    Constructor kwargs (beyond the base Model contract):
        level:      "concat" or 0-5 (see module docstring)
        frame_mode: if True, build_extractor() returns a FrameEmbeddingExtractor
                    (for POP909); if False, a song-level EmbeddingExtractor.
                    A given model_id should only ever be used for one or the
                    other (mirrors the "aria" vs "Dummy" split in the example
                    config) -- make a separate model_id per (level, frame_mode)
                    combination you want to evaluate.
        crop_size, stride: sliding-window params over the song's piano roll.
    """

    default_batch_size = 1

    def __init__(
        self,
        model_id: str = "midi-rae-jepa",
        checkpoint_path: str | Path = "",
        level: str | int = "concat",
        frame_mode: bool = False,
        device: str = "cpu",
        supported_formats: list[str] | None = None,
        batch_size: int | None = None,
        crop_size: int = 128,
        stride: int = 64,
        align_downbeat: bool = False,
        pitch_pool: str = "mean",
    ):
        level = level if level == "concat" else int(level)
        if frame_mode:
            lvl_idx = 5 if level == "concat" else level
            feature_dim = LEVEL_DIMS[lvl_idx] if pitch_pool == "mean" else LEVEL_DIMS[lvl_idx] * LEVEL_GRID[lvl_idx]
        else:
            feature_dim = sum(LEVEL_DIMS) if level == "concat" else LEVEL_DIMS[level]
        super().__init__(
            model_id, checkpoint_path, feature_dim,
            supported_formats or ["midi"], batch_size=batch_size,
        )
        self.level = level
        self.frame_mode = frame_mode
        self.device = torch.device(device)
        self.crop_size, self.stride = crop_size, stride
        self.align_downbeat = align_downbeat
        assert pitch_pool in ("mean", "concat")
        self.pitch_pool = pitch_pool

        self.encoder = SwinEncoder(
            img_height=128, img_width=128, patch_h=4, patch_w=4,
            embed_dim=8, depths=[2, 2, 2, 6, 2, 1], num_heads=[2, 2, 2, 4, 8, 16],
            window_size=4, mlp_ratio=4.0, drop_path_rate=0.0,
        ).to(self.device)
        self.encoder = load_checkpoint(self.encoder, str(checkpoint_path))
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def _crop_locs(self, w: int) -> list[int]:
        return list(range(0, max(w - self.crop_size, 0) + 1, self.stride)) or [0]

    def _crop_batches(self, roll: np.ndarray, max_batch: int = 64):
        h, w = roll.shape
        if w < self.crop_size:
            roll = np.pad(roll, ((0, 0), (0, self.crop_size - w)))
            w = self.crop_size
        locs = self._crop_locs(w)
        for i in range(0, len(locs), max_batch):
            batch_locs = locs[i:i + max_batch]
            crops = np.stack([roll[:, loc:loc + self.crop_size] for loc in batch_locs])
            yield batch_locs, crops

    @torch.no_grad()
    def _embed_batch(self, crops: np.ndarray):
        img = torch.from_numpy(crops.astype(np.float32)).unsqueeze(1).to(self.device)  # (B,1,H,W)
        out = self.encoder(img)
        return out.patches.levels  # [L0(coarsest), L1, L2, L3, L4, L5(finest)], each emb (B,N,D)

    def run(self, raw_score: Path) -> np.ndarray:
        try:
            roll, _fs, _start = midi_to_roll(raw_score, align_downbeat=self.align_downbeat)
        except Exception as e:
            print(f"WARNING: unparseable MIDI {raw_score}: {e} -- returning zero embedding")
            return np.zeros(self.feature_dim, dtype=np.float32)
        sums, n = None, 0
        for _locs, crops in self._crop_batches(roll):
            levels = self._embed_batch(crops)
            # mean over patches, then sum over the batch -> per-level (dim,)
            vecs = [lvl.emb.mean(dim=1).sum(dim=0).cpu().numpy() for lvl in levels]
            if sums is None:
                sums = [np.zeros_like(v) for v in vecs]
            for i, v in enumerate(vecs):
                sums[i] += v
            n += crops.shape[0]
        means = [s / n for s in sums]
        if self.level == "concat":
            return np.concatenate(means).astype(np.float32)
        return means[self.level].astype(np.float32)

    def run_batch(self, raw_scores: list[Path]) -> np.ndarray:
        return np.stack([self.run(s) for s in raw_scores])

    def run_frames(self, raw_score: Path) -> list[tuple[np.ndarray, float, float]]:
        try:
            roll, fs, start_time = midi_to_roll(raw_score, align_downbeat=self.align_downbeat)
        except Exception as e:
            print(f"WARNING: unparseable MIDI {raw_score}: {e} -- returning no frames")
            return []
        lvl_idx = 5 if self.level == "concat" else self.level  # concat -> finest (L5) for frame timing
        frames: list[tuple[np.ndarray, float, float]] = []
        for locs, crops in self._crop_batches(roll):
            levels = self._embed_batch(crops)
            lvl = levels[lvl_idx]
            emb = lvl.emb.cpu().numpy()  # (B, N, dim)
            pos = lvl.pos.cpu().numpy()  # (N, 2) row,col grid coords, shared across the batch
            cols = pos[:, 1]
            rows = pos[:, 0]
            unique_cols = np.unique(cols)
            grid = len(unique_cols)
            patch_w_px = self.crop_size / grid
            # One frame per TIME-column, aggregated across the pitch (row) dimension --
            # chord/root/key labels are per time-window, not per (pitch-band, time) patch.
            # Without *some* aggregation, every pitch-row at a given time-column would be
            # emitted as its own "frame" sharing identical start/end times (grid x too many
            # frames, e.g. 32x at the finest level -- caught when POP909 chord extraction
            # produced 36.7M rows for 909 songs instead of the ~1M expected).
            # pitch_pool="mean" (original) collapses all pitch-rows into one embed_dim-sized
            # vector by averaging -- cheap, but discards which specific pitches were active.
            # pitch_pool="concat" instead concatenates the pitch-rows in a fixed row order,
            # giving a (grid*embed_dim)-sized vector that preserves per-pitch-band identity;
            # grid*embed_dim==256 at every level by construction, so this is a fair
            # equal-capacity comparison across levels (see LEVEL_GRID).
            for b, loc in enumerate(locs):
                for col in unique_cols:
                    mask = cols == col
                    if self.pitch_pool == "mean":
                        pooled = emb[b, mask].mean(axis=0)
                    else:
                        order = np.argsort(rows[mask])
                        pooled = emb[b, mask][order].reshape(-1)
                    start_px = loc + col * patch_w_px
                    end_px = start_px + patch_w_px
                    # start_time > 0 when align_downbeat trimmed leading columns off the roll --
                    # patch column positions are relative to the trimmed roll, so add it back to
                    # get absolute song time matching the label timestamps.
                    frames.append((pooled.astype(np.float32), start_time + start_px / fs, start_time + end_px / fs))
        return frames

    def build_extractor(self, cache_dir: Path | None = None):
        if self.frame_mode:
            return FrameEmbeddingExtractor(self, cache_dir=cache_dir, batch_size=self.batch_size)
        return EmbeddingExtractor(self, cache_dir=cache_dir, batch_size=self.batch_size)

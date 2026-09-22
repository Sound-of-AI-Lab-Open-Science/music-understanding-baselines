# Third-party notices

This package is MIT licensed (see `LICENSE`). It depends on code, model weights
and corpora that are **not** ours and are **not** redistributed here. This file
records what each one is, where it comes from, and what its terms are.

**No corpus and no model weights are redistributed here.** Third-party *source*
appears in exactly three places, each covered below:

* `pretrain/src/data/octuple.py` copies part of the OctupleMIDI codec from
  Microsoft's `muzic` (MIT). Its licence is reproduced in the file's header and
  in full at `third_party/licenses/muzic-LICENSE`.
* `third_party/patches/benchmir_pop909cl_eval_corpus.patch` carries a few lines
  of the evaluation library's source as diff context (MIT, reproduced at
  `third_party/licenses/BenchMIR-LICENSE`; see below).
* `third_party/BenchMIR` is a git submodule — a URL and a commit, not a copy.

---

## BenchMIR — the evaluation library

* **Attribution**: BenchMIR (Broto Clemente)
* **Licence**: MIT. Reproduced in full at
  `third_party/licenses/BenchMIR-LICENSE`, which is vendored here rather than
  left to the submodule: the submodule may not resolve for you (see
  *Availability* below), and `third_party/patches/` carries BenchMIR's source
  as diff context regardless, so a checkout without the submodule still needs
  the licence text.
* **Upstream merge point**: the fork's `jepa-reproduce` branch starts from
  upstream BenchMIR commit `87349a4`. A few report makers name that commit when
  explaining which published numbers predate it.
* **How it is used**: `third_party/BenchMIR` is a git submodule pinned to a
  fork of the upstream project at a specific commit on the branch
  `jepa-reproduce`. `setup.sh` checks it out and installs it editable into the
  evaluation environment. This package does not vendor, modify in place, or
  redistribute the library's source; the submodule is a pointer.
* **Our changes on that branch**, all additive except the last:
  * `src/benchmir/models/ours_{common,jepa,musetok,musicbert}.py` — the three
    model adapters and the shared bar-embedding / content-hash cache layer.
  * `src/benchmir/models/ours_workers/**` — the per-environment subprocess
    workers the adapters spawn, plus the framed stdin/stdout protocol they
    speak. These exist because the three checkpoints' dependencies are mutually
    exclusive; see *Why three environments* in `README.md`.
  * `src/benchmir/models/registry.py` — the registration lines for the above.
  * `src/benchmir/eval/datasets/corpora/pop909cl_eval_corpus.py` — a
    modification, not an addition: it makes the hard-coded 30-song POP909 slice
    configurable (`kwargs: {max_songs: all}`), with the default left at 30 so
    previously published chord/root numbers stay reproducible. The same change
    is kept in `third_party/patches/` as readable documentation of what the
    branch does to an upstream file.
* **Availability**: the submodule points at a fork of a third-party
  repository, and whether that fork is published is the BenchMIR authors'
  decision, not this package's. Treat the submodule URL as possibly
  unreachable: `setup.sh` warns and carries on rather than failing,
  `run_pretrain.sh` and everything else in the pre-training half run without
  it, and only `run_eval.sh` / `run_report.sh` need it. If the URL does not
  resolve for you, request access from the BenchMIR authors.

### A provenance caveat worth reading before comparing numbers

The evaluation library does **not** seed its RNGs: it parses `run.seed` and
never applies it to torch or numpy. Every probe result is therefore ONE unseeded
fit, and every `±` reported by the report makers is across FOLDS, not across
seeds. Treat a difference smaller than the fold spread as no difference.

---

## MuseTok — upstream tokenizer

* **Upstream**: <https://github.com/Yuer867/MuseTok>
* **Pin**: `7b71d16868431331386b1c1088193b3b85bd70e2`
* **Licence status**: the upstream repository publishes **no licence file**. In
  the absence of a licence grant, the default is that no redistribution rights
  are given. This package therefore **neither vendors nor submodules it**:
  `setup.sh --with-musetok` clones it from upstream, at the pin above, into
  `third_party/MuseTok`, which is git-ignored here. Running that step is your
  decision and is governed by upstream's terms, not by this package's licence.
* **What needs it**: the MuseTok arm only — its REMI+ tokenization
  (`pretrain/baselines/data/remi_cache.py`), its training wrapper
  (`pretrain/baselines/models/musetok_module.py`), and the MuseTok worker in
  the evaluation submodule. The other three arms are unaffected.
* **The stated policy is to call upstream, not reproduce it**, and the package
  keeps to that with two named exceptions. Both are de minimis functional code,
  but both are upstream's logic sitting inside this MIT-licensed tree, so they
  are recorded here rather than left implicit:
  * `pretrain/baselines/data/musetok_datamodule.py::_bar_pos_one` reproduces
    the per-piece body of upstream's `REMIEventDataset.build_dataset` (a
    handful of lines of bar-boundary fix-ups) rather than calling the method,
    because the cache builder needs it per piece and out of process.
  * `pretrain/baselines/models/musetok_module.py` is a Lightning wrapper around
    upstream's network, and reproduces upstream's training recipe rather than
    calling its hand-rolled loop: the sequence-first tensor layout its
    `forward` feeds the model, the `recons_ce + beta * commit_loss` objective,
    the Adam settings, and the warmup-then-cosine LR schedule. The network, the
    loss function and the encoder itself are still upstream's own code, called
    from the cloned checkout — none of it is copied into this tree.
* **Released weights and vocabulary** (upstream's `ckpt/` and
  `data/dictionary.pkl`) are also upstream's and are not redistributed.
* **Compatibility warning**: this package trains MuseTok against a
  CORPUS-DERIVED vocabulary, because upstream's released 168-token dictionary
  cannot express roughly a quarter of a broad symbolic corpus. Token ids shift,
  so checkpoints produced here are **not** vocabulary-compatible with the public
  MuseTok weights, and an adapter that assumes `n_token == 168` must never be
  pointed at them.

---

## MusicBERT / `muzic` — Zeng et al., 2021

* **Upstream**: <https://github.com/microsoft/muzic>
* **Pin**: `2b8739671ba06f819f31f568b8a79da581aaf6f9`
* **Licence**: MIT, `Copyright (c) Microsoft Corporation.`, reproduced in full
  at `third_party/licenses/muzic-LICENSE` and, for the one file that copies
  from it, in that file's own header.
* **Attribution**: the architecture, the OctupleMIDI encoding and the official
  pre-training recipe are from *MusicBERT: Symbolic Music Understanding with
  Large-Scale Pre-Training* (Zeng et al., 2021), released by its authors under
  the MIT licence as part of `muzic`.
* **One file copies upstream source**: `pretrain/src/data/octuple.py` is a port
  of `musicbert/preprocess.py`, and parts of it — the constants block and
  several stretches of the encoding path — are literally upstream's lines. It
  carries the MIT copyright and permission notice in its header, as the licence
  requires.
* **These files are re-implementations, not copies.** They follow upstream's
  architecture, recipe or conventions, but share no literal lines with it:
  `pretrain/src/models/musicbert.py` (the OctupleEncoder + RoBERTa LM head, no
  fairseq and no transformers, written so the released weights load 1:1),
  `pretrain/src/data/vocab.py`, `pretrain/src/data/mlm_datamodule.py`,
  `pretrain/scripts/train_mlm.py` and `pretrain/src/tasks/mlm_pretrain.py`.
* **The OctupleMIDI dictionary** `pretrain/src/data/vocab_octuple.txt` is not a
  copied file: it is regenerated by `vocab.py`'s `__main__` from the constants
  in `octuple.py`, following upstream's `gen_dictionary` spec. It comes out
  byte-for-byte identical to upstream's released fairseq `dict.txt`, which is
  the point — that is what makes the released checkpoint's embedding rows line
  up without a permutation.
* **Released weights** are the original authors' and are not redistributed.
  Converting one into the `{"config", "state_dict"}` blob this
  re-implementation loads is left to you; `pretrain/src/utils/fairseq_ckpt.py`
  provides the half that is genuinely awkward — reading a fairseq `.pt` without
  fairseq or omegaconf installed — but the key mapping onto `musicbert.py` is
  not shipped and no script in this package writes `$MUSICBERT_CKPT`.

---

## fairseq — Meta Platforms

* **Upstream**: <https://github.com/facebookresearch/fairseq>
* **Licence**: MIT, `Copyright (c) Facebook, Inc. and its affiliates.`
* **How it is used**: not at all at runtime — fairseq is not a dependency of
  this package. Two files build on its conventions without containing its code:
  `pretrain/src/models/musicbert.py` mirrors `TransformerSentenceEncoder` and
  fairseq's `Dictionary` / padding-offset conventions, and
  `pretrain/src/utils/fairseq_ckpt.py` fabricates placeholder `fairseq` and
  `omegaconf` modules so a released checkpoint can be unpickled without them.

---

## musicbert_hf — Sailor

* **Upstream**: <https://github.com/malcolmsailor/musicbert_hf>
* **Pin**: `64a054c791c2372e146d59cb0b04a5f5eba23fc0`
* **Licence**: MIT
* **How it is used**: consulted, not copied. Its verified fairseq→HF key
  mapping was used to cross-check `pretrain/src/models/musicbert.py`, and the
  fairseq vocabulary file it ships as canonical was used to verify that
  `pretrain/src/data/vocab_octuple.txt` regenerates identically.

---

## Fundamental Music Embedding — Guo et al., 2023

* **Upstream**: <https://github.com/guozixunnicolas/FundamentalMusicEmbedding>
* **Pin**: `793e30079978c859afef73ff4b88d7001bfc5b57`
* **Licence status**: the upstream repository publishes **no licence file**, so
  no redistribution rights are granted and nothing is copied from it.
* **How it is used**: `pretrain/src/models/fme.py` is an independent
  implementation written from the paper (*A Domain-Knowledge-Inspired Music
  Embedding Space and a Novel Attention Mechanism for Symbolic Music Modeling*,
  AAAI 2023) and verified to agree numerically with upstream's formulation. It
  shares no literal lines with the upstream file. Upstream's published
  `ripo_transformer.yaml` defaults are cited as hyper-parameter provenance.

---

## Music-JEPA — the architecture being reproduced

No code was released with the paper. `pretrain/src/models/jepa.py` and
`pretrain/src/tasks/jepa_pretrain.py` are ours, written from the paper's
description; `pretrain/baselines/configs/jepa_paper.yaml` records which reading
of the paper each hyper-parameter comes from, including the two that are easy
to get wrong (absolute *and* relative position encoding; unweighted VICReg).

Supporting methods implemented from their papers, with no upstream code
involved: Fourier metric embedding (Guo et al., 2023 — see the FME section
above), relative self-attention (Shaw et al., 2018; Huang et al., 2020), VICReg
(Bardes et al., 2022), SIGReg / LeJEPA (arXiv:2511.08544, in
`pretrain/src/utils/sigreg.py`) and the shift-equivariance objective of
MIDI-RAE-JEPA (arXiv:2607.14537, in `pretrain/src/utils/equivariance.py`).

---

## Corpora

None of these is redistributed with this package. Each is obtained from its own
source, under its own terms, by whoever runs the pipeline. `check_data.sh`
verifies the layout without downloading anything.

| corpus | used for | terms |
|---|---|---|
| **CIPI** | piano difficulty (Henle grades) | research use; request from its publishers |
| **EMOPIA** | emotion recognition (4Q) | CC BY-NC-SA, non-commercial |
| **POP909** | chord, chord-root and key estimation | research use, per its own `LICENSE` file |
| **Humdrum / KernScores** | composer classification | per the KernScores collection's terms |
| **Lakh MIDI (matched) + TopMAGD** | genre classification | LMD is CC BY 4.0; the TopMAGD genre and partition files come from their own publishers, with their own terms |
| **MuseScore-derived MIDI collection** | pre-training | crowd-uploaded scores; the collection's own terms apply, and they are not uniform across files |

A word about the last row, because it affects how the results should be read:
a large crowd-uploaded pre-training corpus can overlap by CONTENT with a named
evaluation corpus even when no identifier is shared, since the same piece is
uploaded many times under different titles. The split this package builds is
keyed on file content precisely because id-keying leaked held-out test pieces
into training. That removes the leak *within* the pre-training corpus; it says
nothing about overlap between the pre-training corpus and the evaluation
corpora, which remains a real and unmeasured risk for any such corpus.

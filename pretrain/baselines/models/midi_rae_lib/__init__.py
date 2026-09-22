"""Ported subset of the upstream midi-rae package (drscotthawley/midi-rae),
vendored here rather than imported from a live checkout because upstream is
notebook-generated (nbdev) and unpublished on PyPI. See THIRD_PARTY_NOTICES.md.

Only the pieces the ``swin`` encoder/decoder path needs are kept: the ViT
encoder, the MAE decoder and PNG-backed datasets are dropped because neither
midi_rae_enc.yaml nor midi_rae_dec.yaml (translated from the paper's own
config_swin_full_backup.yaml) uses them (``model.encoder: swin`` and
``training.lambda_mae: 0.0`` throughout).
"""

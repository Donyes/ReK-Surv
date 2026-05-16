# Active Context

- Updated: 2026-05-16T00:00:00+08:00
- Current focus: Running trigger_orchard_v3 T1-A dropout regularization under the existing shared-split full5 hold-out protocol.
- Latest outcome: Added a `--v3_dropout` switch for trigger_orchard_v3 MLP blocks while preserving default behavior at 0.0; `train_dynamic.py --help`, py_compile, and a 1-epoch v3 delta+LOD smoke run with `--v3_dropout 0.1` succeeded.
- Main blocker: The full5 dropout scan still needs to be run for p values such as 0.1, 0.2, and 0.3, then judged against train/val/test split gaps, test Ctd, and the unstable 8:5 Bstd window.

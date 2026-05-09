# Active Context

- Updated: 2026-05-06T02:18:11+08:00
- Current focus: Confirmed trigger_orchard_v3 overfitting from split-wise Ctd/Bstd
- Latest outcome: This session added split-aware result recording for dynamic experiments and used train, validation, and test Ctd/Bstd to diagnose generalization. Stable result: the trigger_orchard_v3 delta+LOD original full5 run shows a clear train-versus-val/test gap, so the current issue is real overfitting rather than a reporting bug.
- Main blocker: The trigger_orchard_v3 branch has a confirmed train-versus-val/test generalization gap on the 123-sample regime, and any next variant must reduce that gap without sacrificing test Ctd.

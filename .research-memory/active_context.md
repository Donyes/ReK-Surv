# Active Context

- Updated: 2026-05-16T19:27:08+08:00
- Current focus: Interpreting trigger_orchard_v3 T1-A dropout regularization under the existing shared-split full5 hold-out protocol.
- Latest outcome: Completed `--v3_dropout` scan at p=0.1/0.2/0.3 for trigger_orchard_v3 delta+LOD. Dropout reduced train-test Ctd gap monotonically, and p=0.3 raised test Ctd modestly to 0.5971 versus baseline 0.5823, but all dropout settings worsened test Bstd to about 0.314 and worsened 8:5 Bstd to 0.859-0.915 versus baseline 0.6292.
- Main blocker: T1-A dropout is not sufficient as the anti-overfitting fix because calibration, especially the 8:5 window, remains unstable or worse despite a smaller train-test gap.

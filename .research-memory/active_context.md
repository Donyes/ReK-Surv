# Active Context

- Updated: 2026-05-05T21:23:54+08:00
- Current focus: trigger_orchard_v3 simplification and CT ablation on fixed windows
- Latest outcome: Implemented trigger_orchard_v3 with 10-d daily inputs and no period_env, then completed full5 shared-split comparisons on windows 4:3 4:5 6:3 6:5 8:3 8:5. Stable result: v3 beats v2 on the original dataset, is roughly tied on the tuned latefix3 dataset, and CT auxiliary supervision improves Ctd but not calibration.
- Main blocker: No single trigger_orchard variant currently dominates both ranking and calibration across both datasets, so the next thread must decide whether the paper prioritizes mean Ctd or Brier/calibration.

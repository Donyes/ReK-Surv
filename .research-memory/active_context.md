# Active Context

- Updated: 2026-05-06T00:00:00+08:00
- Current focus: Positioning trigger_orchard_v3 delta+endpoint-LOD within the fixed-window CT ablation line.
- Latest outcome: Replaced the failed v3 endpoint-CT-value head with a hybrid global CT delta vector + window endpoint LOD classifier, then reran the original-dataset shared-split full5 experiment on windows 4:3 4:5 6:3 6:5 8:3 8:5. Stable result: this delta+LOD variant is now the best CT-enabled trigger_orchard_v3 setting on the original dataset, slightly improving the old v3 delta baseline overall (mean Ctd 0.5823 vs 0.5804; mean Brier/Bstd 0.2691 vs 0.2855), but it still does not beat v3-noCT on calibration (mean Bstd 0.2483).
- Main blocker: The trigger_orchard_v3 line still has no variant that simultaneously dominates ranking and calibration; delta+LOD helps enough to become the CT-enabled default, but it does not resolve whether the paper-facing trigger_orchard comparison should favor CT-enabled v3 or v3-noCT.

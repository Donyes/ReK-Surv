# Next Steps

- Run the trigger_orchard_v3 delta+LOD shared-split full5 dropout scan with `--v3_dropout 0.1`, `0.2`, and `0.3`, keeping the current training protocol and fixed split JSON unchanged.
- Compare each dropout variant against the existing baseline by train/val/test Ctd/Bstd, per-window test metrics, and especially whether 8:5 Bstd improves without sacrificing test Ctd.

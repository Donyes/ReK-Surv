# Next Steps

- Treat T1-A dropout as a partial ranking/gap improvement rather than a solved calibration fix; do not pick p=0.1 or p=0.2 as final variants because they are essentially flat on test Ctd and worse on Bstd.
- If continuing the anti-overfitting line, use p=0.3 only as a candidate for a ranking-first branch, then test a calibration-oriented add-on such as patience=10 or weight_decay=5e-4 while explicitly gating on 8:5 Bstd.

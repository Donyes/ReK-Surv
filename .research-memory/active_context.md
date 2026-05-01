# Active Context

- Updated: 2026-05-01T15:57:58.5925775+08:00
- Current focus: trigger_orchard_v2 calibration tuning after the first gamma_ct increase test
- Latest outcome: Doubling `gamma_ct` from 0.10 to 0.20 in the matched 5-repeat `trigger_orchard_v2` run improved the validation CT auxiliary MAE (about 4.02 -> 3.86) and sharply reduced test BStd on the hardest matched window `9->3` (about 1.60 -> 0.77), but it also reduced overall matched-window test Ctd (about 0.655 -> 0.469), especially on `6->1`, `6->3`, and `9->1`.
- Main blocker: Stronger CT supervision helps calibration but over-regularizes ranking, so the next tuning step should avoid another large global increase and instead test a milder `gamma_ct` or a cleaner CT target.

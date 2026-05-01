# trigger_orchard_v2 CT-vector gamma_ct scan

Compared runs:

- `artifacts/dynamic_hlb_trigger_v2_ctvector_full5` (`gamma_ct=0.1`)
- `artifacts/dynamic_hlb_trigger_v2_ctvector_gammact050_full5` (`gamma_ct=0.5`)
- `artifacts/dynamic_hlb_trigger_v2_ctvector_gammact100_full5` (`gamma_ct=1.0`)

All three runs use the same `trigger_orchard_v2` window-prefix CT-vector head (`ct_delta_output_dim=11`) with `use_ct_aux_task`, `use_agro_features`, and 5 repeats.

## Aggregate comparison

| gamma_ct | overall mean Ctd | overall mean BStd | mean best val Ctd | mean best val CT MAE |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 0.5140 | 0.3235 | 0.7472 | 3.6590 |
| 0.5 | 0.4741 | 0.3500 | 0.6787 | 3.7314 |
| 1.0 | 0.5838 | 0.3679 | 0.7167 | 3.7197 |

## Main windows

| gamma_ct | 6->3 Ctd | 6->3 BStd | 9->3 Ctd | 9->3 BStd |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 0.4067 | 0.1693 | 0.6564 | 1.0163 |
| 0.5 | 0.3582 | 0.1998 | 0.5826 | 1.0667 |
| 1.0 | 0.4689 | 0.2093 | 0.6936 | 1.1235 |

## Interpretation

- `gamma_ct=0.5` is not competitive. It is worse than `0.1` on overall Ctd, overall BStd, validation CT MAE, `6->3`, and `9->3`.
- `gamma_ct=1.0` is the best choice only if the main objective is ranking on the trigger windows: it improves overall mean Ctd and both `6->3` / `9->3` Ctd relative to `0.1`.
- `gamma_ct=0.1` remains the better balanced choice for the CT-vector head if calibration and CT fit matter: it has the best overall BStd and the best validation CT MAE.
- Even with `gamma_ct=1.0`, the CT-vector head still does not clearly beat the older scalar-head trigger_orchard_v2 baseline on ranking, so this scan does not justify replacing the scalar path as the default.

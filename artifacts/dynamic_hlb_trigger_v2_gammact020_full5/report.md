# trigger_orchard_v2 gamma_ct=0.20

Baseline: `artifacts/dynamic_hlb_trigger_v2_full5` with `gamma_ct=0.10`

Changed setting:

- `gamma_ct`: `0.10 -> 0.20`
- All other settings matched the baseline `trigger_orchard_v2` 5-repeat run on windows `6->1`, `6->3`, `9->1`, `9->3`

Aggregate effect:

| metric | gamma_ct=0.10 | gamma_ct=0.20 | delta |
| --- | ---: | ---: | ---: |
| overall mean Ctd | 0.6551 | 0.4692 | -0.1859 |
| overall mean BStd | 0.4816 | 0.2569 | -0.2247 |
| mean best validation Ctd | 0.7359 | 0.7804 | +0.0445 |
| mean best validation CT MAE | 4.0181 | 3.8596 | -0.1585 |
| mean best epoch | 10.8 | 6.0 | -4.8 |

Window-level test means:

| window | Ctd 0.10 | Ctd 0.20 | delta | BStd 0.10 | BStd 0.20 | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6->1 | 0.5128 | 0.3333 | -0.1795 | 0.0509 | 0.0470 | -0.0039 |
| 6->3 | 0.5767 | 0.3737 | -0.2030 | 0.1993 | 0.1411 | -0.0581 |
| 9->1 | 0.7576 | 0.4192 | -0.3384 | 0.0801 | 0.0665 | -0.0136 |
| 9->3 | 0.7732 | 0.7505 | -0.0228 | 1.5962 | 0.7729 | -0.8233 |

Interpretation:

- A larger `gamma_ct` improves the CT auxiliary fit and makes the model stop earlier.
- Calibration improves, especially on `9->3`, where BStd drops sharply.
- Ranking quality on the shorter windows degrades substantially, so `gamma_ct=0.20` is too aggressive if overall Ctd must be preserved.
- The result supports a calibration-versus-ranking tradeoff rather than a clean win from stronger CT supervision.

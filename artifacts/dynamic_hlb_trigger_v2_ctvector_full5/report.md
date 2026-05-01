# trigger_orchard_v2 CT vector auxiliary head

Baseline: `artifacts/dynamic_hlb_trigger_v2_full5` with scalar next-period CT-delta auxiliary target.

Changed setting:

- `ct_aux_target_mode`: `next_delta -> window_prefix_vector`
- `ct_delta_output_dim`: `1 -> 11`
- `ct_aux_window_specs`: `[[6, 1], [6, 3], [9, 1], [9, 3]]`
- All other settings matched the baseline `trigger_orchard_v2` 5-repeat run.

Aggregate effect:

| metric | scalar CT | vector CT | delta |
| --- | ---: | ---: | ---: |
| overall mean Ctd | 0.6551 | 0.5140 | -0.1411 |
| overall mean BStd | 0.4816 | 0.3235 | -0.1581 |
| overall Ctd std | 0.2087 | 0.2905 | +0.0818 |
| overall BStd std | 0.5855 | 0.3065 | -0.2790 |
| mean best validation Ctd | 0.7359 | 0.7472 | +0.0112 |
| mean best validation CT MAE | 4.0181 | 3.6590 | -0.3592 |
| mean best epoch | 10.8 | 8.6 | -2.2 |

Window-level test means:

| window | Ctd scalar | Ctd vector | delta | BStd scalar | BStd vector | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6->1 | 0.5128 | 0.5385 | +0.0256 | 0.0509 | 0.0457 | -0.0052 |
| 6->3 | 0.5767 | 0.4067 | -0.1700 | 0.1993 | 0.1693 | -0.0300 |
| 9->1 | 0.7576 | 0.4545 | -0.3030 | 0.0801 | 0.0627 | -0.0174 |
| 9->3 | 0.7732 | 0.6564 | -0.1168 | 1.5962 | 1.0163 | -0.5799 |

Interpretation:

- The vector CT head improves auxiliary CT fit more clearly than increasing `gamma_ct` alone did.
- Test calibration improves across all windows, especially `9->3`.
- Test ranking degrades on `6->3`, `9->1`, and `9->3`, and Ctd becomes less stable across repeats.
- This is not a clean replacement for the scalar CT head. It is another calibration-versus-ranking tradeoff, with a stronger calibration gain and a still-material ranking cost.


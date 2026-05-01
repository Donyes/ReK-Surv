# period_ms attention drilldown

## identity check
- repeat 0, landmark 6: eligible=14, unique_attention_rows=3, max_abs_diff=1.49e-08
- repeat 0, landmark 9: eligible=11, unique_attention_rows=2, max_abs_diff=1.49e-08
- repeat 1, landmark 6: eligible=14, unique_attention_rows=3, max_abs_diff=2.98e-08
- repeat 1, landmark 9: eligible=12, unique_attention_rows=1, max_abs_diff=0.00e+00
- repeat 2, landmark 6: eligible=14, unique_attention_rows=5, max_abs_diff=2.98e-08
- repeat 2, landmark 9: eligible=12, unique_attention_rows=1, max_abs_diff=0.00e+00
- repeat 3, landmark 6: eligible=14, unique_attention_rows=3, max_abs_diff=1.49e-08
- repeat 3, landmark 9: eligible=12, unique_attention_rows=1, max_abs_diff=0.00e+00
- repeat 4, landmark 6: eligible=14, unique_attention_rows=3, max_abs_diff=2.98e-08
- repeat 4, landmark 9: eligible=13, unique_attention_rows=4, max_abs_diff=2.98e-08

## trained attention mean
### landmark 6
- rank 1: P6 (2024-10-27~2024-11-22) weight=0.1955 +- 0.1181
- rank 2: P1 (2024-06-01~2024-06-22) weight=0.1792 +- 0.0523
- rank 3: P2 (2024-06-23~2024-07-20) weight=0.1636 +- 0.0382
- rank 4: P5 (2024-09-26~2024-10-26) weight=0.1591 +- 0.0100
- rank 5: P3 (2024-07-21~2024-08-26) weight=0.1586 +- 0.0332
- rank 6: P4 (2024-08-27~2024-09-25) weight=0.1441 +- 0.0440
### landmark 9
- rank 1: P1 (2024-06-01~2024-06-22) weight=0.1347 +- 0.0418
- rank 2: P9 (2025-01-02~2025-01-21) weight=0.1230 +- 0.0791
- rank 3: P2 (2024-06-23~2024-07-20) weight=0.1191 +- 0.0405
- rank 4: P3 (2024-07-21~2024-08-26) weight=0.1174 +- 0.0265
- rank 5: P4 (2024-08-27~2024-09-25) weight=0.1150 +- 0.0332
- rank 6: P5 (2024-09-26~2024-10-26) weight=0.1136 +- 0.0234
- rank 7: P6 (2024-10-27~2024-11-22) weight=0.1017 +- 0.0236
- rank 8: P8 (2024-12-02~2025-01-01) weight=0.0944 +- 0.0159
- rank 9: P7 (2024-11-23~2024-12-01) weight=0.0812 +- 0.0246

## random init control
### landmark 6
- P1 (2024-06-01~2024-06-22) random_mean=0.1655, uniform=0.1667, diff=-0.0011
- P2 (2024-06-23~2024-07-20) random_mean=0.1673, uniform=0.1667, diff=+0.0006
- P3 (2024-07-21~2024-08-26) random_mean=0.1655, uniform=0.1667, diff=-0.0012
- P4 (2024-08-27~2024-09-25) random_mean=0.1665, uniform=0.1667, diff=-0.0002
- P5 (2024-09-26~2024-10-26) random_mean=0.1708, uniform=0.1667, diff=+0.0041
- P6 (2024-10-27~2024-11-22) random_mean=0.1645, uniform=0.1667, diff=-0.0022
### landmark 9
- P1 (2024-06-01~2024-06-22) random_mean=0.1095, uniform=0.1111, diff=-0.0016
- P2 (2024-06-23~2024-07-20) random_mean=0.1088, uniform=0.1111, diff=-0.0023
- P3 (2024-07-21~2024-08-26) random_mean=0.1120, uniform=0.1111, diff=+0.0009
- P4 (2024-08-27~2024-09-25) random_mean=0.1102, uniform=0.1111, diff=-0.0009
- P5 (2024-09-26~2024-10-26) random_mean=0.1136, uniform=0.1111, diff=+0.0025
- P6 (2024-10-27~2024-11-22) random_mean=0.1115, uniform=0.1111, diff=+0.0004
- P7 (2024-11-23~2024-12-01) random_mean=0.1112, uniform=0.1111, diff=+0.0001
- P8 (2024-12-02~2025-01-01) random_mean=0.1137, uniform=0.1111, diff=+0.0026
- P9 (2025-01-02~2025-01-21) random_mean=0.1095, uniform=0.1111, diff=-0.0016
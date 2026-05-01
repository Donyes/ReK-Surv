# Active Context

- Updated: 2026-05-01T14:35:10+08:00
- Current focus: Stabilized trigger_orchard_v2 results and matched tree-query comparison
- Latest outcome: This session implemented trigger_orchard_v2 as a tree-conditioned continuous-window trigger-search model, ran matched 5-repeat experiments, and compared it against non-spatial period_ms_tree_query. trigger_orchard_v2 consistently improved Ctd over period_ms_tree_query and the original trigger_orchard, but its BStd remained unstable on longer horizons and the CT-delta auxiliary regression stayed near the naive baseline.
- Main blocker: trigger_orchard_v2 still has a calibration problem on longer horizons, especially 9->3, 3->9, and 6->6, even when its Ctd is strong.

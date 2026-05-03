# trigger_orchard_v2 raw-daily vs agro-daily (gamma_ct=0.1)

Compared runs:
- Agro-expanded baseline: `artifacts/dynamic_hlb_trigger_v2_full5`
- Raw daily input: `artifacts/dynamic_hlb_trigger_v2_raw_full5`
- Shared settings: `trigger_orchard_v2`, `use_ct_aux_task`, `gamma_ct=0.1`, 5 repeats, windows `6->1`, `6->3`, `9->1`, `9->3`

Overall summary:
- Summary-window mean Ctd: agro=0.6551, raw=0.5399, delta=-0.1152
- Summary-window mean BStd: agro=0.4816, raw=0.4137, delta=-0.0679
- Long-window mean Ctd (6->3, 9->3): agro=0.6750, raw=0.6415, delta=-0.0334
- Long-window mean BStd (6->3, 9->3): agro=0.8977, raw=0.7536, delta=-0.1441
- Short-window mean Ctd (6->1, 9->1): agro=0.6352, raw=0.4382, delta=-0.1970
- Short-window mean BStd (6->1, 9->1): agro=0.0655, raw=0.0738, delta=+0.0083
- Repeat-level overall test Ctd mean: agro=0.6613, raw=0.5584, delta=-0.1028
- Repeat-level overall test BStd mean: agro=0.4816, raw=0.4137, delta=-0.0679
- Mean best-val CT MAE: agro=4.0181, raw=3.6821, delta=-0.3361

Window deltas:
- 6->1: Ctd 0.5128 -> 0.5128 (+0.0000), BStd 0.0509 -> 0.0714 (+0.0205)
- 6->3: Ctd 0.5767 -> 0.5124 (-0.0644), BStd 0.1993 -> 0.2015 (+0.0022)
- 9->1: Ctd 0.7576 -> 0.3636 (-0.3939), BStd 0.0801 -> 0.0762 (-0.0038)
- 9->3: Ctd 0.7732 -> 0.7707 (-0.0025), BStd 1.5962 -> 1.3057 (-0.2905)

Repeat-level deltas:
- Repeat 0: overall Ctd 0.3597 -> 0.3888 (+0.0291), overall BStd 0.1637 -> 0.2429 (+0.0792), val CT MAE 4.2933 -> 3.8934 (-0.3999)
- Repeat 1: overall Ctd 0.6418 -> 0.6683 (+0.0265), overall BStd 0.2038 -> 0.0960 (-0.1078), val CT MAE 3.6635 -> 3.7475 (+0.0840)
- Repeat 2: overall Ctd 0.7524 -> 0.7010 (-0.0514), overall BStd 0.2494 -> 0.2681 (+0.0188), val CT MAE 3.9383 -> 3.2435 (-0.6947)
- Repeat 3: overall Ctd 0.6661 -> 0.7229 (+0.0567), overall BStd 1.5481 -> 1.2052 (-0.3429), val CT MAE 3.8267 -> 3.5885 (-0.2382)
- Repeat 4: overall Ctd 0.8863 -> 0.3112 (-0.5751), overall BStd 0.2431 -> 0.2562 (+0.0131), val CT MAE 4.3688 -> 3.9374 (-0.4314)

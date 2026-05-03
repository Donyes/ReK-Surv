# Active Context

- Updated: 2026-05-02T01:06:03+08:00
- Current focus: trigger_orchard_v2 CT-vector branch comparison, including the raw-daily versus agro-daily input ablation
- Latest outcome: After the gamma_ct scan, this session ran a matched 5-repeat trigger_orchard_v2 gamma_ct=0.1 ablation with raw 9-dim daily inputs instead of the 75-dim agro-expanded daily features. The raw-input branch slightly improved mean BStd and validation CT MAE, but it reduced mean window Ctd from about 0.655 to about 0.540, with the largest ranking loss on 9->1 and a smaller drop on 6->3, so the agro-expanded daily features remain the stronger default input for trigger_orchard_v2 when ranking matters.
- Main blocker: There is still no trigger_orchard_v2 setting that jointly improves ranking, calibration, and CT fit; both the gamma_ct tradeoff and the raw-daily ablation suggest the branch can recover some calibration only by giving up too much discrimination.

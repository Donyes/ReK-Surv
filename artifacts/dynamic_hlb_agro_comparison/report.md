# HLB Agro-Prior Dynamic Model Comparison

## Overall
- period_ms_agro: overall Ctd=0.5789+/-0.1014, overall BStd=0.3068+/-0.2773, long-window Ctd=0.6288, long-window BStd=0.5634
- baseline_legacy_current: overall Ctd=0.5354+/-0.1352, overall BStd=0.4751+/-0.4204, long-window Ctd=0.5551, long-window BStd=0.8979
- legacy_agro: overall Ctd=0.5269+/-0.0970, overall BStd=0.3495+/-0.3182, long-window Ctd=0.5410, long-window BStd=0.6474
- trigger_orchard_agro: overall Ctd=0.4739+/-0.0487, overall BStd=0.3232+/-0.2390, long-window Ctd=0.5865, long-window BStd=0.5880

## Per Window Winners
- 6->1: best Ctd = baseline_legacy_current (0.8974), best BStd = period_ms_agro (0.0422)
- 6->3: best Ctd = period_ms_agro (0.5625), best BStd = period_ms_agro (0.1331)
- 9->1: best Ctd = legacy_agro (0.4268), best BStd = period_ms_agro (0.0581)
- 9->3: best Ctd = period_ms_agro (0.6951), best BStd = period_ms_agro (0.9937)

## Interpretation
- period_ms_agro is the strongest overall option: it is the only new model that improves both mean Ctd and mean BStd against the baseline, and it is strongest on the longer-horizon windows 6->3 and 9->3.
- legacy_agro mainly improves calibration/Brier score, but does not improve overall discrimination relative to the baseline.
- trigger_orchard_agro is very stable on the long windows, but sacrifices too much short-window discrimination to be the best default model right now.
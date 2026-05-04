# Next Steps

- Run shared-split harmful-sample attribution for windows 4:3, 4:4, and 8:4, then propose the smallest joint Sheet2+Sheet4 edits that directly target those weak windows.
- Decide whether future dataset edits should be judged primarily by mean Ctd or by calibration and repeat stability before making another retuning pass.
- If the paper window set is being finalized now, start from the strict rows in artifacts/recommended_eval_windows_latefix3.csv and rerun the main model families under artifacts/dynamic_hlb_trigger_v2_windowset_shared_splits_seed42.json.

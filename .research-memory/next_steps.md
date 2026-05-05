# Next Steps

- Build a paper-ready comparison that includes trigger_orchard_v3-noCT, the old v3 delta baseline, and the new v3 delta+LOD result on the original shared-split full5 setting, with period_ms_tree_query kept as the project-level mainline reference.
- Decide the trigger_orchard paper-facing default by metric priority: if a CT-enabled v3 variant is needed, use delta+LOD as the current default; if calibration is primary, keep v3-noCT as the stronger reference.
- Before any tuned/latefix3 rerun, inspect why delta+LOD still degrades the 8:5 window relative to the old v3 delta and v3-noCT baselines.

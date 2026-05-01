# Next Steps

- Tune trigger_orchard_v2 with a milder CT-loss change, such as an intermediate `gamma_ct` between 0.10 and 0.20, while checking whether the long-horizon BStd gain can be kept without losing short-window Ctd.
- Replace or simplify the CT auxiliary regression target, starting with direction-based or binned CT targets.
- After the next trigger_orchard_v2 round, decide whether the paper should center the trigger_orchard_v2 branch or keep period_ms / period_ms_tree_query as the more stable main line.

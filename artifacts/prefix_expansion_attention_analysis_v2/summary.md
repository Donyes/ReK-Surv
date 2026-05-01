# Tree-specific period attention diagnostics

## Trained attention diversity
- dynamic_hlb_tree_query_full5 6->1: Ctd=0.3590, BStd=0.0425, eligible=14.0, events=0.6, JS=0.000500, unique_rows=14.0
- dynamic_hlb_tree_query_full5 6->3: Ctd=0.5869, BStd=0.1276, eligible=14.0, events=2.0, JS=0.000500, unique_rows=14.0
- dynamic_hlb_tree_query_full5 6->6: Ctd=0.7290, BStd=1.4223, eligible=14.0, events=4.4, JS=0.000500, unique_rows=14.0
- dynamic_hlb_tree_query_full5 9->1: Ctd=0.5455, BStd=0.0551, eligible=12.0, events=0.6, JS=0.000506, unique_rows=12.0
- dynamic_hlb_tree_query_full5 9->3: Ctd=0.6944, BStd=0.9960, eligible=12.0, events=2.4, JS=0.000506, unique_rows=12.0
- prefix_expansion_probe_lm69_full5 6->1: Ctd=0.3333, BStd=0.0465, eligible=14.0, events=0.6, JS=0.000021, unique_rows=14.0
- prefix_expansion_probe_lm69_full5 6->3: Ctd=0.3507, BStd=0.1648, eligible=14.0, events=2.0, JS=0.000021, unique_rows=14.0
- prefix_expansion_probe_lm69_full5 6->6: Ctd=0.4874, BStd=3.6881, eligible=14.0, events=4.4, JS=0.000021, unique_rows=14.0
- prefix_expansion_probe_lm69_full5 9->1: Ctd=0.6439, BStd=0.0607, eligible=12.0, events=0.6, JS=0.000020, unique_rows=12.0
- prefix_expansion_probe_lm69_full5 9->3: Ctd=0.5706, BStd=2.6859, eligible=12.0, events=2.4, JS=0.000020, unique_rows=12.0
- prefix_expansion_probe_lm6_full5 6->1: Ctd=0.3590, BStd=0.0447, eligible=14.0, events=0.6, JS=0.000017, unique_rows=14.0
- prefix_expansion_probe_lm6_full5 6->3: Ctd=0.3502, BStd=0.1597, eligible=14.0, events=2.0, JS=0.000017, unique_rows=14.0
- prefix_expansion_probe_lm6_full5 6->6: Ctd=0.5203, BStd=3.4929, eligible=14.0, events=4.4, JS=0.000017, unique_rows=14.0
- prefix_expansion_probe_lm6_full5 9->1: Ctd=0.6692, BStd=0.0603, eligible=12.0, events=0.6, JS=0.000018, unique_rows=12.0
- prefix_expansion_probe_lm6_full5 9->3: Ctd=0.5312, BStd=2.5244, eligible=12.0, events=2.4, JS=0.000018, unique_rows=12.0
- prefix_expansion_probe_lm9_full5 6->1: Ctd=0.1026, BStd=0.0467, eligible=14.0, events=0.6, JS=0.000017, unique_rows=14.0
- prefix_expansion_probe_lm9_full5 6->3: Ctd=0.4049, BStd=0.1725, eligible=14.0, events=2.0, JS=0.000017, unique_rows=14.0
- prefix_expansion_probe_lm9_full5 6->6: Ctd=0.4326, BStd=3.8676, eligible=14.0, events=4.4, JS=0.000017, unique_rows=14.0
- prefix_expansion_probe_lm9_full5 9->1: Ctd=0.6818, BStd=0.0748, eligible=12.0, events=0.6, JS=0.000020, unique_rows=12.0
- prefix_expansion_probe_lm9_full5 9->3: Ctd=0.5643, BStd=2.8310, eligible=12.0, events=2.4, JS=0.000020, unique_rows=12.0

## Random-init control
- dynamic_hlb_tree_query_full5 6->1: random JS=0.000025
- dynamic_hlb_tree_query_full5 6->3: random JS=0.000025
- dynamic_hlb_tree_query_full5 6->6: random JS=0.000025
- dynamic_hlb_tree_query_full5 9->1: random JS=0.000028
- dynamic_hlb_tree_query_full5 9->3: random JS=0.000028
- prefix_expansion_probe_lm69_full5 6->1: random JS=0.000025
- prefix_expansion_probe_lm69_full5 6->3: random JS=0.000025
- prefix_expansion_probe_lm69_full5 6->6: random JS=0.000025
- prefix_expansion_probe_lm69_full5 9->1: random JS=0.000028
- prefix_expansion_probe_lm69_full5 9->3: random JS=0.000028
- prefix_expansion_probe_lm6_full5 6->1: random JS=0.000025
- prefix_expansion_probe_lm6_full5 6->3: random JS=0.000025
- prefix_expansion_probe_lm6_full5 6->6: random JS=0.000025
- prefix_expansion_probe_lm6_full5 9->1: random JS=0.000028
- prefix_expansion_probe_lm6_full5 9->3: random JS=0.000028
- prefix_expansion_probe_lm9_full5 6->1: random JS=0.000025
- prefix_expansion_probe_lm9_full5 6->3: random JS=0.000025
- prefix_expansion_probe_lm9_full5 6->6: random JS=0.000025
- prefix_expansion_probe_lm9_full5 9->1: random JS=0.000028
- prefix_expansion_probe_lm9_full5 9->3: random JS=0.000028

## Static-shuffle control
- dynamic_hlb_tree_query_full5 6->1: shuffled Ctd=0.5385, BStd=0.0409, |delta risk|=0.0146
- dynamic_hlb_tree_query_full5 6->3: shuffled Ctd=0.5554, BStd=0.1277, |delta risk|=0.0363
- dynamic_hlb_tree_query_full5 6->6: shuffled Ctd=0.4703, BStd=1.5706, |delta risk|=0.0615
- dynamic_hlb_tree_query_full5 9->1: shuffled Ctd=0.3939, BStd=0.0565, |delta risk|=0.0429
- dynamic_hlb_tree_query_full5 9->3: shuffled Ctd=0.4187, BStd=1.0892, |delta risk|=0.0468
- prefix_expansion_probe_lm69_full5 6->1: shuffled Ctd=0.5470, BStd=0.0464, |delta risk|=0.0012
- prefix_expansion_probe_lm69_full5 6->3: shuffled Ctd=0.5422, BStd=0.1646, |delta risk|=0.0015
- prefix_expansion_probe_lm69_full5 6->6: shuffled Ctd=0.6120, BStd=3.6863, |delta risk|=0.0017
- prefix_expansion_probe_lm69_full5 9->1: shuffled Ctd=0.4318, BStd=0.0608, |delta risk|=0.0019
- prefix_expansion_probe_lm69_full5 9->3: shuffled Ctd=0.4519, BStd=2.6880, |delta risk|=0.0027
- prefix_expansion_probe_lm6_full5 6->1: shuffled Ctd=0.5641, BStd=0.0446, |delta risk|=0.0013
- prefix_expansion_probe_lm6_full5 6->3: shuffled Ctd=0.4981, BStd=0.1597, |delta risk|=0.0015
- prefix_expansion_probe_lm6_full5 6->6: shuffled Ctd=0.5893, BStd=3.4916, |delta risk|=0.0023
- prefix_expansion_probe_lm6_full5 9->1: shuffled Ctd=0.4049, BStd=0.0603, |delta risk|=0.0018
- prefix_expansion_probe_lm6_full5 9->3: shuffled Ctd=0.4801, BStd=2.5259, |delta risk|=0.0030
- prefix_expansion_probe_lm9_full5 6->1: shuffled Ctd=0.3761, BStd=0.0467, |delta risk|=0.0013
- prefix_expansion_probe_lm9_full5 6->3: shuffled Ctd=0.5946, BStd=0.1724, |delta risk|=0.0019
- prefix_expansion_probe_lm9_full5 6->6: shuffled Ctd=0.6458, BStd=3.8657, |delta risk|=0.0017
- prefix_expansion_probe_lm9_full5 9->1: shuffled Ctd=0.5598, BStd=0.0748, |delta risk|=0.0016
- prefix_expansion_probe_lm9_full5 9->3: shuffled Ctd=0.4993, BStd=2.8327, |delta risk|=0.0018

## Outputs
- all_sample_attention.csv: every eligible test tree with risk and attn_P1...attn_P13.
- static_group_attention.csv: attention summaries grouped by Sheet2 static attributes.
- period_occlusion_summary.csv: periods whose masking changes risk the most.
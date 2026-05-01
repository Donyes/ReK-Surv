# Tree-specific period attention diagnostics

## Trained attention diversity
- smoke_tree_query 3->3: Ctd=0.6216, BStd=0.1936, eligible=19.0, events=5.0, JS=0.000021, unique_rows=19.0
- smoke_tree_query 6->3: Ctd=0.5714, BStd=0.1685, eligible=14.0, events=3.0, JS=0.000016, unique_rows=14.0
- smoke_tree_query 9->3: Ctd=0.2222, BStd=0.1509, eligible=11.0, events=2.0, JS=0.000013, unique_rows=11.0
- smoke_tree_query_spatial 3->3: Ctd=0.6081, BStd=0.1940, eligible=19.0, events=5.0, JS=0.000052, unique_rows=19.0
- smoke_tree_query_spatial 6->3: Ctd=0.7429, BStd=0.1745, eligible=14.0, events=3.0, JS=0.000016, unique_rows=14.0
- smoke_tree_query_spatial 9->3: Ctd=0.3333, BStd=0.1533, eligible=11.0, events=2.0, JS=0.000060, unique_rows=11.0

## Random-init control
- smoke_tree_query 3->3: random JS=0.000033
- smoke_tree_query 6->3: random JS=0.000030
- smoke_tree_query 9->3: random JS=0.000036
- smoke_tree_query_spatial 3->3: random JS=0.000029
- smoke_tree_query_spatial 6->3: random JS=0.000030
- smoke_tree_query_spatial 9->3: random JS=0.000039

## Static-shuffle control
- smoke_tree_query 3->3: shuffled Ctd=0.3176, BStd=0.1944, |delta risk|=0.0025
- smoke_tree_query 6->3: shuffled Ctd=0.5429, BStd=0.1686, |delta risk|=0.0032
- smoke_tree_query 9->3: shuffled Ctd=0.2222, BStd=0.1509, |delta risk|=0.0055
- smoke_tree_query_spatial 3->3: shuffled Ctd=0.6014, BStd=0.1943, |delta risk|=0.0053
- smoke_tree_query_spatial 6->3: shuffled Ctd=0.6143, BStd=0.1748, |delta risk|=0.0061
- smoke_tree_query_spatial 9->3: shuffled Ctd=0.3333, BStd=0.1533, |delta risk|=0.0088

## Outputs
- all_sample_attention.csv: every eligible test tree with risk and attn_P1...attn_P13.
- static_group_attention.csv: attention summaries grouped by Sheet2 static attributes.
- period_occlusion_summary.csv: periods whose masking changes risk the most.
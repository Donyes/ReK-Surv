# period_ms 时期级环境时间关注实验报告

- 分析对象：`artifacts\dynamic_hlb_period_ms_full5`
- 窗口：6->1, 6->3, 9->1, 9->3

## Overall

- full: Ctd=0.5721, BStd=0.3068
- best only_last_k: only_last_k_1 -> Ctd=0.5343, BStd=0.3187
- best exclude_last_k: exclude_last_k_2 -> Ctd=0.5886, BStd=0.3278

## Window 6->1

- 风险集规模：eligible_n=14.0, future_event_n=0.6
- attention enrichment 排名：last_1_period(1.173), periods_4_6(1.003), periods_2_3(0.910)
- occlusion Δrisk 排名：periods_4_6(0.0076), last_1_period(0.0072), periods_2_3(0.0038)
- permutation Δrisk 排名：last_1_period(0.0002), periods_4_6(0.0001), periods_2_3(0.0001)
- full：Ctd=0.5385, BStd=0.0422
- only_last_k：k=1 -> Ctd=0.5128, BStd=0.0465, k=2 -> Ctd=0.5128, BStd=0.0468, k=3 -> Ctd=0.5385, BStd=0.0449
- exclude_last_k：k=1 -> Ctd=0.5641, BStd=0.0444, k=2 -> Ctd=0.5385, BStd=0.0460, k=3 -> Ctd=0.5385, BStd=0.0452
- 结论：证据混合：模型并非只看最近时期，但也没有完全摆脱相邻时期的影响，需要结合各窗口分别解释。

## Window 6->3

- 风险集规模：eligible_n=14.0, future_event_n=2.0
- attention enrichment 排名：last_1_period(1.173), periods_4_6(1.003), periods_2_3(0.910)
- occlusion Δrisk 排名：periods_4_6(0.0230), last_1_period(0.0155), periods_2_3(0.0127)
- permutation Δrisk 排名：last_1_period(0.0002), periods_2_3(0.0002), periods_4_6(0.0001)
- full：Ctd=0.5625, BStd=0.1331
- only_last_k：k=1 -> Ctd=0.5277, BStd=0.1398, k=2 -> Ctd=0.4980, BStd=0.1378, k=3 -> Ctd=0.5305, BStd=0.1324
- exclude_last_k：k=1 -> Ctd=0.5682, BStd=0.1310, k=2 -> Ctd=0.5579, BStd=0.1376, k=3 -> Ctd=0.5499, BStd=0.1321
- 结论：证据混合：模型并非只看最近时期，但也没有完全摆脱相邻时期的影响，需要结合各窗口分别解释。

## Window 9->1

- 风险集规模：eligible_n=12.0, future_event_n=0.6
- attention enrichment 排名：periods_7_plus(1.113), last_1_period(1.107), periods_4_6(0.991), periods_2_3(0.790)
- occlusion Δrisk 排名：periods_7_plus(0.0142), periods_4_6(0.0072), periods_2_3(0.0033), last_1_period(0.0018)
- permutation Δrisk 排名：periods_7_plus(0.0002), last_1_period(0.0001), periods_2_3(0.0001), periods_4_6(0.0001)
- full：Ctd=0.4167, BStd=0.0581
- only_last_k：k=1 -> Ctd=0.4470, BStd=0.0590, k=2 -> Ctd=0.2348, BStd=0.0533, k=3 -> Ctd=0.2929, BStd=0.0544
- exclude_last_k：k=1 -> Ctd=0.2929, BStd=0.0517, k=2 -> Ctd=0.4722, BStd=0.0560, k=3 -> Ctd=0.3712, BStd=0.0504
- 结论：证据混合：模型并非只看最近时期，但也没有完全摆脱相邻时期的影响，需要结合各窗口分别解释。

## Window 9->3

- 风险集规模：eligible_n=12.0, future_event_n=2.4
- attention enrichment 排名：periods_7_plus(1.113), last_1_period(1.107), periods_4_6(0.991), periods_2_3(0.790)
- occlusion Δrisk 排名：periods_7_plus(0.0335), periods_4_6(0.0169), periods_2_3(0.0094), last_1_period(0.0046)
- permutation Δrisk 排名：periods_7_plus(0.0002), last_1_period(0.0002), periods_2_3(0.0002), periods_4_6(0.0001)
- full：Ctd=0.6951, BStd=0.9937
- only_last_k：k=1 -> Ctd=0.6063, BStd=1.0296, k=2 -> Ctd=0.6389, BStd=1.0808, k=3 -> Ctd=0.6284, BStd=1.0032
- exclude_last_k：k=1 -> Ctd=0.7175, BStd=0.9340, k=2 -> Ctd=0.7193, BStd=1.0716, k=3 -> Ctd=0.7282, BStd=0.9482
- 结论：支持更长历史的阶段组合与滞后效应：只保留最近时期无法维持性能，而去掉最近 1 到 2 个时期后仍保留较多判别力。

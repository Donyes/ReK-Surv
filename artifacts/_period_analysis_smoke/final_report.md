# period_ms 时期级环境时间关注实验报告

- 分析对象：`artifacts\dynamic_hlb_period_ms_full5`
- 窗口：6->1, 6->3, 9->1, 9->3

## Overall

- full: Ctd=0.6188, BStd=0.0977
- best only_last_k: only_last_k_1 -> Ctd=0.6379, BStd=0.1132
- best exclude_last_k: exclude_last_k_2 -> Ctd=0.6379, BStd=0.0975

## Window 6->1

- 风险集规模：eligible_n=14.0, future_event_n=1.0
- attention enrichment 排名：periods_4_6(1.114), periods_2_3(1.053), last_1_period(0.554)
- occlusion Δrisk 排名：last_1_period(0.0149), periods_4_6(0.0046), periods_2_3(0.0036)
- permutation Δrisk 排名：last_1_period(0.0001), periods_4_6(0.0000), periods_2_3(0.0000)
- full：Ctd=0.9231, BStd=0.0661
- only_last_k：
- exclude_last_k：
- 结论：证据混合：模型并非只看最近时期，但也没有完全摆脱相邻时期的影响，需要结合各窗口分别解释。

## Window 6->3

- 风险集规模：eligible_n=14.0, future_event_n=3.0
- attention enrichment 排名：periods_4_6(1.114), periods_2_3(1.053), last_1_period(0.554)
- occlusion Δrisk 排名：last_1_period(0.0298), periods_4_6(0.0169), periods_2_3(0.0108)
- permutation Δrisk 排名：periods_4_6(0.0000), last_1_period(0.0000), periods_2_3(0.0000)
- full：Ctd=0.6000, BStd=0.1673
- only_last_k：
- exclude_last_k：
- 结论：证据混合：模型并非只看最近时期，但也没有完全摆脱相邻时期的影响，需要结合各窗口分别解释。

## Window 9->1

- 风险集规模：eligible_n=11.0, future_event_n=0.0
- attention enrichment 排名：periods_7_plus(1.228), periods_4_6(0.964), periods_2_3(0.928), last_1_period(0.570)
- occlusion Δrisk 排名：periods_7_plus(0.0091), periods_4_6(0.0075), periods_2_3(0.0040), last_1_period(0.0001)
- permutation Δrisk 排名：periods_7_plus(0.0001), last_1_period(0.0000), periods_4_6(0.0000), periods_2_3(0.0000)
- full：Ctd=0.0000, BStd=0.0047
- only_last_k：
- exclude_last_k：
- 结论：证据混合：模型并非只看最近时期，但也没有完全摆脱相邻时期的影响，需要结合各窗口分别解释。

## Window 9->3

- 风险集规模：eligible_n=11.0, future_event_n=2.0
- attention enrichment 排名：periods_7_plus(1.228), periods_4_6(0.964), periods_2_3(0.928), last_1_period(0.570)
- occlusion Δrisk 排名：periods_7_plus(0.0243), periods_4_6(0.0198), periods_2_3(0.0109), last_1_period(0.0003)
- permutation Δrisk 排名：periods_7_plus(0.0002), last_1_period(0.0001), periods_4_6(0.0001), periods_2_3(0.0000)
- full：Ctd=0.3333, BStd=0.1525
- only_last_k：
- exclude_last_k：
- 结论：证据混合：模型并非只看最近时期，但也没有完全摆脱相邻时期的影响，需要结合各窗口分别解释。

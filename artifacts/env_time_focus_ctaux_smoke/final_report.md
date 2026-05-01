# 环境时间关注实验报告

- 分析对象：`artifacts\dynamic_hlb_ctaux_full5`
- 窗口：6->1, 6->3, 9->1, 9->3

## Overall

- `full` 平均 `Ctd=0.4810`，`BStd=0.1071`
- 最佳 `only_last_k` 变体：`only_last_k_2`，平均 `Ctd=0.5125`，`BStd=0.1237`
- 最佳 `exclude_last_k` 变体：`exclude_last_k_7`，平均 `Ctd=0.4739`，`BStd=0.1065`

## Window 6->1

- 风险集规模：`eligible_n≈14.0`，未来事件数：`future_event_n≈1.0`
- Attention enrichment 排名：days_8_14(1.061), days_3_7(1.058), last_2_days(1.049), days_15_30(1.038), days_31_60(1.033), days_60_plus(0.979)
- 遮挡 Δrisk 排名：last_2_days(0.0019), days_3_7(0.0005), days_60_plus(0.0002), days_31_60(0.0001), days_8_14(0.0001), days_15_30(0.0001)
- 置乱 Δrisk 排名：last_2_days(0.0007), days_3_7(0.0001), days_8_14(0.0000), days_15_30(0.0000), days_60_plus(0.0000), days_31_60(0.0000)
- `full`：`Ctd=0.8462`，`BStd=0.0663`
- `only_last_k`：`k=2` -> Ctd=0.7692, BStd=0.0664, `k=7` -> Ctd=0.7692, BStd=0.0665
- `exclude_last_k`：`k=2` -> Ctd=0.7692, BStd=0.0664, `k=7` -> Ctd=0.7692, BStd=0.0664
- 结论：证据混合：模型既没有表现出明确的“只看检测日前后”特征，也不能简单归结为完全由更早长历史主导，需要结合具体窗口分别解释。

## Window 6->3

- 风险集规模：`eligible_n≈14.0`，未来事件数：`future_event_n≈3.0`
- Attention enrichment 排名：days_8_14(1.061), days_3_7(1.058), last_2_days(1.049), days_15_30(1.038), days_31_60(1.033), days_60_plus(0.979)
- 遮挡 Δrisk 排名：last_2_days(0.0061), days_3_7(0.0018), days_60_plus(0.0006), days_31_60(0.0004), days_8_14(0.0004), days_15_30(0.0003)
- 置乱 Δrisk 排名：last_2_days(0.0014), days_3_7(0.0003), days_8_14(0.0001), days_15_30(0.0000), days_60_plus(0.0000), days_31_60(0.0000)
- `full`：`Ctd=0.4857`，`BStd=0.1736`
- `only_last_k`：`k=2` -> Ctd=0.6571, BStd=0.1737, `k=7` -> Ctd=0.4857, BStd=0.1722
- `exclude_last_k`：`k=2` -> Ctd=0.5143, BStd=0.1723, `k=7` -> Ctd=0.4857, BStd=0.1730
- 结论：证据混合：模型既没有表现出明确的“只看检测日前后”特征，也不能简单归结为完全由更早长历史主导，需要结合具体窗口分别解释。

## Window 9->1

- 风险集规模：`eligible_n≈11.0`，未来事件数：`future_event_n≈0.0`
- Attention enrichment 排名：days_31_60(1.030), last_2_days(1.029), days_8_14(1.025), days_15_30(1.025), days_3_7(1.025), days_60_plus(0.990)
- 遮挡 Δrisk 排名：last_2_days(0.0014), days_3_7(0.0009), days_60_plus(0.0002), days_8_14(0.0001), days_31_60(0.0001), days_15_30(0.0000)
- 置乱 Δrisk 排名：last_2_days(0.0016), days_3_7(0.0004), days_8_14(0.0000), days_15_30(0.0000), days_60_plus(0.0000), days_31_60(0.0000)
- `full`：`Ctd=0.0000`，`BStd=0.0138`
- `only_last_k`：`k=2` -> Ctd=0.0000, BStd=0.0188, `k=7` -> Ctd=0.0000, BStd=0.0113
- `exclude_last_k`：`k=2` -> Ctd=0.0000, BStd=0.0117, `k=7` -> Ctd=0.0000, BStd=0.0113
- 结论：证据混合：模型既没有表现出明确的“只看检测日前后”特征，也不能简单归结为完全由更早长历史主导，需要结合具体窗口分别解释。

## Window 9->3

- 风险集规模：`eligible_n≈11.0`，未来事件数：`future_event_n≈2.0`
- Attention enrichment 排名：days_31_60(1.030), last_2_days(1.029), days_8_14(1.025), days_15_30(1.025), days_3_7(1.025), days_60_plus(0.990)
- 遮挡 Δrisk 排名：last_2_days(0.0039), days_3_7(0.0026), days_60_plus(0.0006), days_8_14(0.0004), days_31_60(0.0002), days_15_30(0.0001)
- 置乱 Δrisk 排名：last_2_days(0.0037), days_3_7(0.0010), days_8_14(0.0001), days_15_30(0.0000), days_60_plus(0.0000), days_31_60(0.0000)
- `full`：`Ctd=0.1111`，`BStd=0.1747`
- `only_last_k`：`k=2` -> Ctd=0.1111, BStd=0.2359, `k=7` -> Ctd=0.1111, BStd=0.1760
- `exclude_last_k`：`k=2` -> Ctd=0.1111, BStd=0.1743, `k=7` -> Ctd=0.1667, BStd=0.1753
- 结论：证据混合：模型既没有表现出明确的“只看检测日前后”特征，也不能简单归结为完全由更早长历史主导，需要结合具体窗口分别解释。

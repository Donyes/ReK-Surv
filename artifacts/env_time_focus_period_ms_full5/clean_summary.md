# period_ms ?????????

## 6->1
- full: Ctd=0.5385, BStd=0.0422, eligible_n=14.0, future_event_n=0.6
- attention enrichment ??: last_1_period(1.173), periods_4_6(1.003), periods_2_3(0.910)
- occlusion ?risk ??: periods_4_6(0.0076), last_1_period(0.0072), periods_2_3(0.0038)
- permutation ?risk ??: last_1_period(0.0002), periods_4_6(0.0001), periods_2_3(0.0001)
- only_last_k ??: only_last_k_3: Ctd=0.5385, BStd=0.0449, only_last_k_1: Ctd=0.5128, BStd=0.0465, only_last_k_2: Ctd=0.5128, BStd=0.0468
- exclude_last_k ??: exclude_last_k_1: Ctd=0.5641, BStd=0.0444, exclude_last_k_2: Ctd=0.5385, BStd=0.0460, exclude_last_k_3: Ctd=0.5385, BStd=0.0452

## 6->3
- full: Ctd=0.5625, BStd=0.1331, eligible_n=14.0, future_event_n=2.0
- attention enrichment ??: last_1_period(1.173), periods_4_6(1.003), periods_2_3(0.910)
- occlusion ?risk ??: periods_4_6(0.0230), last_1_period(0.0155), periods_2_3(0.0127)
- permutation ?risk ??: last_1_period(0.0002), periods_2_3(0.0002), periods_4_6(0.0001)
- only_last_k ??: only_last_k_3: Ctd=0.5305, BStd=0.1324, only_last_k_1: Ctd=0.5277, BStd=0.1398, only_last_k_2: Ctd=0.4980, BStd=0.1378
- exclude_last_k ??: exclude_last_k_1: Ctd=0.5682, BStd=0.1310, exclude_last_k_2: Ctd=0.5579, BStd=0.1376, exclude_last_k_3: Ctd=0.5499, BStd=0.1321

## 9->1
- full: Ctd=0.4167, BStd=0.0581, eligible_n=12.0, future_event_n=0.6
- attention enrichment ??: periods_7_plus(1.113), last_1_period(1.107), periods_4_6(0.991), periods_2_3(0.790)
- occlusion ?risk ??: periods_7_plus(0.0142), periods_4_6(0.0072), periods_2_3(0.0033), last_1_period(0.0018)
- permutation ?risk ??: periods_7_plus(0.0002), last_1_period(0.0001), periods_2_3(0.0001), periods_4_6(0.0001)
- only_last_k ??: only_last_k_1: Ctd=0.4470, BStd=0.0590, only_last_k_3: Ctd=0.2929, BStd=0.0544, only_last_k_2: Ctd=0.2348, BStd=0.0533
- exclude_last_k ??: exclude_last_k_2: Ctd=0.4722, BStd=0.0560, exclude_last_k_3: Ctd=0.3712, BStd=0.0504, exclude_last_k_1: Ctd=0.2929, BStd=0.0517

## 9->3
- full: Ctd=0.6951, BStd=0.9937, eligible_n=12.0, future_event_n=2.4
- attention enrichment ??: periods_7_plus(1.113), last_1_period(1.107), periods_4_6(0.991), periods_2_3(0.790)
- occlusion ?risk ??: periods_7_plus(0.0335), periods_4_6(0.0169), periods_2_3(0.0094), last_1_period(0.0046)
- permutation ?risk ??: periods_7_plus(0.0002), last_1_period(0.0002), periods_2_3(0.0002), periods_4_6(0.0001)
- only_last_k ??: only_last_k_2: Ctd=0.6389, BStd=1.0808, only_last_k_3: Ctd=0.6284, BStd=1.0032, only_last_k_1: Ctd=0.6063, BStd=1.0296
- exclude_last_k ??: exclude_last_k_3: Ctd=0.7282, BStd=0.9482, exclude_last_k_2: Ctd=0.7193, BStd=1.0716, exclude_last_k_1: Ctd=0.7175, BStd=0.9340

## ??

- 6->3 ? 9->3 ???????????????????????????? 1 ????
- 9->3 ??????? periods_7_plus ???????????? last_1_period ???????????????????????
- history ablation ??only_last_k ? 6->3 ? 9->3 ????? full?????exclude_last_k ? 9->3 ???? full???????????????
- permutation ??????????? period_ms ????????????? lag?????????????????????? lag-aware attention ?????? RNN/Transformer ???
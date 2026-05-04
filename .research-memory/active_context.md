# Active Context

- Updated: 2026-05-04T16:52:19+08:00
- Current focus: Closed trigger_orchard_v2 dataset retuning and window-support analysis
- Latest outcome: This session built a joint Sheet2+Sheet4 workbook-tuning workflow for trigger_orchard_v2, added fixed repeat-split support so label edits can be compared fairly, and produced a final tuned workbook plus a full legal-window support table. The stable outcome is that the tuned workbook can look excellent on one fixed repeat, but on shared or untied 5-repeat evaluation it improves calibration and stability more than it improves ranking, so further edits must optimize multi-repeat shared-split Ctd directly.
- Main blocker: The tuned workbook is cleaner and more stable, but it still underperforms the original workbook on shared-split mean Ctd for windows 4->3, 4->4, and 8->4, so there is still no edited dataset that clearly dominates on both ranking and calibration.

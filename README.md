# PoseAITraining — Run Summary

## Config

- Device: cpu

- TOTAL_IMAGES: 2000

- VAL_RATIO: 0.2

- TEST_RATIO: 0.0

- WARMUP_EPOCHS: 1

- FINETUNE_EPOCHS: 4

- UNFREEZE_LAST_BLOCKS: 4


## Outputs

- `outputs/eda_*.png` + `outputs/eda_*json` + `outputs/eda_*.csv` (expanded EDA)

- `outputs/*_predictions.csv` (includes keep/improve/summary columns)

- `outputs/confusion_*.png` + `outputs/classification_report_*.json`

- `outputs/vit_squat.pth` + `outputs/vit_squat_best.pth`

- `outputs/PoseAITraining_artifact.zip`


## Notes

TEST split is empty (TEST_RATIO=0). For convenience, test reports reuse val.


## Metrics

### val

- n = 400

- accuracy = 0.7875

### test

- n = 400

- accuracy = 0.7875

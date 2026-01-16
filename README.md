# PoseAITraining — Run Summary

## Project structure (required by instructor)

- Part 1: Synthetic data generation using 3D pose extraction + 3D pose manipulation + ControlNet Img2Img

- Part 2: Fine-tune a pretrained ViT image model to classify Good vs Bad squats using synthetic data


## Config

- Device: cpu

- TOTAL_IMAGES: 2000

- VAL_RATIO: 0.2

- TEST_RATIO: 0.0

- WARMUP_EPOCHS: 1

- FINETUNE_EPOCHS: 8

- UNFREEZE_LAST_BLOCKS: 4

- MIXUP_ALPHA: 0.0

- LABEL_SMOOTHING: 0.05


## Outputs

- outputs/eda_counts.csv + outputs/eda_class_counts.png

- outputs/*_predictions.csv + confusion_*.png + classification_report_*.json

- outputs/vit_squat_best.pth + outputs/vit_squat_last.pth + outputs/train_log.csv

- outputs/PoseAITraining_artifact.zip


## Notes

Generation uses 3D pose extraction + 3D pose manipulation (dragging + rotations) + ControlNet Img2Img. Training uses ViT fine-tuning with cosine LR, label smoothing, optional mixup, early stopping.
Train artifacts: {'best_val_acc': 0.77, 'best_path': 'C:\\Users\\rotem\\Desktop\\PoseAITraining_artifact\\outputs\\vit_squat_best.pth', 'last_path': 'C:\\Users\\rotem\\Desktop\\PoseAITraining_artifact\\outputs\\vit_squat_last.pth', 'log_path': 'C:\\Users\\rotem\\Desktop\\PoseAITraining_artifact\\outputs\\train_log.csv'}


## Metrics

### val

- n = 400

- accuracy = 0.7700

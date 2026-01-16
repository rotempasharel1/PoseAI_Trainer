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

- outputs/val_predictions.csv

- outputs/confidence_dist_val.png

- outputs/confusion_val.png

- outputs/roc_curve_val.png

- outputs/README.md


## Notes

Generation uses 3D pose extraction + 3D pose manipulation (dragging + rotations) + ControlNet Img2Img. Training uses ViT fine-tuning with cosine LR, label smoothing, optional mixup, early stopping.
Train artifacts (created then pruned in minimal mode): {'best_val_acc': 0.77, 'best_path': 'C:\\Users\\rotem\\Desktop\\PoseAITraining_artifact\\outputs\\vit_squat_best.pth', 'last_path': 'C:\\Users\\rotem\\Desktop\\PoseAITraining_artifact\\outputs\\vit_squat_last.pth', 'log_path': 'C:\\Users\\rotem\\Desktop\\PoseAITraining_artifact\\outputs\\train_log.csv'}


## Metrics

### val

- n = 400

- accuracy = 0.7700

- f1_macro = 0.7669

- f1_weighted = 0.7669

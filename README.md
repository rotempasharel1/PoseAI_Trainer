# Project Name: PoseAI Trainer

## 📋 Submission Checklist
- **Git Link**: Valid and matches project name.
- **Presentation**: Available in **PPT** and **PDF** in `/interim_report_presentation`.
- **Data**: All synthetic datasets (Train/Val) uploaded as ZIP files.
- **Notebooks**: Includes Section 5.1 (Data Gen), 5.2 (EDA), and 5.3 (Baseline/Training).

## 🚀 How to Run
The code is designed to be **Colab-friendly** and handles dependency installation automatically.

1. **Environment Setup**: 
   - Ensure you have a GPU environment (CUDA) for optimal performance.
   - The script will automatically install required libraries: `mediapipe`, `diffusers`, `transformers`, `torch`, etc.
2. **Data Structure** (Mandatory): The script expects the following directory structure to function correctly:
      PoseAI_Trainer/
      ├── poseAI_trainer.py       # Main execution script (Sections 5.1-5.3)
      ├── seeds/                  # Base images for generation
      │   ├── good/               # Good form seeds
      │   └── bad/                # Bad form seeds
      └── synthetic_dataset/      # Training & Validation data
          ├── train/              # Training images
          │   ├── good/           # Correct form
          │   └── bad/            # Incorrect form
          └── val/                # Validation images
              ├── good/           # Correct form
              └── bad/            # Incorrect form
4. **Execution**:
   - Run the main script/notebook: `poseAI_trainer.py`.
   - **Generation**: By default, generation is disabled to use existing data. To enable, set `ENABLE_GENERATION=1`.
   - **Training**: The script trains a **ViT-B/16** model and saves the best weights to `outputs/vit_squat_best.pth`.
5. **Outputs**:
   - After execution, a ZIP artifact `PoseAITraining_artifact.zip` will be created containing all results and a detailed summary.

## 📂 Repository Layout
*Note: Large ZIP files are kept in the root directory due to size constraints.*

- **`/interim_report_presentation`**:
  - `PoseAI_Trainer.pptx` & `PoseAI_Trainer.pdf`.
- **`outputs/`**: Generated after run (also available in `outputs.zip`):
  - **EDA**: Visuals and stats (`eda_class_counts.png`, `eda_stats.csv`).
  - **Baseline**: Training results (`confusion_test.png`, `test_predictions.csv`).
- **Root Directory**:
  - `poseAI_trainer.py`: Main execution script.
  - `synthetic_dataset_*.zip`: Full datasets.



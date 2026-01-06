# Project Name: PoseAI Trainer

## 📋 Submission Checklist
- **Git Link**: Valid and matches project name.
- **Presentation**: Available in **PPT** and **PDF** in `/interim_report_presentation`.
- **Data**: All synthetic datasets (Train/Val) uploaded as ZIP files.
- **Notebooks**: Includes Section 5.1 (Data Gen), 5.2 (EDA), and 5.3 (Baseline/Training).

## 🚀 How to Run
The code is designed to be **Colab-friendly** and handles dependency installation automatically.

1. **Environment Setup**:
   Tip: A GPU (CUDA) is highly recommended for generation and training.
   Manual Installation (Recommended for Local PC):   - The script will automatically install required libraries: `mediapipe`, `diffusers`, `transformers`, `torch`, etc.
      1. Create and activate a virtual environment
      python -m venv venv
      source venv/bin/activate  # Linux/Mac
      venv\Scripts\activate     # Windows
      
      2. Upgrade pip and install dependencies
      python -m pip install --upgrade pip
      pip install mediapipe diffusers transformers accelerate safetensors opencv-python pillow numpy pandas scikit-learn matplotlib tqdm torch torchvision

     Note: The script poseAI_trainer.py also contains an internal ensure_deps() function to verify these libraries.
      
3. **Data Structure** (Mandatory): The script expects the following directory structure to function correctly:
   <img width="500" height="2000" alt="image" src="https://github.com/user-attachments/assets/1d36269d-4c81-43fb-9723-b21b695395d3" />

4. **Execution**:
   - Run the main script/notebook: `poseAI_trainer.py`.
   - **Generation**: By default, generation is disabled to use existing data. To enable, set `ENABLE_GENERATION=1`.
   - **Training**: The script trains a **ViT-B/16** model and saves the best weights to `outputs/vit_squat_best.pth`.
5. **Outputs**:
   - After execution, a ZIP artifact `PoseAITraining_artifact.zip` will be created containing all results and a detailed summary.

## 📝 Results (CSV)
The final analysis is saved in outputs/test_predictions.csv (or val_predictions.csv). This file includes the core classification and the LLM-generated feedback in the following columns:

keep_1, keep_2: Positive points identified in the performance.

improve_1, improve_2: Points for improvement.

llm_summary: A comprehensive textual summary of the pose analysis.

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






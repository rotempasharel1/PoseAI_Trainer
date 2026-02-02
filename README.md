# PoseAI Trainer 🏋️‍♂️
<img width="752" height="550" alt="unnamed" src="https://github.com/user-attachments/assets/83752a97-7f20-4eba-93f7-feb3ad99b16f" />

## 📖 Project Motivation
Performing physical exercises without proper guidance often leads to incorrect form and potential injuries. The PoseAI Trainer provides an AI-driven model for correcting squat technique, serving as a digital alternative to a human trainer.By training on high-fidelity synthetic data, the model delivers accurate feedback to help ensure safety and improve athletic performance.

## 🎯 Problem Statement
This project focuses on a model that classifies squat performance quality (Good vs. Bad) using a synthetic dataset. This approach overcomes the common shortage of labeled real-world data for specific movement errors by creating dynamic variations in camera angles, lighting, and body types based on a limited set of seed images.

## 🖼️ Visual Abstract 
<img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/a51bd2cc-411f-4b0d-9c9e-4d065053e9aa" />

*The pipeline from a seed image to 3D pose extraction and final synthetic realistic output using ControlNet.*

---

## 📊 Datasets Used or Collected
The model was trained on a comprehensive **Synthetic Dataset** developed through:
* **Source Seeds:** Base skeletons were extracted from real-world images using MediaPipe.
* **Diversity:** We generated numerous variations of camera angles and body structures from a small number of seeds to increase robustness.
* **Quality Control (EDA):** Visual features like sharpness, brightness, and contrast were analyzed to ensure data quality and identify outliers.

## 🔧 Data Augmentation and Generation Methods
Our data generation process utilized **ControlNet** and **Stable Diffusion** to transform poses into realistic images. We performed precise **3D skeleton manipulations** to simulate specific errors (e.g., knees collapsing inward), allowing for the creation of targeted learning examples that are rare in standard datasets.

## 📤 Input/Output Examples
| Input Image | Prediction | Confidence | Status | LLM Kepp Feedback | LLM Improvement Feedback |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ![Good Squat Example](https://github.com/user-attachments/assets/3588a6c9-00c0-409d-bfbb-d5b482239036) | **Good** | 0.860 | ✅ Correct | Knees track in line with the toes (no inward collapse); Heels stay grounded throughout the movement  | Keep a neutral lower back (avoid rounding); Maintain controlled depth without losing balance  |
| ![Bad Squat Example](https://github.com/user-attachments/assets/f4b944de-e942-4b1c-9b52-6a06d1a42fef) | **Bad** | 0.984 | ✅ Correct | Maintain stable foot placement at a comfortable stance width; Keep gaze forward with chest up  | Reduce forward torso lean and keep a neutral spine; Improve depth gradually while maintaining stability  |


*The model outputs a classification label and a confidence level, accompanied by detailed textual feedback powered by an integrated LLM that highlights both technical strengths to maintain and specific areas for improvement.*

---

## 🧠 Models and Pipelines Used
* **Vision Transformer (ViT-B/16):** The core classification model that learns spatial relationships to identify squat qualit.
* **MediaPipe:** Utilized for 3D landmark extraction and skeleton baseline modeling.
* **LLM Integration:** Combined with a Large Language Model to generate explainable tips and instructions for improvement.

## ⚙️ Training Process and Parameters
* **Strategy:** Fine-tuning in two stages: Initial **Warmup** followed by **Unfreezing & Fine-tuning**.
* **Environment:** Designed to be Colab-friendly with automated installation of required libraries such as `mediapipe`, `diffusers`, `transformers`, and `torch`.
---

## 📈 Metrics & Results
Based on the validation analysis:
* **AUC-ROC:** The model demonstrated high discriminative ability with an **$AUC \approx 0.89$**.
* **Classification Accuracy:** In the validation set, the model correctly identified **177 "Bad"** cases and **131 "Good"** cases.
* **Confidence Correlation:** Accuracy increases as the model's confidence levels rise, particularly for predictions with confidence above 0.8.
<p align="center">
   <img width="33%" height="300" alt="roc_curve_val" src="https://github.com/user-attachments/assets/6684dfed-a58c-41f0-b884-f11c3b333a35" />
  <img width="33%" height="300" alt="confusion_val" src="https://github.com/user-attachments/assets/b2d885b5-4428-4328-92b2-dc2c20d9144b" />
  <img width="33%" height="300" alt="confidence_dist_val" src="https://github.com/user-attachments/assets/c715e9f3-90ed-43a9-9520-b9898ecee6cd" />
</p>

## 📂 Repository Structure
* `proposal_presentation/, interim_presentation/, final_presentation/`: Project presentation files across different stages (PPTX/PDF).
* `outputs/`: EDA graphs, confusion matrix, and best model weights (`vit_squat_best.pth`).
* `project_url_drive.txt`: A text file containing the link to download the organized project from Google Drive.  *(Note: The Drive folder includes a comprehensive project walkthrough video created via NotebookLLM)*.
* `poseAI_trainer.py`: Main execution script.
* `synthetic_dataset_*.zip`: Full synthetic datasets split into archives.

---

## 👥 Team Members
* Rotem Pasharel
* Nofar Hatam
* May Eden



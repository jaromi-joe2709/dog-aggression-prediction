# Dog Aggressive Behaviour Prediction

## Project Overview
A dual-branch deep learning system that classifies dog behaviour as
Aggressive or Non-Aggressive from a single image using ResNet50 (visual 
features) and EfficientNet-B2 (posture/keypoint features) with attention-
gated fusion.

## Dependencies / Libraries Required
pip install torch torchvision timm scikit-learn opencv-python 
           pillow splitfolders matplotlib seaborn tqdm numpy pandas

## How to Run
1. Open Dog_Aggressive_Behaviour_prediction.ipynb in Jupyter Notebook / Google Colab
2. Run all cells top to bottom (Kernel → Restart & Run All)
3. Dataset is auto-downloaded from Kaggle (set your Kaggle API key first)
4. Trained model saved as best_dog_behavior_model.pth
5. For video inference, set VIDEO_PATH to your video file and run the last cell

## Modules Description
- Step 1–4  : Dataset loading, label mapping, train/val/test split
- Step 5–6  : Data augmentation, WeightedRandomSampler
- Step 7    : DogBehaviorClassifier model (ResNet50 + EfficientNet-B2 + Attention)
- Step 8–10 : Focal Loss, AdamW optimiser, Cosine Annealing scheduler
- Step 11   : Training loop with best-F1 checkpoint saving
- Step 12   : Evaluation — Accuracy, F1, Recall, AUC-ROC, Confusion Matrix
- Step 13   : Video inference using OpenCV

## Team
Group :Jaromi D,Dayana ,Rakshitha
| Team Lead: [Jaromi D]

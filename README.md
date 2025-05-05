# Action-Localization-WTMR--Net-
# WEIGHTED TIMESFORMER MULTI-REGION CNN NETWORK FOR PERSONALIZED AND CONTEXT-AWARE ACTIVITY LOCALIZATION IN AMBIENT ASSISTED LIVING
# Overview
This repository presents WTMR-Net, a novel architecture that combines the strengths of Context-Aware Multi-Region CNNs with the temporal modeling capabilities of TimeSformer to achieve personalized and context-aware human activity localization in Ambient Assisted Living (AAL) systems.

The project introduces a hybrid model that leverages both spatial and temporal cues to improve action localization and recognition in constrained environments, like elderly care and health monitoring.
# Datasets Used
- **HMDB51:** It is a widely used action detection dataset with 6,766 video frame samples spanning 51 action categories. The RGB video clips are sourced from diverse locations, including movies and online videos, making the dataset more realistic. 
- **NTU RGB+D 60:** This dataset is a large-scale skeleton-based action recognition dataset. NTU RGB+D 60 depicts human motions across 3D skeleton joints. It comprises of 44,888 action clips distributed over 60 categories.
- **SVideoQA:** The SVideoQA dataset is designed for video quality assessment rather than traditional action recognition. The 152 pedestrian frames in the sam-ple largely depict walking, standing, sitting, and looking about. Diversity of ac-tion classes in the pedestrian-oriented dataset is much less than in HMDB51 and NTU RGB+D 60.

# Model Architecture
**Context-Aware Multi-Region CNN Module:** Splits input into distinct spatial regions, applies region-specific convolution, and aggregates features to capture fine-grained motion and spatial cues for robust activity recognition.

**Contextual Feature Embedding:** Transforms region-level features into 8×8 patches, embeds them into a latent space, and adds positional and temporal encodings to maintain spatio-temporal order.

**TimeSformer Attention Block:** Applies global space-time self-attention to the embedded patches, modeling dependencies across frames and regions using multi-head attention and transformer encoders.

**Personalized Activity Tracking:** Uses Haar Cascade to detect individuals, enabling per-user activity logging with timestamps, durations, and contextual keyframes for adaptive AAL behavior modeling.

# Performance Summary
| Dataset   | Model       | Train Accuracy (%) | Params (M) | GFLOPs | Throughput (clips/s) |
| --------- | ----------- | ------------------ | ---------- | ------ | -------------------- |
| HMDB51    | TimeSformer | 15.06              | 42.75      | 133.91 | 6.16                 |
|           | WTMR-Net    | 98.41              | 91.43      | 86.92  | 12.88                |
| NTU RGB+D | TimeSformer | 60.51              | 4.75       | 1.90   | 273.33               |
|           | WTMR-Net    | 98.59              | 4.81       | 1.92   | 288.35               |
| SVideoQA  | TimeSformer | 47.24              | 90.44      | 268.52 | 3.12                 |
|           | WTMR-Net    | 63.45              | 92.61      | 6.94   | 99.07                |

# Evaluation & Visualization
- Confusion Matrices across all datasets for comparative insights
- Ablation Studies for each module (Context-Aware MultiRegionCNN Module, TimeSformer Block)
- Localized Action Frames with bounding boxes and contextual annotations
- Detailed experiment logs and performance graphs

# Requirements
- Python 3.8+
- PyTorch ≥ 1.12
- OpenCV, NumPy, Matplotlib
- timm, torchvision, einops, tqdm
- two NVIDIA Tesla T4 GPUs

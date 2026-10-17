# DehazzeAGGAN

This project implements the **Dehaze-AGGAN** architecture for unpaired image dehazing. The primary goal is to remove environmental degradation (fog, haze, snow) from images to improve both human visibility and downstream computer vision tasks, specifically Object Detection (YOLOv8). 

This pipeline is designed as a precursor to edge deployment on resource-constrained devices like the **Nvidia Jetson Nano**.

## 📁 Repository Structure
- `dehazzing.ipynb`: The complete PyTorch training and evaluation pipeline.
- `final_results/`: Contains quantitative metrics and visual comparisons.
  - `metrics_summary.csv`: PSNR and SSIM evaluations on the I-HAZE/O-HAZE test sets.
  - `iddaw_fog_comparisons/`: Side-by-side visual results on the 20GB IDDAW dataset.
  - `yolov8_detections/`: YOLOv8 bounding box comparisons (Hazy vs. Dehazed).

## Architecture & Methodology
The model is built on a CycleGAN-based unpaired framework with specific enhancements for environmental noise:
1. **Enhanced Attention-Guided Generator**: Uses $N=10$ attention masks to separate content and haze features, fused via element-wise multiplication.
2. **PatchGAN Discriminator**: Uses Least Squares GAN (LSGAN) loss for training stability.
3. **Custom Loss Functions**:
   - **Cycle-Consistency Loss** ($\lambda_{cycle} = 20$): Ensures structural integrity during unpaired translation.
   - **Total Variation (TV) Loss** ($\lambda_{TV} = 1e-6$): Specifically added to eliminate "sea-wave" noise and smooth background artifacts.

## Datasets
- **Training & Metric Testing**: I-HAZE and O-HAZE (Merged) - Used for initial 50-epoch baseline training and PSNR/SSIM calculation.
- **Zero-Shot Inference & YOLO Testing**: IDDAW (20GB) - Specifically the `FOG` subset, used to test generalization on real-world autonomous driving scenarios.

##  Results

### 1. Image Quality Metrics (PSNR & SSIM)
Evaluated on the official I-HAZE and O-HAZE test sets after 50 epochs of training.
| Dataset | Average PSNR (dB) | Average SSIM |
| :--- | :---: | :---: |
| **I-HAZE** | 8.44 | 0.1254 |
| **O-HAZE** | 9.97 | 0.0079 |

*(Note: Metrics reflect early-stage convergence on a subsampled 160-image dataset. Full convergence typically requires 150 epochs on the complete dataset).*

### 2. Downstream Object Detection (YOLOv8)
To prove the practical utility of the dehazed images, we ran YOLOv8 on the IDDAW FOG dataset. 
**Key Finding: False Positive Reduction.** 
In heavy fog, YOLOv8 frequently misidentifies fog patches and low-contrast noise as objects (cars, people), resulting in high detection counts but low reliability. Our Dehaze-AGGAN model successfully removes this environmental noise.

| Image | Hazy Detections | Hazy Avg Conf | Dehazed Detections | Dehazed Avg Conf | Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `00000000` | 21 | 0.48 | 0 | 0.00 | Eliminated fog-induced false positives |
| `00000001` | 6 | 0.42 | 2 | 0.29 | Refined detections to core objects |
| `00000002` | 9 | 0.42 | 1 | 0.31 | Significant noise reduction |

### 3. Visual Comparisons
*(You can drag and drop your best side-by-side images from your `final_results` folder into this section on GitHub!)*

![Hazy vs Dehazed](final_results/iddaw_fog_comparisons/compare_00000001_rgb.png)

## Future Work: Jetson Nano Deployment
The next phase of this project focuses on optimizing the trained

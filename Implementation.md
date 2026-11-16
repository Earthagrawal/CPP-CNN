# CatDogCNN: C++ & CUDA Deep Learning Framework from Scratch

## 1. Executive Summary

**CatDogCNN** is a modular, high-performance Convolutional Neural Network (CNN) framework engineered in modern C++17 and NVIDIA CUDA C++. Built entirely from foundational principles without relying on third-party deep learning frameworks (such as PyTorch, TensorFlow, or LibTorch), the framework provides an end-to-end pipeline covering image preprocessing, dataset stratification, mini-batch streaming DataLoader, GPU memory management, custom CUDA kernels, cuBLAS acceleration, forward and backward autodiff propagation, modular neural network layers, loss functions, optimizers, evaluation metrics, and complete checkpoint serialization.

---

## 2. Project Directory Structure

```text
CatDogCNN/
│
├── core/                           # Core numerical & GPU acceleration engine
│   ├── CUDABackend.hpp / .cpp     # Unified GPU/CPU runtime, custom CUDA kernels, cuBLAS wrappers
│   ├── Tensor.hpp / .cpp          # Multi-dimensional N-D tensor with unified host/device memory
│   ├── Matrix.hpp / .cpp          # 2D matrix abstraction with GEMM capabilities
│   └── Parameter.hpp / .cpp       # Trainable model parameter container (data, grad, velocity)
│
├── data/                           # Dataset handling, stratification & mini-batch DataLoader
│   ├── ImageReader.hpp / .cpp     # Multi-format raw image reader (JPEG/PNG via stb_image)
│   ├── LabelMapper.hpp / .cpp     # String-to-integer class label mapping (Cat=0, Dog=1)
│   ├── Dataset.hpp / .cpp         # Dataset indexing and raw/processed manifest handling
│   ├── DatasetSplitter.hpp / .cpp # Stratified deterministic dataset partitioner (80/10/10, seed 42)
│   └── DataLoader.hpp / .cpp      # Mini-batch loader (PNG -> FP32 [0,1], HWC -> NCHW, shuffle, CUDA transfer)
│
├── preprocessing/                  # Image preprocessing pipeline (224x224 RGB)
│   ├── ImageInspector.hpp / .cpp  # Header validation, corruption check, dimension extraction
│   ├── Resize.hpp / .cpp          # Bilinear aspect-preserving image resizer
│   ├── Padder.hpp / .cpp          # Reflection and constant letterbox padding
│   ├── Normalization.hpp / .cpp   # uint8 to FP32 scaling [0, 1] & channel standardization
│   └── PreprocessingRunner.hpp/.cpp # Multi-threaded parallel preprocessing pipeline
│
├── layers/                         # Modular neural network layers (Dual CPU & CUDA)
│   ├── Layer.hpp                  # Abstract layer interface (forward, backward, toDevice, train/eval)
│   ├── Conv2D.hpp / .cpp          # 2D Convolution with Xavier initialization & im2col / direct kernels
│   ├── BatchNorm2D.hpp / .cpp     # 2D Batch Normalization with learnable gamma/beta & running statistics
│   ├── ReLU.hpp / .cpp            # Rectified Linear Unit activation
│   ├── MaxPool2D.hpp / .cpp       # 2D Max Pooling with spatial argmax tracking for backprop
│   ├── GlobalAveragePool2D.hpp/.cpp # Spatial reduction (HxW -> 1x1) pooling
│   └── Linear.hpp / .cpp          # Fully connected layer accelerated by cuBLAS SGEMM
│
├── loss/                           # Objective functions
│   ├── Loss.hpp                   # Abstract loss interface
│   └── BCEWithLogitsLoss.hpp/.cpp # Numerically stable Binary Cross Entropy with Logits
│
├── optimizers/                     # Parameter optimization
│   ├── Optimizer.hpp              # Abstract optimizer interface
│   └── SGD.hpp / .cpp             # Stochastic Gradient Descent with momentum and weight decay
│
├── model/                          # Composed neural network models
│   └── CatDogCNNModel.hpp / .cpp  # End-to-end 3-Block CNN classifier
│
├── evaluation/                     # Metric evaluation & reporting
│   └── Evaluator.hpp / .cpp       # Accuracy, Precision, Recall, F1, and Confusion Matrix
│
├── training/                       # Training engine & standalone runners
│   ├── Trainer.hpp / .cpp         # Unified training loop, validation, checkpoint serialization
│   └── train_one_epoch_bs64.cpp   # Verified standalone 1-epoch fresh runner (Batch Size 64)
│
├── tests/                          # Automated verification suites
│   ├── preprocessing_test.cpp     # Preprocessing pipeline verification
│   ├── cnn_foundation_test.cpp    # CPU foundation layer test
│   ├── gpu_verification_test.cpp  # Physical NVIDIA GTX 1650 CUDA execution benchmarks
│   ├── model_eval_test.cpp        # 9-part test suite (Model, Eval, DataLoader, Splits, CPU vs CUDA)
│   └── batch_benchmark.cpp        # Empirical throughput benchmark (Batch Sizes 8, 32, 64, 128)
│
├── vendor/                         # Third-party single-header dependencies
│   ├── stb_image.h
│   ├── stb_image_write.h
│   └── stb_impl.cpp
│
├── Implementation.md              # Project technical documentation and execution guide
└── main.cpp                       # CLI entry point

checkpoints/
├── checkpoint_epoch_1_bs64.bin    # Trained Epoch 1 checkpoint (785 KB)
└── latest_checkpoint.bin          # Latest resumable checkpoint pointer (785 KB)

dataset/
├── PetImages/                      # Raw dataset (Cat/, Dog/)
├── noisy_images/                   # Ignored noise directory
├── processed/
│   ├── images/                     # 24,989 preprocessed 224x224 RGB PNG images
│   └── labels.csv                  # Processed label manifest (Cat=0, Dog=1)
├── split_manifest.csv              # Deterministic shared stratified split manifest (seed 42)
├── train.csv                       # 19,992 training samples (80.0%)
├── validation.csv                  # 2,499 validation samples (10.0%)
└── test.csv                        # 2,498 test samples (10.0%)
```

---

## 3. Network Architecture Specifications

The `CatDogCNNModel` processes RGB images with dimension $[N, 3, 224, 224]$:

```
Input [N, 3, 224, 224] (FP32, [0, 1])
  │
  ├── [Block 1]
  │     ├── Conv2D (in=3, out=32, kernel=3x3, stride=1, padding=1)  ──> [N, 32, 224, 224]
  │     ├── BatchNorm2D (features=32, eps=1e-5, momentum=0.1)       ──> [N, 32, 224, 224]
  │     ├── ReLU Activation                                         ──> [N, 32, 224, 224]
  │     └── MaxPool2D (pool=2x2, stride=2)                          ──> [N, 32, 112, 112]
  │
  ├── [Block 2]
  │     ├── Conv2D (in=32, out=64, kernel=3x3, stride=1, padding=1) ──> [N, 64, 112, 112]
  │     ├── BatchNorm2D (features=64, eps=1e-5, momentum=0.1)       ──> [N, 64, 112, 112]
  │     ├── ReLU Activation                                         ──> [N, 64, 112, 112]
  │     └── MaxPool2D (pool=2x2, stride=2)                          ──> [N, 64, 56, 56]
  │
  ├── [Block 3]
  │     ├── Conv2D (in=64, out=128, kernel=3x3, stride=1, padding=1)──> [N, 128, 56, 56]
  │     ├── BatchNorm2D (features=128, eps=1e-5, momentum=0.1)      ──> [N, 128, 56, 56]
  │     ├── ReLU Activation                                         ──> [N, 128, 56, 56]
  │     └── MaxPool2D (pool=2x2, stride=2)                          ──> [N, 128, 28, 28]
  │
  ├── [Global Pooling]
  │     └── GlobalAveragePool2D (28x28 spatial reduction)           ──> [N, 128]
  │
  └── [Classifier Head]
        ├── Linear (in=128, out=32)                                 ──> [N, 32]
        ├── ReLU Activation                                         ──> [N, 32]
        └── Linear (in=32, out=1)                                   ──> [N, 1] (Raw Logits)
```

### Parameter Count Breakdown

| Layer | Parameter Name | Dimensions | Number of Elements |
|---|---|---|---|
| Conv1 | `conv1.weight` | $[32, 3, 3, 3]$ | $864$ |
| Conv1 | `conv1.bias` | $[32]$ | $32$ |
| BN1 | `bn1.gamma` | $[32]$ | $32$ |
| BN1 | `bn1.beta` | $[32]$ | $32$ |
| Conv2 | `conv2.weight` | $[64, 32, 3, 3]$ | $18,432$ |
| Conv2 | `conv2.bias` | $[64]$ | $64$ |
| BN2 | `bn2.gamma` | $[64]$ | $64$ |
| BN2 | `bn2.beta` | $[64]$ | $64$ |
| Conv3 | `conv3.weight` | $[128, 64, 3, 3]$ | $73,728$ |
| Conv3 | `conv3.bias` | $[128]$ | $128$ |
| BN3 | `bn3.gamma` | $[128]$ | $128$ |
| BN3 | `bn3.beta` | $[128]$ | $128$ |
| FC1 | `fc1.weight` | $[32, 128]$ | $4,096$ |
| FC1 | `fc1.bias` | $[32]$ | $32$ |
| FC2 | `fc2.weight` | $[1, 32]$ | $32$ |
| FC2 | `fc2.bias` | $[1]$ | $1$ |
| **Total** | **16 Parameters** | — | **97,825 parameters (~391.3 KB FP32)** |

---

## 4. Phase 5: Fresh 1-Epoch Training Results (Batch Size 64)

### 4.1. Training Configuration
- **Compute Backend:** NVIDIA GeForce GTX 1650 (CUDA 13.3, Compute Capability 7.5)
- **Batch Size:** 64
- **Epochs:** 1 (Fresh start, no checkpoints reused)
- **Optimizer:** SGD ($\eta = 0.01$, $\text{momentum} = 0.9$, $\text{weight decay} = 10^{-4}$)
- **Loss:** `BCEWithLogitsLoss`
- **Training Samples:** 19,992 images (313 batches)
- **Validation Samples:** 2,499 images (40 batches)

### 4.2. Before vs. After Training Comparison Table

| Metric | Before Training (Epoch 0 Baseline) | After 1 Epoch (Epoch 1) | Change |
|---|:---:|:---:|:---:|
| **BCE Loss** | **0.7315** | **0.6808** | **-0.0507** |
| **Accuracy** | **49.9800%** | **57.9032%** | **+7.9232%** |
| **Precision** | **0.0000%** | **89.2857%** | **+89.2857%** |
| **Recall** | **0.0000%** | **18.0000%** | **+18.0000%** |
| **F1-Score** | **0.0000%** | **29.9601%** | **+29.9601%** |

### 4.3. Confusion Matrix Progression
- **Before Training:**
  - $\text{TN} = 1249$, $\text{FP} = 0$, $\text{FN} = 1250$, $\text{TP} = 0$ (untrained model output negative logits for all samples; 0% Dog detection)
- **After 1 Epoch:**
  - $\text{TN} = 1222$, $\text{FP} = 27$, $\text{FN} = 1025$, $\text{TP} = 225$
  - Model successfully learned to recognize dogs with **89.29% precision** while retaining **97.84% true negative rate on cats**.

### 4.4. Training Trajectory
- Batch 25 / 313: Loss = 0.6922, Acc = 56.56%
- Batch 100 / 313: Loss = 0.6562, Acc = 60.67%
- Batch 200 / 313: Loss = 0.6444, Acc = 61.89%
- Batch 313 / 313: Loss = 0.6327, Acc = 63.06%
- **Final Training Loss:** 0.6327 | **Final Training Accuracy:** 63.06%

---

## 5. Checkpoint Storage & Resumption

Trained model checkpoints are saved in `checkpoints/`:
- `checkpoints/checkpoint_epoch_1_bs64.bin` (785,034 bytes)
- `checkpoints/latest_checkpoint.bin` (785,034 bytes)

Each checkpoint binary contains:
- Magic header (`CDCNN01`)
- Epoch number (1)
- Learning rate, momentum, and weight decay
- All 16 parameter weight tensors and momentum velocity buffers
- Running mean and running variance buffers for `bn1`, `bn2`, `bn3`
- Full evaluation results and confusion matrix

### To Resume Training for Additional Epochs:
```powershell
./CatDogCNN/main.exe --train --resume checkpoints/latest_checkpoint.bin --epochs 5 --batch-size 64
```

---

## 6. cuBLAS GEMM & Multi-Threaded Prefetching Optimization

### 6.1. Optimization Motivation & Technical Architecture
To match PyTorch's execution performance without leaving C++:
1. **Conv2D Forward:** Converted spatial feature map unrolling to coalesced `im2col` followed by **NVIDIA cuBLAS `cublasSgemm`** ($Y = W \times \text{Col}$).
2. **Conv2D Backward (Zero `atomicAdd`):**
   - Weight Gradient: $\nabla W = \nabla Y \times (\text{Col})^T$ via **cuBLAS `cublasSgemm`** (accumulating across batch with $\beta = 1.0f$).
   - Input Gradient: $\nabla Col = W^T \times \nabla Y$ via **cuBLAS `cublasSgemm`**, followed by parallel `col2im` (where each thread writes its own unique pixel with zero atomics).
   - Bias Gradient: Vectorized shared-memory channel reduction kernel.
3. **Multi-Threaded DataLoader with Asynchronous Prefetching:**
   - 8 concurrent worker threads decode PNG images simultaneously across CPU cores.
   - Background worker asynchronously loads and prepares Batch $k+1$ in pinned host/GPU memory while Batch $k$ runs on the GPU.

### 6.2. Benchmark Results on NVIDIA GTX 1650 (4 GB VRAM)

| Batch Size | GPU Fwd+Bwd+Step (ms) | Data Loading (ms) | Total Time / Batch (ms) | VRAM Used | 1-Epoch Train Time |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 552.4 ms | 317.6 ms | 870.0 ms | 1,700 MB (41.5%) | **9.1 minutes** |
| **64** | **958.4 ms** | **268.8 ms** | **1,227.2 ms** | **2,596 MB (63.4%)** | **6.4 minutes** |
| **128** | 11,982.3 ms | 0.0 ms | 11,982.3 ms | 4,095 MB (100.0%) | 31.4 minutes (paging) |

### 6.3. Verification Summary
- **Unit Test Suite (`model_eval_test_gpu.exe`):** 9/9 tests passed (100%).
- **GPU Test Suite (`gpu_verification_test.exe`):** 10/10 tests passed (100%).
- **Numerical Parity:** Conv2D forward, backward input gradients, and weight gradients match the mathematical CPU ground truth within $10^{-5}$ tolerance.
- **Speedup:** Reduced 1-epoch training time from **137.6 minutes** to **~9.2 minutes** (**~15x faster end-to-end**).

### 6.4. Live Verification: Epoch 2 Training Run
Using the cuBLAS and parallel prefetch pipeline, Epoch 2 was trained by resuming directly from `checkpoints/checkpoint_epoch_1_bs64.bin`:
- **Training Duration:** 9m 13s (Train: 8m 29s across 19,992 images; Full Validation: 43s across 2,499 images)
- **Time Reduction vs Epoch 1:** **137.6 minutes $\to$ 9.2 minutes (Saved over 128 minutes!)**
- **Training Loss / Acc:** Loss dropped $0.6327 \to 0.5946$ | Acc reached **67.53%**
- **Validation Metrics (2,499 images):**
  - **Val Loss:** $0.6808 \to \mathbf{0.5818}$ (-0.0990)
  - **Val Accuracy:** $57.90\% \to \mathbf{69.59\%}$ (+11.69%)
  - **Dog Precision:** **77.10%**
  - **Dog Recall:** $18.00\% \to \mathbf{55.76\%}$ (+37.76%)
  - **F1-Score:** $29.96\% \to \mathbf{64.72\%}$ (+34.76%)
  - **Confusion Matrix:** TN=1042 (Cat), FP=207, FN=553, TP=697 (Dog)
- **Checkpoints Saved:** `checkpoints/checkpoint_epoch_2.bin` and `checkpoints/best_model.bin` (785 KB).

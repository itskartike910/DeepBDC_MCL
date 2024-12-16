
# **DeepBDC in Mutual Centralized Learning (MCL) for Few-shot Classification**

## **Overview**

This repository implements **Deep Brownian Distance Covariance (DeepBDC)** integrated into **Mutual Centralized Learning (MCL)** for few-shot image classification tasks. The method combines DeepBDC for extracting rich and non-linear feature representations and MCL for identifying the class with the strongest mutual affiliation to the query image through bidirectional random walks. Our approach achieves high accuracy on standard few-shot benchmarks like **miniImageNet** and **tieredImageNet**.

---

![image](https://github.com/user-attachments/assets/4d39f040-824d-4d75-a1b6-0e0243356677)


---

## **Code Prerequisites**

Before running the code, ensure the following dependencies are installed:

- [PyTorch >= version 1.4](https://pytorch.org)
- [tensorboard](https://www.tensorflow.org/tensorboard)
- Optional (based on methods):
  - OpenCV (for DeepEMD)
  - qpth, cvxpy (for MetaOptNet)

---

## **Dataset Preparation**

The datasets (e.g., **miniImageNet**, **tieredImageNet**) should be organized as follows:

```bash
MCL
├── data
│   ├── miniImagenet
│   │   ├── train
│   │   │   ├── n01532829
│   │   │   │   ├── n0153282900000987.png
│   │   ├── val
│   │   ├── test
```

You can download the datasets from [DeepEMD Dataset Link](https://drive.google.com/drive/folders/1sXJgi9pXo8i3Jj1nk08Sxo6x7dAQjf9u?usp=sharing).

---

## **Usage**

### **1. Pretraining**

Pretraining the backbone model using FRN, DN4, or a linear classifier.  
For example, FRN pretraining with ResNet-12 on miniImageNet:

```bash
python experiments/run_pretrainer.py --cfg ./configs/miniImagenet/ResNet-12/R12.yaml --device 0
```

The pretrained model and logs are saved in:

```bash
snapshots/ResNet-12/pretrainer/
```

---

### **2. Meta-Training**

To train DeepBDC integrated with MCL for few-shot learning:  
For example, a 5-way 1-shot experiment with ResNet-12 backbone:

```shell
sh ./fast_train_test.sh ./configs/miniImagenet/ResNet-12/R12.yaml 0
```

---

### **3. Meta-Evaluation**

Evaluate the meta-trained model:  
Example for ResNet-12 backbone with MCL-Katz:

```bash
python experiments/run_evaluator.py --cfg ./snapshots/ResNet-12/MCL/miniImagenet_MCL_N5K1.yaml -c ./snapshots/ResNet-12/MCL/ebest_5way_1shot.pth --device 0
```

---

## **Results**

### **Few-Shot Classification Results**

We report results on standard few-shot benchmarks (**miniImageNet** and **tieredImageNet**) with **ResNet-12** and **Conv-4** backbones. Results include average accuracy over 10,000 randomly sampled episodes for both **1-shot** and **5-shot** tasks.

| Method                 | Backbone    | 1-shot (%) | 5-shot (%) |
|------------------------|-------------|------------|------------|
| DeepBDC-MCL (ours)     | ResNet-12   | **67.5**   | **82.4**   |
| DeepBDC-MCL (ours)     | Conv-4      | **55.7**   | **72.1**   |

## **Folder Structure**

Here’s an overview of the key project structure:

```bash
MCL
├── data/                      # Dataset folder (e.g., miniImageNet)
├── configs/                   # Configuration files for training/evaluation
├── experiments/               # Scripts for pretraining, training, and evaluation
├── snapshots/                 # Saved models and logs
├── README_imgs/               # Images for README visualization
├── modules/                   # Implementation of MCL and DeepBDC methods
└── fast_train_test.sh         # Shell script for fast train and test
```

---

## **Key Features**

1. **DeepBDC for Feature Extraction**:
   - Extracts non-linear, rich feature representations through Brownian Distance Covariance.
   - Improves the quality of feature maps for few-shot classification tasks.

2. **Mutual Centralized Learning (MCL)**:
   - Uses bidirectional random walks to explore feature connections.
   - Stable stationary distributions determine the class with the strongest affiliation.

3. **Flexible Backbone**:
   - Compatible with backbones like ResNet-12 and Conv-4.

4. **Few-Shot Learning Tasks**:
   - Supports N-way K-shot learning tasks with various configurations (1-shot, 5-shot).

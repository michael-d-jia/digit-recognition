# Digit Recognition using CNN (MNIST)

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from the **MNIST dataset**.  
It demonstrates image preprocessing, model training, and evaluation using modern deep learning techniques.

---

## Features
- Loads and preprocesses the MNIST dataset
- Builds a CNN using TensorFlow/Keras
- Trains the model with validation tracking
- Evaluates accuracy on unseen test data
- Optionally saves and reloads the trained model

---

## Model Architecture
- **Input:** 28×28 grayscale images  
- **Layers:**
  1. Conv2D (32 filters, 3×3, ReLU)
  2. Conv2D (64 filters, 3×3, ReLU)
  3. MaxPooling2D (2×2)
  4. Dropout (0.25)
  5. Flatten
  6. Dense (128, ReLU)
  7. Dropout (0.5)
  8. Dense (10, Softmax)

---

## Results
| Metric | Value |
|--------|--------|
| **Training Accuracy** | ~99% |
| **Test Accuracy** | ~98% |
| **Loss Curve** | Smooth convergence after ~10 epochs |

*(These values depend on training parameters and random seeds.)*

---

## Requirements
Install dependencies with:
```bash
pip install tensorflow numpy matplotlib

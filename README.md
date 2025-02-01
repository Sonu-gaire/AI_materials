# Machine Learning Projects Collection

This repository contains several Jupyter notebooks implementing different machine learning algorithms and exercises.

## Projects Overview

### 1. Neural Networks Exercise
**File**: `Neural_networks_exercises.ipynb`
- Uses a synthetic dataset with 4 spiral-shaped clusters for classification
- Implements a neural network to achieve 99.5% classification accuracy
- Dataset contains datapoints x[i]=[x1[i],x2[i]] with labels y[i]=0,1,2,3

### 2. Convolutional Neural Networks Exercise
**File**: `Convolutional_neural_networks_exercises.ipynb`
- Uses two main datasets:
  1. Hand Gesture Dataset
     - Image size: 100x100 pixels
     - Color space: RGB
     - 10 classes (Digits 0-9)
     - 218 participant students
     - 10 samples per student
  2. TensorFlow Flowers Dataset
     - Source: tensorflow.org/datasets/catalog/tf_flowers

### 3. Logistic Regression Exercise
**File**: `Logistic_regression_exercises.ipynb`
- Uses the Breast Cancer Wisconsin (Diagnostic) Dataset
- Dataset characteristics:
  - 569 datapoints
  - 30 variables/features
  - 2 classes (Malignant and Benign)
  - Features include measurements like radius, texture, perimeter, area, etc.

### 4. Introduction to AI Solution
**File**: `intro_to_ai_solution (1).ipynb`
- Uses the Combined Cycle Power Plant dataset
- Implements various machine learning techniques

## Requirements

To run these notebooks, you'll need:
- Python 3.x
- Jupyter Notebook/Lab
- Required Python packages:
  - numpy
  - matplotlib
  - pandas
  - scikit-learn
  - tensorflow (for CNN exercises)
  - keras

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install numpy matplotlib pandas scikit-learn tensorflow keras
```

## Running the Code

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the desired notebook file (.ipynb)

3. Run the cells in sequence (Shift + Enter)

Note: Some notebooks may require specific datasets to be placed in the same directory as the notebook.

## Dataset Sources

- Breast Cancer Wisconsin Dataset: Available through scikit-learn
- TensorFlow Flowers Dataset: Available through TensorFlow datasets
- Other datasets are either synthetic or included in the notebooks

## Project Structure

```
.
├── Neural_networks_exercises.ipynb
├── Convolutional_neural_networks_exercises.ipynb
├── Logistic_regression_exercises.ipynb
└── intro_to_ai_solution (1).ipynb
```

Each notebook is self-contained with its own dataset loading, preprocessing, model training, and evaluation code.

## Notes

- Make sure to run the cells in sequence as later cells may depend on variables or models defined in earlier cells
- Some notebooks may require GPU support for efficient training of neural networks
- The notebooks contain both implementation code and explanatory markdown cells

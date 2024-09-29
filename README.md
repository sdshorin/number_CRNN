# Handwritten Number Recognition using CRNN

This project implements a Convolutional Recurrent Neural Network (CRNN) for recognizing handwritten numbers. The system is designed to recognize numeric sequences of up to five digits (0 to 99999) from images.

## Project Overview

The main goal of this project is to develop and train a neural network capable of accurately recognizing numeric sequences of up to five digits from images of size 32x128 pixels. The model is intended for integration into a mobile application - an educational math game for children, developed using the Godot game engine for iOS and Android platforms.

## Key Features

1. **CRNN Architecture**: Utilizes a Convolutional Recurrent Neural Network combining CNN and RNN layers for effective sequence recognition.
2. **CTC Loss**: Implements Connectionist Temporal Classification loss for training without explicit alignment between input and target sequences.
3. **Custom Dataset Generation**: Combines MNIST dataset with custom-drawn digits to create a diverse training set.
4. **Data Augmentation**: Applies various transformations to generate a large, diverse dataset of handwritten numbers.
5. **Model Optimization**: Experiments with different model sizes to find the optimal balance between accuracy and model size.
6. **Quantization**: Applies model quantization to reduce the size of the trained models.


## Detailed Components

### Data Generation (`numbers_generator.py`)

- Combines MNIST dataset with 1,852 custom-drawn digit images.
- Generates synthetic images of number sequences (1-5 digits).
- Applies augmentation techniques:
  - Random rotation (-10 to 10 degrees)
  - Random scaling (90-110% of original size)
  - Random shifts (up to 4 pixels vertically and horizontally)
  - Variable spacing between digits
- Produces 32x128 pixel images, right-aligned with padding.

### Model Architecture (`models.py`)

Three CRNN variants were implemented and tested:

1. **Base Model**: 
   - 7 convolutional layers (max 512 filters)
   - 2 bidirectional LSTM layers

2. **Optimized Model**: 
   - 7 convolutional layers (max 256 filters)
   - 1 bidirectional LSTM layer

3. **Lightweight Model**: 
   - 5 convolutional layers (max 64 filters)
   - 2 bidirectional LSTM layers

### Training (`train.py`)

- Dataset: 120,000 images, regenerated each epoch
- Batch size: 64 images
- Epochs: 75
- Optimizer: Adam (learning rate: 0.001)
- Loss function: CTC Loss

### Model Evaluation and Selection

| Model      | Accuracy | Training Time (V100) | Size (MB) | Size After Quantization (MB) | Accuracy After Quantization |
|------------|----------|----------------------|-----------|------------------------------|----------------------------|
| Base       | 0.9749   | 73.04 s              | 33.3      | 8.4                          | 0.9749                     |
| Optimized  | 0.9815   | 23.04 s              | 9.8       | 2.5                          | 0.9814                     |
| Lightweight| 0.9267   | 25.00 s              | 8.3       | 2.1                          | 0.9274                     |


### ONNX Export (`export_to_onnx.py`)

Exports trained PyTorch models to ONNX format for potential cross-platform deployment.

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Generate synthetic data:
   ```
   python numbers_generator.py
   ```

3. Train the model:
   ```
   python train.py
   ```

4. Export to ONNX:
   ```
   python export_to_onnx.py
   ```

## Acknowledgements

This project was developed as part of a course project at the National Research University Higher School of Economics, Faculty of Computer Science, Applied Mathematics and Computer Science program.

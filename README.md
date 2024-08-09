# A Neural Network using NumPy for MNIST

**Note**

For 2-D arrays, np.dot performs matrix multiplication. To be more readable and explicit:
- **np.matmul()** is used for matrix multiplication
- "*" is used for element-wise multiplication




## Overview

This project is an implementation of a basic neural network using only **NumPy**, without relying on deep learning frameworks like PyTorch or TensorFlow. The network is trained on the MNIST dataset (accessible via Google Colab), and it classifies handwritten digits into one of ten categories (0-9).




## Key Features

- **Fully Connected Neural Network**: The network consists of 4 layers: an input layer, 2 hidden layers, and an output layer.
- **Custom Activation and Helper Functions**: Efficiently implemented functions such as ReLU, Softmax, One-hot Encoding, Cross-entropy loss, and ReLU derivative (ReLU prime).
- **Custom Training Loop**: The training loop is implemented from scratch, offering detailed insights into *vectorized* forward and backward propagation, and gradient descent optimization.
- **Visualization**: Includes functionality to visualize predictions and a confusion matrix for evaluating model performance.
- **Weights & Biases Integration**: Integrated with [Weights & Biases](https://wandb.ai/) for tracking experiments, logging metrics, and visualizing the training process.




## Project Structure

- `Neural_Network_in_NP.ipynb`: The Jupyter Notebook where the entire neural network is implemented, available in Google Colab.
- `data/`: The MNIST dataset is accessible in Google Colab under `/content/sample_data/mnist_train_small.csv` and `/content/sample_data/mnist_test.csv`. A cell to load the full version of the dataset from `sklearn.datasets` is also included for convenience.




## Results

With an optimized learning rate of `5.5e-2`, the model achieved a **94% validation accuracy**. Below are the results:

### Confusion Matrix:
![Confusion Matrix](confusion_matrix_MNIST.png)

### Example Predictions:
![Sample Predictions](digits&predictions.png)

### Hyperparameter Optimization:
The hyperparameter optimization was logged using Weights & Biases:

<iframe src="https://wandb.ai/gfs3-university-of-illinois-urbana-champaign/mnist-neural-network/reports/MNIST-prediction-with-a-NumPy-Neural-Network---Vmlldzo4OTc1MDA4" width="100%" height="1024px"></iframe>




## Requirements

To run this project, you'll need to install the following Python packages:

### Core Dependencies
- **NumPy**: Used for matrix operations and core numerical computations.
- **Pandas**: Used for loading and preprocessing the MNIST dataset.

### Visualization
- **Matplotlib**: Used for plotting images and graphs.
- **Seaborn**: Used for creating a confusion matrix plot.

### Model Evaluation
- **Scikit-learn**: Used for calculating accuracy and generating the confusion matrix.

### Experiment Tracking
- **Weights & Biases**: Used for logging metrics and visualizing the training process.
- **TQDM**: Used for displaying progress bars during training.




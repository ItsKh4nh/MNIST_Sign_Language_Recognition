# MNIST Sign Language Recognition

## Project Overview
This project implements a deep learning system for recognizing American Sign Language (ASL) letters using the MNIST Sign Language dataset. The model is built using Convolutional Neural Networks (CNN) to classify hand gesture images into different letters of the alphabet (excluding J and Z which require motions).

The classifier achieves high accuracy in recognizing static hand gestures, providing a foundation for ASL recognition applications that could help bridge communication gaps between the deaf/hard-of-hearing community and others.

## Project Structure
```
MNIST_Sign_Language_Recognition/
├── data/               # Dataset directory
│   ├── test/          
│   ├── train/          
├── models/             # Saved model files
├── main.py             # Main script for training and inference
├── main.ipynb          # Jupyter notebook version
└── requirements.txt    # Project dependencies
```

## Technologies Used
- **Python 3.x**
- **Data Processing**: NumPy, Pandas
- **Deep Learning**: TensorFlow, Keras
- **Machine Learning**: Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn

## Dataset
The model is trained on the [Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) from Kaggle, which contains 28x28 grayscale images of hand gestures representing:
- 24 letters (A-Z excluding J and Z which require motion)
- 27,455 training examples
- 7,172 test examples

## Model Architecture
The model uses a deep CNN architecture with:
- Multiple convolutional blocks with batch normalization
- MaxPooling and dropout for regularization
- Dense layers for final classification
- Data augmentation to improve generalization

## Results
The model achieves pin-point perfect classification performance:

| Metric      | Score | Support |
|-------------|-------|---------|
| accuracy    | 1.00  | 7172    |
| macro avg   | 1.00  | 7172    |
| weighted avg| 1.00  | 7172    |

The model successfully recognizes all 24 letters in the test dataset with 100% Accuracy.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MNIST_Sign_Language_Recognition
cd MNIST_Sign_Language_Recognition

# Create a virtual environment (Optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle
# Place it in the data/ directory
```

## Usage

### Training the Model
To train the model from scratch:

```bash
python main.py
```
or
```bash
jupyter notebook main.ipynb
```

This will:
1. Load and preprocess the sign language image data
2. Split the data into training, validation, and test sets
3. Apply data augmentation to improve model generalization
4. Train a CNN model on the processed images
5. Save the best model to the `models/` directory
6. Display performance metrics and visualizations

## Applications
This model can be used as a foundation for:
- Sign language translation applications
- Educational tools for learning sign language
- Assistive technologies for the deaf and hard-of-hearing community
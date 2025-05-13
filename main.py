### 1. Import the necessary libraries ###
# Data Manipulation and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


### 2. Data Loading and Preprocessing ###
train_df = pd.read_csv("data/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
test_df = pd.read_csv("data/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")

train_df.info()
test_df.info()

train_df.head()

# Extract labels and features
y_train = train_df["label"]
X_train = train_df.drop("label", axis=1)
y_test = test_df["label"]
X_test = test_df.drop("label", axis=1)

# Examine the distribution of labels
plt.figure(figsize=(12, 4))
sns.countplot(x=y_train)
plt.title("Distribution of Sign Language Classes in Training Data")
plt.xlabel("Class Label 0-25 (excluding 9=J and 25=Z)")
plt.ylabel("Count")
plt.show()


# Convert labels to alphabet letters for better visualization
def label_to_letter(label):
    if label >= 9:  # Adjust for missing J (9)
        label += 1
    return chr(label + 65)  # ASCII: A=65, B=66, etc.


# Display the alphabet mapping
letter_mapping = {i: label_to_letter(i) for i in range(24) if i != 9}
print("\nLabel to Letter Mapping:")
for label, letter in letter_mapping.items():
    print(f"Label {label} → Letter {letter}")


# Visualize some sample images
def display_sample_images(X, y, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        random_idx = np.random.randint(0, len(X))
        img = X.iloc[random_idx].values.reshape(28, 28)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap="gray")
        label = y.iloc[random_idx]
        plt.title(f"Label: {label} ({label_to_letter(label)})")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


display_sample_images(X_train, y_train)

### 3. Data Preprocessing ###
# Reshape the data to include the channel dimension
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Get the number of unique classes
num_classes = len(np.unique(y_train)) + 1
print(f"Number of unique classes: {num_classes}")

# Convert labels to one-hot encoded vectors
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

print(f"Original label shape: {y_train.shape}")
print(f"One-hot encoded label shape: {y_train_categorical.shape}")

# Create a validation set from the training data (20%)
X_train, X_val, y_train_categorical, y_val_categorical = train_test_split(
    X_train, y_train_categorical, test_size=0.2, random_state=42
)

print(f"Final training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Create a validation set from the training data (20%)
X_train, X_val, y_train_categorical, y_val_categorical = train_test_split(
    X_train, y_train_categorical, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create a data augmentation generator for training
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
)


# Visualize the effect of data augmentation on a sample image
def show_augmented_images(image, num_augmentations=5):
    augmentation = keras.Sequential(
        [
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        ]
    )

    plt.figure(figsize=(15, 3))
    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    for i in range(num_augmentations):
        augmented_image = augmentation(tf.expand_dims(image, 0), training=True)
        plt.subplot(1, num_augmentations + 1, i + 2)
        plt.imshow(augmented_image[0].numpy().reshape(28, 28), cmap="gray")
        plt.title(f"Augmented {i+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Show augmentation examples on one sample
sample_idx = np.random.randint(0, len(X_train))
sample_image = X_train[sample_idx]
show_augmented_images(sample_image)


### 4. Build the model ###
# Define the model architecture
def build_model(input_shape=(28, 28, 1), num_classes=num_classes):
    model = Sequential(
        [
            keras.Input(shape=input_shape),
            # First convolutional block
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Second convolutional block
            Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Third convolutional block
            Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Flatten and dense layers
            Flatten(),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    return model


# Build the model
model = build_model()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

### 5. Train the model ###
# Define callbacks
callbacks = [
    EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001, verbose=1
    ),
]

# Train the model
batch_size = 128
epochs = 50
train_generator = train_datagen.flow(
    X_train,
    y_train_categorical,
    batch_size=batch_size,
)
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=(X_val, y_val_categorical),
    callbacks=callbacks,
    verbose=1,
)


# Visualize training history
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot training & validation accuracy
    axes[0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Plot training & validation loss
    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


plot_training_history(history)

### 6. Model Evaluation ###
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get predicted classes and confidence scores
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test_categorical, axis=1)

# Generate classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, num_classes=25):
    cm = confusion_matrix(y_true, y_pred)

    # Convert to percentage for better visibility
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
    )

    # Adjust label display (convert numbers to letters)
    label_mapping = {i: chr(i + 65) if i < 9 else chr(i + 66) for i in range(24)}
    letter_labels = [label_mapping.get(i, str(i)) for i in range(num_classes)]

    plt.title("Confusion Matrix (% of true class)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.xticks(np.arange(num_classes) + 0.5, letter_labels)
    plt.yticks(np.arange(num_classes) + 0.5, letter_labels)

    plt.tight_layout()
    plt.show()


plot_confusion_matrix(y_true, y_pred)

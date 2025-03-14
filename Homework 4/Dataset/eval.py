import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define parameters
IMG_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 0.005  # same as used during training

def loadImagesFromDirectory(dataset, label):
    images, labels = [], []
    for imgFile in os.listdir(dataset):
        if imgFile.endswith(('.jpg', '.png')):  # Handles both JPG and PNG files
            imgPath = os.path.join(dataset, imgFile)
            img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(IMG_SIZE, IMG_SIZE))
            imgArray = tf.keras.preprocessing.image.img_to_array(img)
            images.append(imgArray)
            labels.append(label)
    
    print(f"Loaded {len(images)} images from {dataset}")  # Debugging statement
    return np.array(images), np.array(labels)

def load_dataset(dataset_path):
    pos_images, pos_labels = loadImagesFromDirectory(os.path.join(dataset_path, 'Flower'), 1)
    neg_images, neg_labels = loadImagesFromDirectory(os.path.join(dataset_path, 'NonFlower'), 0)
    
    # Merge and normalize images
    images = np.concatenate((pos_images, neg_images), axis=0) / 255.0
    labels = np.concatenate((pos_labels, neg_labels), axis=0)

    print(f"Total images loaded: {len(images)} (Expected: 10)")  # Debugging statement
    return images, labels

# Load dataset
dataset_path = "dataset"
images, labels = load_dataset(dataset_path)

# Ensure test set contains 10 images (5 positive, 5 negative)
X_test, y_test = images, labels  # Using full dataset for evaluation

# Load the trained model
model = tf.keras.models.load_model("model.h5")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("Model loaded and compiled successfully.")

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)

# Print evaluation results
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Total Test Samples: {len(y_test)} (Expected: 10)")
print(f"Positive Samples: {np.sum(y_test)} (Expected: 5)")
print(f"Negative Samples: {len(y_test) - np.sum(y_test)} (Expected: 5)")

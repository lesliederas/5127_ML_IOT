import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Activation, Flatten, BatchNormalization,
                                     Conv2D, MaxPooling2D, Dropout)
from tensorflow.keras.regularizers import l2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define parameters
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2  # Binary classification
LEARNING_RATE = 0.005
L2_REG = 1e-4

def loadImagesFromDirectory(dataset, label):
    images, labels = [], []
    for imgFile in os.listdir(dataset):
        if imgFile.endswith(('.jpg', '.png')):  # Handle both JPG and PNG
            imgPath = os.path.join(dataset, imgFile)
            img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(IMG_SIZE, IMG_SIZE))
            imgArray = tf.keras.preprocessing.image.img_to_array(img)
            images.append(imgArray)
            labels.append(label)
    return np.array(images), np.array(labels)

def load_dataset(dataset_path):
    pos_images, pos_labels = loadImagesFromDirectory(os.path.join(dataset_path, 'Flower'), 1)
    neg_images, neg_labels = loadImagesFromDirectory(os.path.join(dataset_path, 'NonFlower'), 0)
    
    # Merge and normalize images
    images = np.concatenate((pos_images, neg_images), axis=0) / 255.0
    labels = np.concatenate((pos_labels, neg_labels), axis=0)

    return images, labels

# Load dataset
dataset_path = "dataset"
images, labels = load_dataset(dataset_path)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model using Functional API
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(L2_REG))(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(L2_REG))(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(L2_REG))(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save the trained model
model.save("model.h5")
print("Model trained and saved as model.h5")

# Extract final training and validation accuracy/loss
trainAcc = history.history['accuracy'][-1]
valAcc = history.history['val_accuracy'][-1]
trainLoss = history.history['loss'][-1]
valLoss = history.history['val_loss'][-1]

print(f"Training Accuracy: {trainAcc:.4f}")
print(f"Validation Accuracy: {valAcc:.4f}")
print(f"Training Loss: {trainLoss:.4f}")
print(f"Validation Loss: {valLoss:.4f}")
print(f"Positive Samples: {np.sum(y_train) + np.sum(y_val)}")
print(f"Negative Samples: {len(y_train) + len(y_val) - np.sum(y_train) - np.sum(y_val)}")

# Plot training accuracy & loss
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.show()

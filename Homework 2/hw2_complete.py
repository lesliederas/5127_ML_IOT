import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
def build_model1():
  model1 = models.Sequential([
    layers.Conv2D(32,(3,3), strides=(2,2),padding="same",activation="relu",input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation="softmax")
  ]) # Add code to define model 1.
  model1.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]) 
  model1.summary()
  # Add code to compile model 1.
  return model1

def build_model2():
  model2 = models.Sequential([
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), strides=(2, 2), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation="softmax")
    ])
  model2.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]) 
  model2.summary()
  return model2

def build_model3():
    inputs = Input(shape=(32,32,3))
        
    residual = layers.Conv2D(32, (3, 3), strides=(2, 2), name='Conv1', activation = 'relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(residual)
    conv1 = layers.Dropout(0.5)(conv1)

    conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), name='Conv2', activation = 'relu', padding='same')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(0.5)(conv2)
    
    conv3 = layers.Conv2D(128, (3, 3), strides=(2, 2), name='Conv3',  activation = 'relu', padding='same')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Dropout(0.5)(conv3)

    skip1 = layers.Conv2D(128, (1,1), strides=(4, 4), name="Skip1")(residual)
    skip1 = layers.Add()([skip1, conv3])

    conv4 = layers.Conv2D(128, (3, 3), name='Conv4', activation = 'relu', padding='same')(skip1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Dropout(0.5)(conv4)

    conv5 = layers.Conv2D(128, (3, 3), name='Conv5', activation = 'relu', padding='same')(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Dropout(0.5)(conv5)

    skip2 = layers.Add()([skip1, conv5])

    conv6 = layers.Conv2D(128, (3, 3), name='Conv6', activation = 'relu', padding='same')(skip2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Dropout(0.5)(conv6)

    conv7 = layers.Conv2D(128, (3, 3), name='Conv7', activation = 'relu', padding='same')(conv6)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Dropout(0.5)(conv7)

    skip3 = layers.Add()([skip2, conv7])
    
    pool = layers.MaxPooling2D((4, 4), strides=(4, 4))(skip3)
    flatten = layers.Flatten()(pool)
    
    dense = layers.Dense(128, activation = 'relu')(flatten)
    dense = layers.BatchNormalization()(dense)

    output = layers.Dense(10, activation = 'softmax')(dense)
    model3 = Model(inputs=inputs, outputs=output)

    # Compile the model
    model3.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Print model summary
    model3.summary()
    return model3

def build_model50k():
  model50k = models.Sequential([
    # First Convolutional Block
    layers.SeparableConv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),

    # Second Convolutional Block
    layers.SeparableConv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),

    # Fully Connected Layer
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')  # CIFAR-10 has 10 classes
])
  model50k.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model50k.summary()

  return model50k

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
     (train_images_all, train_labels_all), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Split training set into training and validation sets
     train_images, val_images, train_labels, val_labels = train_test_split(train_images_all, train_labels_all, test_size=0.1, random_state=42)

    # Normalize pixel values to be between 0 and 1
     train_images = train_images.astype('float32') / 255
     val_images = val_images.astype('float32') / 255
     test_images = test_images.astype('float32') / 255
    ########################################
    # Build model
     model1 = build_model1()

    # Compile the model
     model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model for 50 epochs
     history = model1.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

    # Evaluate the model on the test set
     test_loss, test_accuracy = model1.evaluate(test_images, test_labels)

    # Print test accuracy
     print("Test accuracy:", test_accuracy)
     train_accuracy = history.history['accuracy']
     val_accuracy = history.history['val_accuracy']

     print("Final Training Accuracy:", train_accuracy[-1])
     print("Final Validation Accuracy:", val_accuracy[-1])
    
     # Save the model after training
     model1.save('model1_50_epochs.keras')
     print("Model 1/4 saved")

     # Load the model
     model1 = tf.keras.models.load_model('model1_50_epochs.keras') 
     model1.summary() 
     print(f"\nModel 1 loaded with all 50 epochs")

     img_path = 'dog.png'
     img = image.load_img(img_path, target_size=(32, 32))
     img_array = image.img_to_array(img)
     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
     img_array = img_array / 255.0  

     processed_img_path = "test_image_dog.png"
     predict = model1.predict(img_array)

     predict_class = np.argmax(predict)
     class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
     predict_class_name = class_names[predict_class]

     print(f"Predicted class: {predict_class_name}")
   
  ## Build, compile, and train model 2 (DS Convolutions)
     model2 = build_model2()
     model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model for 50 epochs
     history = model2.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

    # Evaluate the model on the test set
     test_loss, test_accuracy = model2.evaluate(test_images, test_labels)

    # Print test accuracy
     print("Test accuracy of model 2:", test_accuracy)
     train_accuracy = history.history['accuracy']
     val_accuracy = history.history['val_accuracy']

     print("Final Training Accuracy of model 2:", train_accuracy[-1])
     print("Final Validation Accuracy of model 2:", val_accuracy[-1])
    
     # Save the model after training
     model2.save('model2_50_epochs.keras')
     print("Mode2 2/4 saved")

     # Load the model
     model2 = tf.keras.models.load_model('model2_50_epochs.keras') 
     model2.summary() 
     print(f"\nModel 2 loaded with all 50 epochs")

  
  ### Repeat for model 3 and your best sub-50k params model
  
     model3 = build_model3()

    # Compile the model
     model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model for 50 epochs
     history = model3.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

    # Evaluate the model on the test set
     test_loss, test_accuracy = model3.evaluate(test_images, test_labels)

    # Print test accuracy
     print("Test accuracy of model 3:", test_accuracy)
    
    # Print training, validation, and test accuracies during training
     train_accuracy = history.history['accuracy']
     val_accuracy = history.history['val_accuracy']

     print("Final Training Accuracy: of model 3", train_accuracy[-1])
     print("Final Validation Accuracy of model 3:", val_accuracy[-1])
    
    # Save the model after training
     model3.save('model3_50_epochs.keras')
     print("Model 3/4 saved")
    
    # Load the model
     model3 = tf.keras.models.load_model('model3_50_epochs.keras') 
     model3.summary() 
     print(f"\nModel 3 loaded with all 50 epochs")
    
    #50K Model
     model50k = build_model50k()

    # Compile the model
     model50k.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model for 50 epochs
     history = model50k.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))
     # Evaluate the model on the test set
     test_loss, test_accuracy = model50k.evaluate(test_images, test_labels)

    # Print test accuracy
     print("Test accuracy of 50K Model:", test_accuracy)
    
    # Print training, validation, and test accuracies during training
     train_accuracy = history.history['accuracy']
     val_accuracy = history.history['val_accuracy']

     print("Final Training Accuracy of 50k Model:", train_accuracy[-1])
     print("Final Validation Accuracy of 50K model:", val_accuracy[-1])
    
    # Save the model after training
     model50k.save('best_model.h5')
     print("Model 50k saved 4/4")
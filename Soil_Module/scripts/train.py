import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Dataset paths
PROCESSED_DATASET_PATH = "/Users/harshithyvs/Desktop/VIT/MarghaDharshi/Soil_Module/pdata"
MODEL_PATH = "/Users/harshithyvs/Desktop/VIT/MarghaDharshi/Soil_Module/models/soil_classifier_model.h5"
HISTORY_PATH = "/Users/harshithyvs/Desktop/VIT/MarghaDharshi/Soil_Module/history.npy"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 40  # Can be increased for better accuracy

# Load dataset using ImageDataGenerator
data_gen = ImageDataGenerator(rescale=1./255)
train_gen = data_gen.flow_from_directory(os.path.join(PROCESSED_DATASET_PATH, 'train'), 
                                         target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_gen = data_gen.flow_from_directory(os.path.join(PROCESSED_DATASET_PATH, 'val'), 
                                       target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# Get number of classes
num_classes = len(train_gen.class_indices)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Regularization to avoid overfitting
    Dense(num_classes, activation='softmax')  # Multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save the model
model.save(MODEL_PATH)
print(f"✅ Model training completed and saved at: {MODEL_PATH}")

# Save the training history
np.save(HISTORY_PATH, history.history)
print(f"✅ Training history saved at: {HISTORY_PATH}")

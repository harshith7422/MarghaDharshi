import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

# Define paths
DATASET_PATH = "/Users/harshithyvs/Desktop/VIT/MarghaDharshi/Soil_Module/pdata"
MODEL_PATH = "/Users/harshithyvs/Desktop/VIT/MarghaDharshi/Soil_Module/models/soil_classifier_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Create validation data generator with correct label encoding
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATASET_PATH, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"  # Ensure categorical labels
)

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(val_gen)
print(f"✅ Validation Accuracy: {val_acc:.4f}")
print(f"📉 Validation Loss: {val_loss:.4f}")

# Load history if stored separately
HISTORY_PATH = "/Users/harshithyvs/Desktop/VIT/MarghaDharshi/Soil_Module/history.npy"
if os.path.exists(HISTORY_PATH):
    history = np.load(HISTORY_PATH, allow_pickle=True).item()
else:
    print("⚠️ No history file found. Skipping accuracy/loss plots.")
    history = None

# Plot Accuracy & Loss Curves if history exists
if history:
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')

    plt.show()

# Function to Predict Soil Type from a New Image
def predict_soil(image_path, model, class_names):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    
    print(f"🧪 Predicted Soil Type: {predicted_class}")

# Get class names
class_names = list(val_gen.class_indices.keys())

# Test on a New Image
sample_image = "/Users/harshithyvs/Desktop/sample.png"  # Change this to an actual test image path
predict_soil(sample_image, model, class_names)

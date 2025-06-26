import os
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
model = load_model('lettuce_health_classifier.h5')

roboflow_dir = "dataset/roboflow_dataset/train/images"

def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return img

image_files = [os.path.join(roboflow_dir, f) for f in os.listdir(roboflow_dir)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

results = []
for img_path in image_files:
    img = preprocess_image(img_path)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    pred = model.predict(img)[0][0]
    label = "healthy" if pred < 0.5 else "bad"
    results.append([os.path.basename(img_path), label, float(pred)])

# Write results to CSV
with open('roboflow_predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'predicted_label', 'score'])
    writer.writerows(results)

print("✅ Predictions saved to roboflow_predictions.csv")

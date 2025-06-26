import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

dataset_dir = "dataset/kaggle_dataset"
IMG_SIZE = (224, 224)

def collect_image_paths(root_dir):
    image_paths = []
    labels = []
    for subdir, dirs, files in os.walk(root_dir):
        if os.path.basename(subdir) in ['healthy', 'bad']:
            label = 0 if os.path.basename(subdir) == 'healthy' else 1
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(subdir, file))
                    labels.append(label)
    return image_paths, labels

image_paths, labels = collect_image_paths(dataset_dir)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(preprocess).batch(32).shuffle(1000)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(preprocess).batch(32)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

model.save('lettuce_health_classifier.h5')
print("✅ Model trained and saved as lettuce_health_classifier.h5")

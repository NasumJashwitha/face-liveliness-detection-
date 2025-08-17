import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Path to dataset folder
DIRECTORY = "sample_liveness_data"
CATEGORIES = ["fake", "real"]  # expected folder names

data = []
labels = []

# Load images and labels
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    class_num = category  # we'll encode later
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (32, 32))  # resize to smaller for speed
        data.append(image)
        labels.append(category)

# Convert to numpy arrays
data = np.array(data) / 255.0  # normalize images
labels = np.array(labels)

# Encode string labels ("fake", "real") into integers (0,1)
le = LabelEncoder()
labels = le.fit_transform(labels)  # fake=0, real=1
print("Classes found and encoded:", le.classes_)

# Save label encoder for later use in demo.py
with open("le.pickle", "wb") as f:
    f.write(pickle.dumps(le))

# One-hot encode the labels
labels = to_categorical(labels, 2)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.2)

# Save the trained model in .h5 format (compatible with keras load_model)
model.save("liveness.model.h5")

print("Training completed and model saved as liveness.model.h5")

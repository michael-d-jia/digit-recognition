# mnist_viewer_with_preds.py

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np

# -----------------------------
# Load MNIST data
# -----------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
images = np.concatenate([x_train, x_test])
labels = np.concatenate([y_train, y_test])
num_images = len(images)

# -----------------------------
# Load your trained CNN model
# -----------------------------
# Make sure your model is saved (e.g., model.save("mnist_cnn.h5"))
model = load_model("saved_model/mnist_cnn_model.keras")

# Preprocess images for prediction
images_input = np.expand_dims(images / 255.0, -1)  # normalize and add channel dimension
pred_probs = model.predict(images_input)
pred_labels = np.argmax(pred_probs, axis=1)

# -----------------------------
# Viewer setup
# -----------------------------
index = 0
fig, ax = plt.subplots()
im = ax.imshow(images[index], cmap='gray')

# Set initial title with color
color = 'green' if labels[index] == pred_labels[index] else 'red'
title = ax.set_title(f"T:{labels[index]} P:{pred_labels[index]}", color=color)
ax.axis('off')

# Function to update image
def update_image(idx):
    im.set_data(images[idx])
    color = 'green' if labels[idx] == pred_labels[idx] else 'red'
    title.set_text(f"Index: {idx} | True: {labels[idx]} Pred: {pred_labels[idx]}")
    title.set_color(color)
    fig.canvas.draw_idle()

# Key press event handler
def on_key(event):
    global index
    if event.key == 'right':
        index = (index + 1) % num_images
        update_image(index)
    elif event.key == 'left':
        index = (index - 1) % num_images
        update_image(index)
    elif event.key == 'escape':
        plt.close(fig)

# Connect key event
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

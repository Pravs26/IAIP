import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Load model
model = tf.keras.models.load_model(
    "saved_model/flower_classifier.keras"
)

# Flower classes
class_names = [
    'daisy',
    'dandelion',
    'roses',
    'sunflowers',
    'tulips'
]

# Input image name
img_path = input("Enter image file name: ")

# Image settings
img_height = 224
img_width = 224

# Load image
img = load_img(
    img_path,
    target_size=(img_height, img_width)
)

# Convert image to array
img_array = img_to_array(img)

# Expand dimensions
img_array = tf.expand_dims(img_array, 0)

# Predict
predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])

# Output
print(
    "\nThis image most likely belongs to {} with {:.2f}% confidence."
    .format(
        class_names[np.argmax(score)],
        100 * np.max(score)
    )
)
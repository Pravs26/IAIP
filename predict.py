import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

model = tf.keras.models.load_model(
    "saved_model/flower_classifier.keras"
)

class_names = [
    'daisy',
    'dandelion',
    'roses',
    'sunflowers',
    'tulips'
]

img_path = input("Enter image file name: ")

img_height = 224
img_width = 224

img = load_img(
    img_path,
    target_size=(img_height, img_width)
)

img_array = img_to_array(img)

img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])

print(
    "\nThis image most likely belongs to {} with {:.2f}% confidence."
    .format(
        class_names[np.argmax(score)],
        100 * np.max(score)
    )
)

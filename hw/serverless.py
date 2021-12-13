import os
from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import tensorflow_runtime as tflite
from keras_image_helper import create_preprocessor


model = keras.models.load_model('data/dogs_cats_10_0.687.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
model_path = 'data/dogs_cats.tflite'
with open(model_path, 'wb') as f:
  f.write(tflite_model)

# Question 1
num = 1
ans = os.path.getsize(model_path)
print(f"Question {num}: {ans}")


interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()  # Set weights

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Question 2
num = 2
ans = output_index
print(f"Question {num}: {ans}")


def download_image(url):
  with request.urlopen(url) as resp:
    buffer = resp.read()
  stream = BytesIO(buffer)
  img = Image.open(stream)
  return img


def prepare_image(img, target_size):
  if img.mode != 'RGB':
    img = img.convert('RGB')
  img = img.resize(target_size, Image.NEAREST)
  return img


url = 'https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg'
img = download_image(url)
img = prepare_image(img, (150, 150))

preprocessor = create_preprocessor('xception', target_size=(150, 150))

# preprocessor.image_to_array(img)?
# Question 3
num = 3
ans = np.array(img)[0][0][0]
print(f"Question {num}: {ans}")




def predict(url):
  X = preprocessor.from_url(url)

  interpreter.set_tensor(input_index, X)
  interpreter.invoke()
  preds = interpretor.get_tensor(output_index)

  classes = [
      'dress',
      'hat',
      'longsleeve',
      'outwear',
      'pants',
      'shirt',
      'shoes',
      'shorts',
      'skirt',
      't-shirt'
  ]
  float_predictions = preds[0].tolist()

  return dict(zip(classes, float_predictions))



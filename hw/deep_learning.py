import pandas as pd
from tensorflow import keras
from tensorflow.keras.applications.xception import (
  Xception, preprocess_input, decode_predictions
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def make_model(
  input_size: int = 150,
  learning_rate: float = 0.002,
  dense_neurons: int = 64,
  droprate: float = 0.2,
  optimizer: str = 'SGD',
  loss: str = 'binary',
):

  def _design_model(
    imagenet: bool = False,
    maxpool: bool = True,
    drop: bool = False,
  ):
    inputs = keras.Input(shape=(input_size, input_size, 3))

    if imagenet:
      # Get Convolutional Layers from ImageNet
      base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3),
      )
      base_model.trainable = False
      base = base_model(inputs, training=False)
    else:
      # Make our own Convolutional Layers
      base = keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
      )(inputs)

    if maxpool:
      pool = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
      )(base)
    else:
      pool = keras.layers.GlobalAveragePooling2D()(base)

    vectors = keras.layers.Flatten()(pool)

    inner = keras.layers.Dense(
      units=dense_neurons,
      activation='relu',
    )(vectors)

    if drop:
      inner = keras.layers.Dropout(droprate)

    outputs = keras.layers.Dense(
      units=1,
      activation='sigmoid',
    )(inner)

    model = keras.Model(inputs, outputs)
    return model

  model = _design_model()

  if optimizer == 'SGD':
    optimizer = keras.optimizers.SGD(
      learning_rate=learning_rate,
      momentum=0.8,
    )
  else:
    optimizer = keras.optimizers.Adam(
      learning_rate=learning_rate,
    )

  # Set `from_logits` = True if we don't specify output activation
  if loss == 'binary':
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
  else:
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

  # Compile model
  model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy'],
  )

  return model


model = make_model()

# Question 1
num = 1
ans = 'BinaryCrossentropy'
print(f"Question {num}: {ans}")


# Question 2
num = 2
print(model.summary())
ans = '11,215,873'
print(f"Question {num}: {ans}")


def train_model(model, augment=False):
  if augment:
    img_gen = ImageDataGenerator(
      preprocessing_function=None,  # preprocess_input from Xception
      rescale=1 / 255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
    )
  else:
    img_gen = ImageDataGenerator(
      preprocessing_function=None,  # preprocess_input from Xception
      rescale=1 / 255,
    )

  train_ds = img_gen.flow_from_directory(
    directory='./data/dogs-cats/train/',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',          # Could be categorical, etc
    shuffle=True,
  )
  val_ds = img_gen.flow_from_directory(
      directory='./data/dogs-cats/validation/',
      target_size=(150, 150),
      batch_size=20,
      class_mode='binary',          # Could be categorical, etc
      shuffle=True,
  )

  history = model.fit(
    train_ds,
    steps_per_epoch=100,
    epochs=10,
    validation_data=val_ds,
    validation_steps=50,
  )
  return history


# Fit model without augmentation
history = train_model(model)
stats = pd.DataFrame(history.history)

# Question 3
num = 3
ans = stats['accuracy'].median().round(2)
print(f"Question {num}: {ans}")


# Question 4
num = 4
ans = stats['loss'].std().round(2)
print(f"Question {num}: {ans}")


# Refit with augmentation
history = train_model(model, augment=True)
stats = pd.DataFrame(history.history)

# Question 5
num = 5
ans = stats['val_loss'].mean().round(2)
print(f"Question {num}: {ans}")


# Question 6
num = 6
ans = stats['val_accuracy'].tail(5).mean().round(2)
print(f"Question {num}: {ans}")

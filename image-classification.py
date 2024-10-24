import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plot

print('Creating model')

image_size = (180, 180)

# Build a model

def make_model(input_shape, num_classes):
  print("Creating model")

  inputs = keras.Input(shape=input_shape)

  x = layers.Rescaling(1.0 / 255)(inputs)
  x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x

  for size in [256, 512, 728]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(size, 1, strides=2, padding="same")(
      previous_block_activation
    )

    x = layers.add([x, residual])
    previous_block_activation = x

  x = layers.SeparableConv2D(1024, 3, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  x = layers.GlobalAveragePooling2D()(x)

  if (num_classes == 2):
    units = 1
  else:
    units = num_classes

  x = layers.Dropout(0.25)(x)

  outputs = layers.Dense(units, activation=None)(x)

  print("Returning model")

  return keras.Model(inputs, outputs)

model = make_model(
  input_shape=image_size + (3,),
  num_classes=2
)

print("Generate dataset")

batch_size = 128

train_ds, val_ds = keras.utils.image_dataset_from_directory(
   "PetImages",
   validation_split=0.2,
   subset="both",
   seed=1337,
   image_size=image_size,
   batch_size=batch_size
)

print("Plotting dataset")

for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plot.subplot(3, 3, i + 1)
    plot.imshow(np.array(images[i]).astype("uint8"))
    plot.title(int(labels[i]))
    plot.axis("off")

plot.show()

# Data augmentation

print("Setup augmentation")

data_augmentation_layers = [
   layers.RandomFlip("horizontal"),
   layers.RandomRotation(0.1)
]

def data_augmentation(images):
  for layer in data_augmentation_layers:
    images = layer(images)
  return images

# Apply data augmentation

print("Apply augmentation")

train_ds = train_ds.map(
  lambda img, label: (data_augmentation(img), label),
  num_parallel_calls = tf_data.AUTOTUNE
)

print("Prefetch")

train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

print("Plotting model")

keras.utils.plot_model(
  model=model,
  show_shapes=True,
  to_file="model.png"
)

# Train the model

print("Compiling model")

epochs = 25
callbacks = [
  keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")
]

model.compile(
  optimizer=keras.optimizers.Adam(3e-4),
  loss=keras.losses.BinaryCrossentropy(from_logits=True),
  metrics=[keras.metrics.BinaryAccuracy(name="acc")]
)

print("Fitting model")

model.fit(
  train_ds,
  epochs=epochs,
  callbacks=callbacks,
  validation_data=val_ds
)

print("Loading new data")

img = keras.utils.load_img(
  "PetImages/Cat/6779.jpg",
  target_size=image_size
)

plot.imshow(img)
plot.show()

print("Inference on new data")

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))

print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog")

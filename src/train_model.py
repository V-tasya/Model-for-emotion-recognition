import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models
from settings.config import IMAGES_DIR, IMAGE_SIZE, BATCH_SIZE, CHANNELS, EPOCHS

def get_datasets() -> tuple:
  '''
  This function loads images from directory, splits them into training and validation subsets,
  normalizes pixel values, and configures performance tuning.

  Args:
    None
  Returns:
    tuple: A tuple containing (train_dataset, validation_dataset, class_names), where both train_dataset and validation_dataset are tf.data.Dataset objects ready for training.
  '''
  train_dataset = tf.keras.utils.image_dataset_from_directory(IMAGES_DIR, validation_split=0.2,
    subset="training", seed=123, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE)
  validation_dataset = tf.keras.utils.image_dataset_from_directory(IMAGES_DIR, validation_split=0.2,
    subset="validation", seed=123, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE)
  class_names = train_dataset.class_names
  
  normalization_layer = layers.Rescaling(1./255)
  train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
  validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

  autotune = tf.data.AUTOTUNE
  train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
  validation_dataset = validation_dataset.cache().prefetch(buffer_size=autotune)

  return train_dataset, validation_dataset, class_names

def build_model(number_of_classes: int) -> tf.keras.Model:
  '''
  This function takes a pre-trained MobileNetV2 model, freezes its weights to retain feature 
  extraction capabilities, and adds a new classification head on top.

  Args:
    number_of_classes (int): The number of emotion categories to recognize.
  Returns:
    my_model (tf.keras.Model): A compiled Keras model ready for training.
  '''
  data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.1),])
  base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), include_top=False, weights='imagenet')
  base_model.trainable = False

  my_model = models.Sequential([layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)), data_augmentation, base_model,
    layers.GlobalAveragePooling2D(), layers.Dense(256, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.4),  
    layers.Dense(number_of_classes, activation="softmax")])
  my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"])

  return my_model

def main() -> None:
  tr_ds, val_ds, class_names = get_datasets()
  model = build_model(len(class_names))
  #print(f"Classes found: {class_names}")
  #print("Datasets are ready for training.")
  #model.summary()
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
  history = model.fit(tr_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])
  model.save('emotions_recognition_model.keras')

if __name__ == '__main__':
  main()
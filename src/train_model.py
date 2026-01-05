import os
import tensorflow as tf
from tensorflow.keras import layers
from settings.config import IMAGES_DIR, IMAGE_SIZE, BATCH_SIZE

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
  validation_dataset =validation_dataset.map(lambda x, y: (normalization_layer(x), y))

  autotune = tf.data.AUTOTUNE
  train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
  validation_dataset = validation_dataset.cache().prefetch(buffer_size=autotune)

  return train_dataset, validation_dataset, class_names

def main() -> None:
  tr_ds, val_ds, class_names = get_datasets()
  print(f"Classes found: {class_names}")
  print("Datasets are ready for training.")

if __name__ == '__main__':
  main()
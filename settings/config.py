import os

DATA_DIR = 'data'
DATASET_PATH = os.path.join(DATA_DIR, 'fer2013.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

IMAGE_SIZE = 48
CHANNELS = 3

EMOTIONS = {
  0: 'angry',
  1: 'disgust',
  2: 'fear',
  3: 'happy',
  4: 'sad',
  5: 'surprise',
  6: 'neutral'
}
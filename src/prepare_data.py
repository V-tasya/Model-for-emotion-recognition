import numpy as np
import pandas as pd
import cv2
import os
import sys
from settings.config import DATASET_PATH, IMAGES_DIR, IMAGE_SIZE, EMOTIONS

def read_csv() -> pd.DataFrame:
  try:
    return pd.read_csv(DATASET_PATH)
  except Exception as exception:
    print('Error occured while reading file: ', exception)
    sys.exit(1)

def prepare_data(data_frame: pd.DataFrame) -> None:
  '''
  This function creates directories for each emotion, processes raw pixel data 
  by reshaping, resizing, and converting to RGB, and saves them as images.

  Args:
    data_frame (pd.DataFrame): The FER2013 dataset containing 'emotion' and 'pixels' columns.
  Returns:
    None: Saves images directly to the output directory.
  '''
  try: 
    for emotion in EMOTIONS.values():
      os.makedirs(os.path.join(IMAGES_DIR, emotion), exist_ok=True)
    
    for index, row in data_frame.iterrows():
      pixels = np.array(row['pixels'].split(), dtype='uint8')
      image = pixels.reshape(48, 48)

      if IMAGE_SIZE != 48:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

      image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

      current_emotion = EMOTIONS[row['emotion']]
      file_path = os.path.join(IMAGES_DIR, current_emotion, f'{index}.jpg')
      cv2.imwrite(file_path, image)

  except Exception as exception:
    print('Error occured while data preparation: ', exception)

def main() -> None:
  if os.path.exists(IMAGES_DIR) and len(os.listdir(IMAGES_DIR)) > 0:
    print('Dataset already prepared')
    return

  df: pd.DataFrame = read_csv()
  prepare_data(df)
  print('Data was prepared successfully :)')

if __name__ == '__main__':
  main()
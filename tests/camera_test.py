import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('emotions_recognition_model.keras')
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("Enter 'q' to exit the program")

while True:
  ret, frame = cap.read()
  if not ret:
    break

  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

  for (x, y, w, h) in faces:
    roi_gray = gray_frame[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_gray = roi_gray / 255.0
    img_pixels = np.stack((roi_gray,)*3, axis=-1)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    predictions = model.predict(img_pixels, verbose=0)
    max_index = np.argmax(predictions[0])
    emotion = class_names[max_index]
    confidence = float(predictions[0][max_index])
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    text = f"{emotion} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

  cv2.imshow('Emotion Detector Test', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
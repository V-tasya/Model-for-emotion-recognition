import tensorflow as tf

model = tf.keras.models.load_model('emotions_recognition_model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Light model('model.tflite') is ready")
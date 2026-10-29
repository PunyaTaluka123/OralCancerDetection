import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('models/best_oral_cancer_model.keras')

# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Quantization for optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('models/optimized_model.tflite', 'wb') as f:
    f.write(tflite_model)

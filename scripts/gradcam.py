from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import Model


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    return image


def compile_model(model):
    """
    Compile the model (necessary for Grad-CAM, since we're not training).
    Here, we use a dummy loss and optimizer as we only need to perform inference.
    """
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


def grad_cam(model, image, target_layer_name):
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    target_layer = model.get_layer("sequential_13").get_layer(
        "model4").get_layer(target_layer_name)

    grad_model = Model(inputs=model.inputs[0], outputs=[
                       target_layer.output, model.outputs])
    print(model.inputs[0])
    print("Grad Model Created")

    with tf.GradientTape() as tape:
        tape.watch(image)
        print("Gradient Tape Watch Started")
        conv_output, predictions = grad_model(image)
        print("Predictions Finished")

        # Get the predicted class
        predicted_class = tf.argmax(predictions[0])
        print("Predicted Class Fetched")

    gradients = tape.gradient(predicted_class, conv_output)

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_output = conv_output[0]
    for i in range(conv_output.shape[-1]):
        conv_output[..., i] *= pooled_gradients[i]

    heatmap = tf.reduce_sum(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)

    return heatmap


def overlay_heatmap_on_image(image, heatmap):
    heatmap_resized = cv2.resize(
        heatmap.numpy(), (image.shape[2], image.shape[1]))
    image_rgb = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = heatmap_colored * 0.4 + image_rgb * 0.6
    return superimposed_img


def display_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def main(image_path, model_path, target_layer_name):
    model = load_model(model_path)
    image = load_and_preprocess_image(image_path)
    model.predict(image)
    print("Prediction loaded")
    compile_model(model)
    print("Model loaded")
    heatmap = grad_cam(model, image, target_layer_name)
    superimposed_img = overlay_heatmap_on_image(image, heatmap)
    display_image(superimposed_img)


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]  # e.g. data/non-cancer/tongue.jpg
    model_path = 'models/baseline/keras_model.h5'
    target_layer_name = 'Conv_1'
    main(image_path, model_path, target_layer_name)

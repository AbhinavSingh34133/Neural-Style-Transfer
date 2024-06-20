# Adam_model.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing
from PIL import Image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import os

load_img = preprocessing.image.load_img
img_to_array = preprocessing.image.img_to_array

def load_to_process(img_path):
    img = load_img(img_path, target_size=(400, 400))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_image(image, directory, filename):
    image = deprocess_img(image)
    if len(image.shape) == 4:
        image = np.squeeze(image, 0)
    img = Image.fromarray(image.astype('uint8'))
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, filename)
    img.save(save_path)

def perform_nst(content_image_path, style_image_path, output_image_path):
    content_image = load_to_process(content_image_path)
    style_image = load_to_process(style_image_path)

    noise_image = np.random.uniform(-20, 20, (1, 400, 400, 3)).astype('float32')

    model = VGG19(weights='imagenet', include_top=False)
    model.trainable = False

    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']

    style_models = []
    content_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(content_layer).output)

    style_models = [tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(x).output) for x in style_layers]

    def content_loss(original, generated):
        P = content_model(original)
        F = content_model(generated)
        return tf.reduce_mean(tf.square(P - F))

    def gram_matrix(X):
        channels = int(X.shape[-1])
        X = tf.reshape(X, [-1, channels])
        n = tf.shape(X)[0]
        gram = tf.matmul(X, X, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    no_of_style_models = len(style_models)

    def style_cost(original, generated):
        J = 0
        for style_model in style_models:
            P = style_model(original)
            F = style_model(generated)
            G_P = gram_matrix(P)
            G_F = gram_matrix(F)
            cost = tf.reduce_mean(tf.square(G_P - G_F))
            J += cost

        return J / no_of_style_models

    iterations = 50
    lr = 7.0
    alpha = 10
    beta = 1000
    generated = tf.Variable(content_image, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for iteration in range(iterations):
        with tf.GradientTape() as tape:
            J_content = content_loss(content_image, generated)
            J_style = style_cost(style_image, generated)
            J_total = alpha * J_content + beta * J_style

        grads = tape.gradient(J_total, generated)
        optimizer.apply_gradients([(grads, generated)])

        print("Iteration: ", iteration)
        print('Total Loss: ', J_total.numpy())

        save_image(generated.numpy(),os.path.dirname('./Generated_Image/Generated_0.jpg'),f'Generated_Image{iteration}.png')
    save_image(generated.numpy(), os.path.dirname(output_image_path), os.path.basename(output_image_path))
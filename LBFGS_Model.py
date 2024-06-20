import math
import numpy as np
import tensorflow as tf
import scipy.optimize
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing
from PIL import Image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import os

# Load and preprocess images
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


content_image = load_to_process('dog.jpg')
style_image = load_to_process('vangogh.jpg')


def save_image(image, directory, filename):
    image = deprocess_img(image)
    if len(image.shape) == 4:
        image = np.squeeze(image, 0)
    img = Image.fromarray(image.astype('uint8'))
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, filename)
    img.save(save_path)


def show_image(image, title=None):
    image = deprocess_img(image)
    if len(image.shape) == 4:
        image = np.squeeze(image, 0)
    plt.imshow(image.astype('uint8'))
    if title:
        plt.title(title)
    plt.show()


noise_image = np.random.uniform(-20, 20, (1, 400, 400, 3)).astype('float32')

model = VGG19(weights='imagenet', include_top=False)
model.trainable = False

content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']

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


alpha = 10
beta = 1000
generated_image = tf.Variable(content_image, dtype=tf.float32)


def compute_loss_and_grads(image):
    with tf.GradientTape() as tape:
        tape.watch(image)
        J_content = content_loss(content_image, image)
        J_style = style_cost(style_image, image)
        J_total = alpha * J_content + beta * J_style
    grads = tape.gradient(J_total, image)
    return J_total, grads


# Flatten the image for the optimizer
def get_flattened_image(image):
    return image.numpy().flatten().astype(np.float64)


def set_flattened_image(flattened_image, shape):
    return tf.convert_to_tensor(flattened_image.reshape(shape), dtype=tf.float32)


def lbfgs_minimize():
    initial_position = get_flattened_image(generated_image)
    shape = generated_image.shape

    def loss_and_grads(flattened_image):
        image = set_flattened_image(flattened_image, shape)
        loss, grads = compute_loss_and_grads(image)
        return loss.numpy().astype(np.float64), grads.numpy().flatten().astype(np.float64)

    result = scipy.optimize.fmin_l_bfgs_b(loss_and_grads, initial_position, maxiter=1)
    optimized_image = set_flattened_image(result[0], shape)
    return optimized_image


# Perform the optimization
optimized_image = lbfgs_minimize()

# Show and save the final image
show_image(optimized_image.numpy())
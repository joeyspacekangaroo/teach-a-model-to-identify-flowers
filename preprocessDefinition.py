import tensorflow as tf
from tensorflow import keras
import numpy as np

def preprocess(image,label):
    resized_image = tf.image.resize(image, [244,244])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image,label
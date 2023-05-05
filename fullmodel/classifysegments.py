import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models
from ../classification/foodclassification.py import create_model

def classifyList(images):
    classlist = loadclasses()
    model = create_model()
    labellist = []
    
    for i in images:
        image = preprocess_img(i)
        predictions = model.predict(image)
        labelarg = np.argmax(predictions)
        labellist.append(classlist[labelarg])





def preprocess_img(image):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
    image = tf.keras.applications.vgg16.preprocess_input(image) #vgg16 required
    return tf.cast(image, tf.float32) # return (float32_image, label) tuple
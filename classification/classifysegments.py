import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models
from foodclassification import create_model, load_classes

img_shape = 224

def classifyList(images):
    classlist = load_classes()[0]

    # model = create_model()
    # print('model.summary():', model.summary())
    # model.load_weights('model_checkpoints/cp.ckpt')

    model = tf.keras.models.load_model('saved_model/my_model')
    print('model.summary():', model.summary())
    labellist = []

    print('classlist:', classlist)
    
    for i in images:
        image = preprocess_img(i)
        image = tf.expand_dims(image,0)
        print('image.shape:', image.shape)
        predictions = model.predict(image)
        labelarg = np.argmax(predictions)
        labellist.append(classlist[labelarg])

    # preprocessed_images = [preprocess_img(img) for img in images]
    # predictions=model.predict(preprocessed_images)
    # print(predictions)
    print('exiting classifyList')
    return labellist

def preprocess_img(image):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
    image = tf.keras.applications.vgg16.preprocess_input(image) #vgg16 required

    return tf.cast(image, tf.float32) # return (float32_image, label) tuple

npz = np.load('cropped_output.npz')
segments = npz['arr_0']

print('segments.shape:', segments.shape)

labeled_list = classifyList(segments)
print(labeled_list)
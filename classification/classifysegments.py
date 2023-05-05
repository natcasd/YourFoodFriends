import numpy as np
import tensorflow as tf
from foodclassification import create_model, load_classes

# from ..segmentation.sam_auto_tutorial import run_model

img_shape = 224

def classifyList(images):
    classlist = load_classes()[0]

    '''
    # Load Checkpoints Strategy
    model = create_model()
    model.load_weights('model_checkpoints/cp.ckpt')
    '''

    # Load Model Strategy
    model = tf.keras.models.load_model('saved_model/my_model')
    labellist = []

    # print('classlist:', classlist)
    
    for i in images:
        image = preprocess_img(i)
        image = tf.expand_dims(image,0)
        predictions = model.predict(image, verbose=0)
        labelarg = np.argmax(predictions)
        labellist.append(classlist[labelarg])

    print('exiting classifyList')
    return labellist

def preprocess_img(image):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    # image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape # Already resized
    image = tf.keras.applications.vgg16.preprocess_input(image) #vgg16 required

    return tf.cast(image, tf.float32) # return (float32_image, label) tuple

npz = np.load('cropped_output.npz')
segments = npz['arr_0']

print('segments.shape:', segments.shape)

labeled_list = classifyList(segments)
print('labeled_list:', labeled_list)
import numpy as np
import tensorflow as tf
from classification import food_InceptionResNetV2
import pickle
from segmentation import sam_auto_tutorial

img_shape = 224

def classify_list(images):
    with open('class_names.pkl', 'rb') as f:
        class_list = pickle.load(f)[0]

    
    # Load Checkpoints Strategy
    model = food_InceptionResNetV2.create_model()
    model.load_weights('classification/checkpoints/050523-180311/weights.hdf5')
    

    '''
    # Load Model Strategy
    model = tf.keras.models.load_model('saved_model/my_model')
    '''
    label_list = []
    # print('classlist:', classlist)
    
    for i in images:
        image = preprocess_img(i)
        image = tf.expand_dims(image,0)
        predictions = model.predict(image, verbose=0)
        labelarg = np.argmax(predictions)
        label_list.append(class_list[labelarg])

    print('exiting classifyList')
    return label_list

def preprocess_img(image):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [299, 299]) # reshape to img_shape # Already resized
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image) #vgg16 required

    return tf.cast(image, tf.float32) # return (float32_image, label) tuple

# npz = np.load('cropped_output.npz')
# segments = npz['arr_0']

segments = sam_auto_tutorial.run_model(device='cuda')

print('segments.shape:', segments.shape)

labeled_list = classify_list(segments)
print('labeled_list:', labeled_list)
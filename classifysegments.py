import numpy as np
from segmentation import sam_auto_tutorial
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import cv2
img_shape = 224

counter = 0
def classify_list(images):
    global counter
    with open('classification/class_names.pkl', 'rb') as f:
        class_list = pickle.load(f)[0]
    
    # Load Model Strategy
    model = tf.keras.models.load_model('classification/finetune_checkpoints/050823-171401/model.hdf5')
    label_list = []
    max_list = []

    # Predict each image
    for i in images:
        plt.imshow(i)
        plt.savefig(f'fnf/experiments/mask{counter}.jpg') # Save masks
        image = preprocess_img(i)
        image = tf.expand_dims(image,0)
        predictions = model.predict(image, verbose=0)
        labelarg = np.argmax(predictions)
        max = np.max(predictions)
        max_list.append(max)
        label_list.append(class_list[labelarg])
        counter += 1

    return label_list, max_list

def preprocess_img(image):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [299, 299]) # reshape to img_shape # Already resized
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image) #vgg16 required
    return tf.cast(image, tf.float32) # return (float32_image, label) tuple

print('before sam')
segments1, segments2, coordinatelist, imageloc = sam_auto_tutorial.run_model(device='cuda', filter_method='boundingboxes', data = 'segmentation/images/ratty1.jpg')
print('after sam')
print('coordinateList.shape:', len(coordinatelist))

labeled_list1, max_list1 = classify_list(segments1)
labeled_list2, max_list2 = classify_list(segments2)
#labeled_list = ['yomama' for x in range(70)]

import matplotlib.pyplot as plt
import matplotlib.patches as patches

wholeimage = plt.imread(imageloc)
fig, ax = plt.subplots(1)
ax.imshow(wholeimage)

for i,j in enumerate(coordinatelist):
    if max(max_list1[i], max_list2[i]) > 0.45:
    # if max_list1[i] > 0.6:
        rect = patches.Rectangle((j[1], j[0]), j[3], j[2], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        labelx = j[1]
        labely = j[0]-5 # Slightly above the top-left corner
        if(max_list1[i]>=max_list2[i]):
            ax.text(labelx, labely, labeled_list1[i], fontsize=12, color='k', weight='bold')
        else:
            ax.text(labelx, labely, labeled_list2[i], fontsize=12, color='k', weight='bold')

# plt.show()
plt.savefig('output.png')
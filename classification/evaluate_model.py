import tensorflow as tf
import tensorflow_datasets as tfds

model = tf.keras.models.load_model('finetune_checkpoints/050823-171401/model.hdf5')
print(model.summary())


(train_data, test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                        split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                        shuffle_files=True, # shuffle files on download?
                                        as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                        with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)

def preprocess_img(image, label, img_shape=299):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
    # image = tf.keras.applications.xception.preprocess_input(image)
    return tf.cast(image, tf.float32), label # return (float32_image, label) tuple

# Map preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(32).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map prepreprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

print(model.evaluate(test_data,verbose=2))
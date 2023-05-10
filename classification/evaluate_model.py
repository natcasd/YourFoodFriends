import tensorflow as tf
import tensorflow_datasets as tfds

model = tf.keras.models.load_model('finetune_checkpoints/050823-171401/model.hdf5')
print(model.summary())


(train_data, test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                        split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                        shuffle_files=True, # shuffle files on download?
                                        as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                        with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)

def preprocess_image(image, label, img_shape=299):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape])
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
    return tf.cast(image, tf.float32), label

test_data = test_data.map(preprocess_image, 
                          num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE) 

print(model.evaluate(test_data,verbose=2))
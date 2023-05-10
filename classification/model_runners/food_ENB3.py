import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers, models
import argparse
from keras import mixed_precision
import os
from datetime import datetime

num_classes = 101

# Accelerate 
mixed_precision.set_global_policy(policy="mixed_float16") # set global policy to mixed precision 
mixed_precision.global_policy()

def preprocess_image(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape])
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return tf.cast(image, tf.float32), label

def process_callbacks(loaded_model):
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    # CHECKPOINT CALLBACK
    checkpoint_path = f"finetune_checkpoints/{timestamp}/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path += 'model.hdf5'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor="val_accuracy", # Trigger on value accuracy
                                                        save_best_only=True, # Only save best monitor (val_accuracy)
                                                        save_weights_only=True, # Save only weights; USE WITH EFFICIENTNET
                                                        verbose=0) # Don't print save messages
    # TENSORBOARD LOGGING CALLBACK
    log_path = f"logs/{timestamp}/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)

    # MODEL SUMMARY LOGGING
    summary_path = f"finetune_checkpoints/{timestamp}/summary.txt"
    with open(summary_path, 'w') as f:
        loaded_model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # CSV LOGGING CALLBACK
    csv_logger_path = f"finetune_checkpoints/{timestamp}/epochs.csv"
    csv_logger_callback = tf.keras.callbacks.CSVLogger(csv_logger_path)

    return [checkpoint_callback, log_callback, csv_logger_callback]

def create_model():
    input_shape = (224, 224, 3)
    base_model = tf.keras.applications.EfficientNetB3(include_top=False)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = base_model(inputs, training=False)
    
    
    # Default
    x = layers.GlobalAveragePooling2D(name='pooling_layer')(x)
    x = layers.Dense(101, name='logits')(x)
    
    outputs = layers.Activation('softmax', dtype=tf.float32, name='softmax_float32')(x)
    model = tf.keras.Model(inputs, outputs)


    # Compile the model
    model.compile(loss="sparse_categorical_crossentropy", # Use sparse_categorical_crossentropy when labels are *not* one-hot
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
    
    return model

# def run_model(load):
def run_model(load_checkpoint):
    # Load and process data
    (train_data, test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                            split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                            shuffle_files=True, # shuffle files on download?
                                            as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                            with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)

    # Preprocess training and test data
    train_data = train_data.map(map_func=preprocess_image, 
                                num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.map(preprocess_image, 
                              num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch data
    train_data = train_data.shuffle(32).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE) #Shuffle!
    test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE) 

    # Create and compile the model
    model = create_model()

    # Load callbacks
    processed_callbacks = process_callbacks(model)

    model.fit(
        train_data,
        epochs=6,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=int(0.15 * len(test_data)),
        callbacks = processed_callbacks,
        verbose=2
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--load_checkpoint',
                        default='n',help='Load existing test/train data')
    
    args = parser.parse_args()
    print('args:', args)

    # run_model(load=args.load)
    run_model(load_checkpoint=args.load_checkpoint)
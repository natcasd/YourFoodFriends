import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers, mixed_precision
import argparse
import os
from datetime import datetime



num_classes = 101

# Accelerate on Nvidia
mixed_precision.set_global_policy(policy="mixed_float16") # set global policy to mixed precision 
mixed_precision.global_policy()

def preprocess_image(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return tf.cast(image, tf.float32), label

def process_callbacks(loaded_model):
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    # CHECKPOINT CALLBACK
    checkpoint_path = f"../checkpoints/{timestamp}/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path += 'model.hdf5'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor="val_accuracy", # Trigger on value accuracy
                                                        save_best_only=True, # Only save best monitor (val_accuracy)
                                                        save_weights_only=False, # Save entire model;
                                                        verbose=0) # Don't print save messages
    # TENSORBOARD LOGGING CALLBACK
    log_path = f"logs/{timestamp}/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)

    # MODEL SUMMARY LOGGING
    summary_path = f"../checkpoints/{timestamp}/summary.txt"
    with open(summary_path, 'w') as f:
        loaded_model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # CSV LOGGING CALLBACK
    csv_logger_path = f"../checkpoints/{timestamp}/epochs.csv"
    csv_logger_callback = tf.keras.callbacks.CSVLogger(csv_logger_path)

    return [checkpoint_callback, log_callback, csv_logger_callback]

def create_model():
    input_shape = (299, 299, 3)
    base_model = tf.keras.applications.Xception(include_top=False)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D(name='pooling_layer')(x)
    x = layers.Dense(2048, name='Dense2048', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
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
    (train_data, test_data), ds_info = tfds.load(name="food101", 
                                            split=["train", "validation"], 
                                            shuffle_files=True,
                                            as_supervised=True,
                                            with_info=True)
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

    # Load weights if given in argument
    if(load_checkpoint != 'n'):
        model.load_weights(load_checkpoint)

    # Load callbacks
    processed_callbacks = process_callbacks(model)
    
    # Fit the model
    model.fit(
        train_data,
        epochs = 5,
        validation_data=test_data,
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
import tensorflow as tf
from keras import layers, utils, mixed_precision
import os
from datetime import datetime

mixed_precision.set_global_policy(policy="mixed_float16") # set global policy to mixed precision 
mixed_precision.global_policy()

food5k_path = '../data/food5k/'
train_data = utils.image_dataset_from_directory(
    directory=food5k_path+'training/',
    labels='inferred',
    label_mode='categorical'
)
test_data = utils.image_dataset_from_directory(
    directory=food5k_path+'validation/',
    labels='inferred',
    label_mode='categorical'
)
evaluation_data = utils.image_dataset_from_directory(
    directory=food5k_path+'evaluation/',
    labels='inferred',
    label_mode='categorical'
)


input_shape = (256, 256, 3)
base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')
base_model.trainable = False

inputs = layers.Input(shape=input_shape, name='input_layer')
x = base_model(inputs, training=False)


# Default
x = layers.GlobalAveragePooling2D(name='pooling_layer')(x)
x = layers.Dense(2048, name='Dense2048', activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(2, name='logits')(x)

outputs = layers.Activation('softmax', dtype=tf.float32, name='softmax_float32')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

############################################
# LOGGING
############################################
time_now = datetime.now()
timestamp = time_now.strftime("%m%d%y-%H%M%S")
checkpoint_path = f"checkpoints/{timestamp}/"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path += 'models.hdf5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    monitor="val_accuracy", # save the model weights with best validation accuracy
                                                    save_best_only=True, # only save the best weights
                                                    save_weights_only=False, # want entire model!
                                                    verbose=0) # don't print out whether or not model is being saved 

log_path = f"logs/{timestamp}/"
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)

summary_path = f"checkpoints/{timestamp}/summary.txt"
with open(summary_path, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

csv_logger_path = f"checkpoints/{timestamp}/epochs.csv"
csv_logger_callback = tf.keras.callbacks.CSVLogger(csv_logger_path)

############################################
# FITTING
############################################
model.fit(
    train_data,
    epochs = 5,
    validation_data=test_data,
    callbacks = [checkpoint_callback, log_callback, csv_logger_callback],
    verbose=2
)
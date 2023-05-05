import tensorflow as tf
from food_InceptionResNetV2 import create_model

model = create_model()
model.load_weights('checkpoints/050523-162810/weights.hdf5')

print(model.summary())
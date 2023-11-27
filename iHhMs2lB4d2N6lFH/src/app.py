import tensorflow as tf
from tensorflow.keras import layers as tfl
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.initializers import glorot_uniform
from keras import backend as K
import matplotlib.pyplot as plt
import os

train_data_dir = './data/images/training'
test_data_dir = './data/images/testing'
model_dir = './models/'

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


#parameters
BATCH_SIZE = 32
IMG_SIZE = (128, 128)

#train data
train_dataset, validation_dataset = image_dataset_from_directory(train_data_dir,
                                                                 shuffle=True,
                                                                 validation_split = 0.2,
                                                                 subset = 'both',
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE,
                                                                 seed=42)
#test data
test_dataset = image_dataset_from_directory(test_data_dir,
                                            shuffle=False,
                                            image_size=IMG_SIZE,
                                            seed=42)

class_names = train_dataset.class_names
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

tf.random.set_seed(42)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

#define model
input_shape = IMG_SIZE + (3,)
data_augmentation = tf.identity
base_model = tf.keras.applications.MobileNetV2(input_shape = input_shape, include_top = False, weights = 'imagenet')
base_model.trainable = False
    
inputs = tf.keras.Input(shape=input_shape) 
x = data_augmentation(inputs)
x = preprocess_input(x) 
x = base_model(x, training=False) 
x = tfl.GlobalAveragePooling2D()(x) 
x = tfl.Dropout(0.2)(x)
outputs = tfl.Dense(1, activation = 'linear')(x)
    
trained_model = tf.keras.Model(inputs, outputs)


trained_model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
                     metrics = ['accuracy', f1_score])

#load trained weights
trained_model.load_weights(model_dir + 'MobileNetV2')

#apply on test data
print('Applying trained model on test data')
trained_model.evaluate(test_dataset)
test_predictions = trained_model.predict(test_dataset)

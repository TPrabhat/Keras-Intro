# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
#tf.python.control_flow_ops = tf

with open('/home/prabhat/PycharmProjects/Keras-Intro/small_traffic_set/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

print ("Shape of training dataset: ", np.shape(X_train), np.shape(y_train))


# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

batch_size = 128
num_classes = 5
epochs = 20


with open('/home/prabhat/PycharmProjects/Keras-Intro/small_traffic_set/small_test_traffic.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']


# Build the Fully Connected Neural Network in Keras
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape= (32, 32, 3)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))


# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

print ("Shape of normalized training dataset: ", np.shape(X_normalized))

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

print ("Shape of lables: ", np.shape(y_one_hot))
print ("datapoint: ", y_one_hot[0])

# preprocess data
X_normalized_test = np.array(X_test / 255.0 - 0.5 )
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")


model.compile('adam', 'categorical_crossentropy', ['accuracy'])

history = model.fit(X_normalized, y_one_hot, epochs=10, validation_split=0.2)

metrics = model.evaluate(x=X_normalized_test, y=y_one_hot_test, verbose = 1)

for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np

pict_size = 28
classes_number = 10
img_rows, img_cols = pict_size, pict_size
dims = pict_size * pict_size
dropout_coef = 0.4


def get_trainable_model_full_unreg():
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)

    x = Flatten(input_shape=(img_rows, img_cols))(inputs)
    x = Dense(dims, activation='sigmoid')(x)
    x = Dense(dims, activation='sigmoid')(x)
    x = Dense(dims, activation='sigmoid')(x)
    x = Dense(classes_number, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def get_trainable_model_cutted_unreg():
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)

    x = Flatten(input_shape=(img_rows, img_cols))(inputs)
    x = Dense(dims, activation='sigmoid')(x)
    x = Dense(dims - 4, activation='sigmoid')(x)
    x = Dense(dims - 12, activation='sigmoid')(x)
    x = Dense(classes_number, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def get_trainable_model_dropout():
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)

    x = Flatten(input_shape=(img_rows, img_cols))(inputs)
    # x = Dense(dims, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L1(0.1))(x)
    x = Dense(dims, activation='sigmoid')(x)
    x = Dropout(dropout_coef)(x)
    # x = Dense(dims, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L1(0.1))(x)
    x = Dense(dims, activation='sigmoid')(x)
    x = Dropout(dropout_coef)(x)
    # x = Dense(dims, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L1(0.1))(x)
    x = Dense(dims, activation='sigmoid')(x)
    x = Dropout(dropout_coef)(x)
    # x = Dense(classes_number, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L1(0.1))(x)
    x = Dense(classes_number, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def get_trainable_model_full_cutted_by_dropout():
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)

    x = Flatten(input_shape=(img_rows, img_cols))(inputs)
    x = Dense(int(dims / 2), activation='sigmoid')(x)
    x = Dense(int(dims / 2), activation='sigmoid')(x)
    x = Dense(int(dims / 2), activation='sigmoid')(x)
    x = Dense(classes_number, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


model = get_trainable_model_dropout()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

ctm_epochs = 30
ctm_batch_size = 100
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss', mode='min', min_delta=0.01),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='data/models/tmp/ep{epoch:03d}-loss{loss:.3f}-accuracy{accuracy:.3f}_' + datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.h5',
        monitor='loss', mode='min', save_weights_only=False)
]


def transf_num_to_matrix(y):
    res = np.zeros((len(y), classes_number))
    for i in range(len(y)):
        res[i, y[i]] = 1
    return res


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_images = x_train / 255.0
test_images = x_test / 255.0

y_train_ord = transf_num_to_matrix(y_train)
y_test_ord = transf_num_to_matrix(y_test)

model.fit(train_images, y_train_ord, epochs=ctm_epochs, callbacks=my_callbacks)

y_pred = model.predict(x_test, batch_size=ctm_batch_size)

y_pred_vec = np.argmax(y_pred, axis=1)

f1_ = f1_score(y_pred_vec, y_test, average="micro")
acc = accuracy_score(y_pred_vec, y_test)

print(f"F1: {f1_:.09} accuracy: {acc:.09}")

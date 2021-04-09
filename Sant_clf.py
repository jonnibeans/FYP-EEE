import numpy as np
from tensorflow import keras
from keras.layers import LSTM, Input, TimeDistributed, Lambda, Dense, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

#test the lossweight sanitizer
uweight = 4
pweight = 5.5

n_raw_input = 8
n_lstm_input = 8
n_lstm_steps = 64
n_lstm_hidden = 16
batch_size = 1000


def load_data():
    """x: speed, course, acceleration_x, acceleration_y,
    acceleration_z, roll, pitch, yaw"""

    data_set = loadmat('C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/RawData.mat')
    raw_data = data_set['x_data']
    label = data_set['y_db']

    data_len = len(raw_data)
    indexes = np.arange(data_len)
    np.random.shuffle(indexes)
    raw_data = raw_data[indexes]
    label = label[indexes]

    label = np_utils.to_categorical(label, 3)
    return raw_data, label


def build_driving_behavior_clf():
    x_in = Input(shape=(n_lstm_steps, n_lstm_input))
    x = TimeDistributed(Dense(n_lstm_hidden))(x_in)
    x = LSTM(units=n_lstm_hidden,
             return_sequences=True)(x)
    x = LSTM(units=n_lstm_hidden,
             return_sequences=True)(x)
    x = LSTM(units=n_lstm_hidden,
             return_sequences=False)(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=x_in, outputs=x)

    return model

ep = 750
raw_data, label = load_data()
sanitizer = keras.models.load_model('./Sanitizers/' + str(uweight) + ' ' + str(pweight) + 'sanitizer.h5')
sanitized_data = sanitizer.predict(raw_data)

clf = build_driving_behavior_clf()
clf.compile(optimizer=Adam(lr=0.0002),
            loss=['categorical_crossentropy'],
            metrics=[['acc']])
history = clf.fit(sanitized_data,
                  label,
                  epochs= ep,
                  batch_size=60,
                  validation_split=0.2)

# print loss
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, ep + 1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./Final_CLF/' + str(uweight) + ' ' + str(pweight) + 'loss.png')
plt.show()

# print accuracy
acc_train = history.history['acc']
acc_val = history.history['val_acc']
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./Final_CLF/' + str(uweight) + ' ' + str(pweight) + 'acc.png')
plt.show()

clf.save('./Final_CLF/' + str(uweight) + ' ' + str(pweight) + 'sanitizerclf.h5')

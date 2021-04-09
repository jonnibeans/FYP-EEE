import numpy as np
from tensorflow import keras
from keras.layers import LSTM, Input, TimeDistributed, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from scipy.io import loadmat
import matplotlib.pyplot as plt

uweight = 5
pweight = 6.8
runs = 4000

class PrivacyModel:
    def __init__(self, n_raw_input=8, n_lstm_input=8,
                 n_lstm_steps=64, n_lstm_hidden=16,
                 batch_size=1000):
        self.n_raw_input = n_raw_input
        self.n_lstm_input = n_lstm_input
        self.n_lstm_hidden = n_lstm_hidden
        self.n_lstm_steps = n_lstm_steps
        self.batch_size = batch_size

        self.sanitizer = self.build_sanitizer()
        # self.driving_behavior_clf = self.build_driving_behavior_clf()
        self.driving_behavior_clf = keras.models.load_model('./classifier_prep.h5')
        self.adversary = self.build_adversary()

        print(self.sanitizer.summary())
        print(self.driving_behavior_clf.summary())
        print(self.adversary.summary())

        self.driving_behavior_clf.compile(optimizer=Adam(lr=0.0001),
                                          loss=['categorical_crossentropy'],
                                          metrics=[['acc']])

        self.sanitizer.compile(optimizer=Adam(lr=0.001),
                               loss=['mae'],
                               metrics=[['mae']])

        self.adversary.compile(optimizer=Adam(lr=0.0001),
                               loss=['mse'],
                               metrics=['mse'])

        self.driving_behavior_clf.trainable = False

        utility = self.driving_behavior_clf(self.sanitizer.output)
        privacy = self.adversary(self.sanitizer.output)
        self.combined = Model(self.sanitizer.input, [utility, privacy])

        self.combined.compile(optimizer=Adam(lr=0.0001),
                              loss_weights=[uweight, -pweight],
                              loss=['categorical_crossentropy', 'mse'],
                              metrics=[['acc'], ['mse']])

    def pretrain(self):
        raw_data, label = self.load_data()
        ep = 350
        history = self.driving_behavior_clf.fit(x=raw_data, y=label,
                                                batch_size=60,
                                                epochs=ep,
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
        plt.show()
        self.driving_behavior_clf.save('./classifier_prep.h5')

    def train(self):
        plossplot = []
        accplot = []
        raw_data, label = self.load_data()
        batch_iterator = self.data_generator(raw_data, label)

        #reset the sanitizer and adversary models
        self.sanitizer.fit(raw_data, raw_data, epochs=100)
        self.adversary.fit(raw_data, raw_data, epochs=100)

        sanitized_data = self.sanitizer.predict(raw_data)
        self.driving_behavior_clf.evaluate(sanitized_data, label)

        for seq in range(runs):
            raw_batch, label_batch = batch_iterator.__next__()

            sanitized_batch = self.sanitizer.predict(raw_batch)
            p_loss = self.adversary.train_on_batch(sanitized_batch, raw_batch)
            u_loss = self.driving_behavior_clf.train_on_batch(sanitized_batch, label_batch)
            c_loss = self.combined.train_on_batch(raw_batch, [label_batch, raw_batch])

            plossplot.append(c_loss[2])
            accplot.append(c_loss[3])

            print('\r step:%d, p_loss:%.4f, u_loss:%.4f, u_acc:%.4f'
                  % (seq, c_loss[2], c_loss[1], c_loss[3]))
        print('u:' + str(uweight) + ' p:' +str(pweight))
        x = np.linspace(0, runs, runs)

        fig, ax1 = plt.subplots()
        ax1.plot(x, accplot, 'b', label='Classifier Accuracy')
        ax2 = ax1.twinx()
        ax2.plot(x, plossplot, 'r', label='Privacy Loss')
        ax1.set_xlabel('Sequence')
        ax1.set_ylabel('CLF Accuracy', color='b')
        ax2.set_ylabel('Privacy Ability', color='r')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax1.set_ylim([0.8, 1])
        plt.legend()
        fig.tight_layout()
        fig.savefig('./Sanitizers/' + str(uweight) + ' ' + str(pweight) + 'lossweight.png')
        self.sanitizer.save("./Sanitizers/" + str(uweight) + ' ' + str(pweight) + "sanitizer.h5")
        plt.show()

    def build_driving_behavior_clf(self):
        x_in = Input(shape=(self.n_lstm_steps, self.n_lstm_input))
        x = TimeDistributed(Dense(self.n_lstm_hidden))(x_in)
        x = LSTM(units=self.n_lstm_hidden,
                 return_sequences=True)(x)
        x = LSTM(units=self.n_lstm_hidden,
                 return_sequences=True)(x)
        x = LSTM(units=self.n_lstm_hidden,
                 return_sequences=False)(x)
        x = Dense(3, activation='softmax')(x)
        model = Model(inputs=x_in, outputs=x, name='classifier')

        return model

    def build_sanitizer(self):
        x_in = Input(shape=(self.n_lstm_steps, self.n_raw_input))
        x = TimeDistributed(Dense(7))(x_in)
        x = TimeDistributed(Dense(self.n_lstm_input))(x)
        model = Model(inputs=x_in, outputs=x, name='sanitizer')

        return model

    def build_adversary(self):
        x_in = Input(shape=(self.n_lstm_steps, self.n_lstm_input))
        x = TimeDistributed(Dense(self.n_raw_input))(x_in)
        model = Model(inputs=x_in, outputs=x, name='privacy')

        return model

    @staticmethod
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

    @staticmethod
    def data_generator(x, y, batch_size=1000):
        samples_per_epoch = x.shape[0]
        number_of_batches = int(samples_per_epoch / batch_size)
        counter = 0

        indexes = np.arange(samples_per_epoch)
        np.random.shuffle(indexes)
        x = x[indexes]
        y = y[indexes]

        while True:
            x_batch = x[batch_size * counter: batch_size * (counter + 1)]
            y_batch = y[batch_size * counter: batch_size * (counter + 1)]
            counter += 1

            if (counter + 1) >= number_of_batches:
                indexes = np.arange(samples_per_epoch)
                np.random.shuffle(indexes)
                x = x[indexes]
                y = y[indexes]
                counter = 0

            yield x_batch, y_batch

test = PrivacyModel()
# test.pretrain()

test.train()

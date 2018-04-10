import pandas as pd

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Conv1D
from keras.layers import LSTM, MaxPooling1D, Embedding
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model


class embedding():
    def __init__(self, X_train, y_train, X_pred, max_features=20000, maxlen=150,
                 embedding_size=300, kernel_size=5,
                 filters=64, pool_size=4, lstm_output_size=70,
                 batch_size=256, epochs=5):
        self.max_features = max_features
        self.maxlen = maxlen
        self.X_pred = X_pred
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size
        self.lstm_output_size = lstm_output_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.init_X_train = X_train
        self.init_y_train = y_train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_train, y_train,
                                                                                test_size=0.25,
                                                                                random_state=42)

    def train_test(self):
        return (self.X_train, self.X_test, self.y_train, self.y_test )

    def train(self, model, filename, weightName, submission):
        print('Train...')
        model.fit(self.X_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=(self.X_test, self.y_test))
        score, acc = model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
        y_pred = model.predict(self.X_pred, batch_size=1024, verbose=1)

        featureNeam = filename

        submission = submission.iloc[:y_pred.shape[0], :]
        submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
        filename = 'modelBackup/' + str(filename) + '.csv'
        submission.to_csv(filename, index=False)
        weightName = 'weightBackup/' + str(weightName)
        model.save_weights(weightName)

        embedding.outputFetures(self,model,featureNeam)


    def outputFetures(self,model,filename):
        y_pred = model.predict(self.init_X_train, batch_size=1024, verbose=1)
        y_pred_y = model.predict(self.X_pred, batch_size=1024, verbose=1)
        train_fileName = 'train_outcome/' + filename + '.csv'
        test_fileName = 'test_outcome/' + filename + '.csv'

        y_pred = pd.DataFrame(y_pred)
        y_pred_y = pd.DataFrame(y_pred_y)
        y_pred.to_csv(train_fileName, index=False)
        y_pred_y.to_csv(test_fileName, index=False)


    def cnn_lstm_model(self,embedflag,embedding_matrix):
        print('Build model...')
        model = Sequential()
        if embedflag:
            model.add(Embedding(self.max_features, self.embedding_size, weights=[embedding_matrix],
                                input_length=self.maxlen))
        else:
            model.add(Embedding(self.max_features, self.embedding_size,
                                input_length=self.maxlen))

        model.add(Dropout(0.25))
        model.add(Conv1D(self.filters,
                         self.kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(LSTM(self.lstm_output_size))
        model.add(Dense(6))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def cnn_lstm_features(self, embedflag, embedding_matrix, filename):
        model = embedding.cnn_lstm_model(self, embedflag, embedding_matrix)
        model.layers.pop()
        model.layers.pop()
        embedding.outputFetures(self,model,filename)

    def cnn_lstm_weights(self, weightName, filename, submission, embedding_matrix, embedflag=False):
        model = embedding.cnn_lstm_model(self,embedflag,embedding_matrix)
        embedding.train(self, model, filename, weightName, submission)

    def cnn_1d3(self, embedflag, embedding_matrix):
        sequence_input = Input(shape=(self.maxlen,))
        if embedflag:
            x = Embedding(self.max_features, self.embedding_size, weights=[embedding_matrix])(sequence_input)
        else:
            x = Embedding(self.max_features, self.embedding_size)(sequence_input)
        x = Dropout(0.2)(x)
        x = Conv1D(self.filters, self.kernel_size, activation='relu')(x)
        x = MaxPooling1D(self.pool_size)(x)
        x = Conv1D(self.filters, self.kernel_size, activation='tanh')(x)
        x = MaxPooling1D(self.pool_size)(x)
        x = Conv1D(self.filters, self.kernel_size, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(6, activation='sigmoid')(x)
        model = Model(inputs=sequence_input, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def cnn_1d3_train(self, weightName, filename, submission, embedding_matrix, embedflag=False):
        model = embedding.cnn_1d3(self, embedflag, embedding_matrix)
        embedding.train(self, model, filename, weightName, submission)

    def cnn_1d3_features(self,filename,embedding_matrix, embedflag=False):
        model = embedding.cnn_1d3(self, embedflag, embedding_matrix)
        model.layers.pop()
        model.layers.pop()
        embedding.outputFetures(self,model,filename)

    def gru_model(self, embedflag, embedding_matrix):
        inp = Input(shape=(self.maxlen,))
        if embedflag:
            x = Embedding(self.max_features, self.embedding_size, weights=[embedding_matrix])(inp)
        else:
            x = Embedding(self.max_features, self.embedding_size)(inp)
        x = Bidirectional(LSTM(self.lstm_output_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def gru_train(self, weightName, filename, submission, embedding_matrix, embedflag=False):
        model = embedding.gru_model(self, embedflag, embedding_matrix)
        embedding.train(self, model, filename, weightName, submission)



#! -*- coding: utf-8 -*-

from Capsule_Keras import *
from keras import utils
from keras.datasets import mnist
from keras.models import Model
from keras.layers import *
from keras import backend as K


#准备训练数据
batch_size = 128
num_classes = 10
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


#准备自定义的测试样本
#对测试集重新排序并拼接到原来测试集，就构成了新的测试集，每张图片有两个不同数字
idx = range(len(x_test))
np.random.shuffle(idx)
X_test = np.concatenate([x_test, x_test[idx]], 1)
Y_test = np.vstack([y_test.argmax(1), y_test[idx].argmax(1)]).T
X_test = X_test[Y_test[:,0] != Y_test[:,1]] #确保两个数字不一样
Y_test = Y_test[Y_test[:,0] != Y_test[:,1]]
Y_test.sort(axis=1) #排一下序，因为只比较集合，不比较顺序


#搭建普通CNN分类模型
input_image = Input(shape=(None,None,1))
cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
cnn = AveragePooling2D((2,2))(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = GlobalAveragePooling2D()(cnn)
dense = Dense(128, activation='relu')(cnn)
output = Dense(10, activation='sigmoid')(dense)

model = Model(inputs=input_image, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))

Y_pred = model.predict(X_test) #用模型进行预测
greater = np.sort(Y_pred, axis=1)[:,-2] > 0.5 #判断预测结果是否大于0.5
Y_pred = Y_pred.argsort()[:,-2:] #取最高分数的两个类别
Y_pred.sort(axis=1) #排序，因为只比较集合

acc = 1.*(np.prod(Y_pred == Y_test, axis=1)).sum()/len(X_test)
print u'CNN+Pooling，不考虑置信度的准确率为：%s'%acc
acc = 1.*(np.prod(Y_pred == Y_test, axis=1)*greater).sum()/len(X_test)
print u'CNN+Pooling，考虑置信度的准确率为：%s'%acc



#搭建CNN+Capsule分类模型
input_image = Input(shape=(None,None,1))
cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
cnn = AveragePooling2D((2,2))(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = Reshape((-1, 128))(cnn)
capsule = Capsule(10, 16, 3, True)(cnn)
output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(10,))(capsule)

model = Model(inputs=input_image, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))

Y_pred = model.predict(X_test) #用模型进行预测
greater = np.sort(Y_pred, axis=1)[:,-2] > 0.5 #判断预测结果是否大于0.5
Y_pred = Y_pred.argsort()[:,-2:] #取最高分数的两个类别
Y_pred.sort(axis=1) #排序，因为只比较集合

acc = 1.*(np.prod(Y_pred == Y_test, axis=1)).sum()/len(X_test)
print u'CNN+Capsule，不考虑置信度的准确率为：%s'%acc
acc = 1.*(np.prod(Y_pred == Y_test, axis=1)*greater).sum()/len(X_test)
print u'CNN+Capsule，考虑置信度的准确率为：%s'%acc



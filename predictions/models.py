import pandas as pd
import glob
import os
import numpy as np
import copy
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import keras
from keras.layers import Dropout
from numpy import argmax
from sklearn.metrics import accuracy_score
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import hankel
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import xgboost as xgboost



def LSTM_tuning(X,additional_data = None, lag = 30, layers = 2, train_ratio = 0.7):

    scaler = MinMaxScaler(feature_range=(0,1))
    x = scaler.fit_transform(np.array(X).reshape(-1,1))
    training_size = int(len(x) * train_ratio)
    train_data, test_data = x[0:training_size,:], x[training_size:len(x),:1]
    

    def create_dataset(dataset, time_step=1):
      dataX, dataY = [], []
      for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
      return np.array(dataX), np.array(dataY)
    


    time_step = lag
    X_train, y_train = create_dataset(train_data, time_step)

    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    X_train.shape
    add_vars_num = 0

    if str(type(additional_data)) == "<class 'pandas.core.frame.DataFrame'>":
        add_vars_num = len(additional_data.columns.tolist())
        for column in additional_data.columns.tolist():
            scaler2 = MinMaxScaler(feature_range=(0,1))
            add_sc = scaler2.fit_transform(np.array(additional_data[str(column)]).reshape(-1,1))
            train_add, test_add = add_sc[0:training_size,:], add_sc[training_size:len(x),:1]

            X_train_add, _ = create_dataset(train_add, time_step)
            X_test_add, _ = create_dataset(test_add, time_step)
            X_train_add = X_train_add.reshape(X_train_add.shape[0],X_train_add.shape[1] , 1)
            X_test_add = X_test_add.reshape(X_test_add.shape[0],X_test_add.shape[1] , 1)

            X_test = np.concatenate([X_test,X_test_add],axis = 2)
            X_train = np.concatenate([X_train,X_train_add],axis = 2)

    else:
        print('No additional data')
    

    model = Sequential()
    model.add(LSTM(64*layers, input_shape=(lag, 1 + add_vars_num)))
    for i in range(0,layers):
        i += 1
        model.add(Dense(64*layers/(i-0.5), activation='relu'))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    #model.summary()
    
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict_norm = scaler.inverse_transform(train_predict)
    test_predict_norm = scaler.inverse_transform(test_predict)

    y_train_norm = scaler.inverse_transform(train_data[lag:])
    y_test_norm = scaler.inverse_transform(test_data[lag:])

    len(test_predict)
    
    def MSE(real,pred):
        return np.sum((real - pred)**2)/(len(real))


    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true = y_true + 1
        y_pred = y_pred + 1

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def smape(act,forc):
        return 100/len(act) * np.sum(2 * np.abs(forc - act) / (np.abs(act) + np.abs(forc)))



    mse_obs = MSE(test_data[lag:], test_predict) 
    
    mape_obs = mean_absolute_percentage_error(test_data[lag:], test_predict)

    smape_obs = smape(test_data[lag:], test_predict)

    return np.concatenate(train_predict_norm).ravel().tolist(),np.concatenate(test_predict_norm).ravel().tolist() ,np.concatenate(y_train_norm).ravel().tolist() ,np.concatenate(y_test_norm).ravel().tolist() 



def LSTM_tuning_ud(X,additional_data = None, lag = 10, layers = 2, train_ratio = 0.7):

    scaler = MinMaxScaler(feature_range=(0,1))
    x = scaler.fit_transform(np.array(X).reshape(-1,1))
    training_size = int(len(x) * train_ratio)
    train_data, test_data = x[0:training_size,:], x[training_size:len(x),:1]
    

    def create_dataset(dataset, time_step=1):
      dataX, dataY = [], []
      for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
      return np.array(dataX), np.array(dataY)
    


    time_step = lag
    X_train, y_train = create_dataset(train_data, time_step)
    y_train_bin = np.array([0. if i < 0 else 1. for i in np.diff(y_train)])

    X_test, y_test = create_dataset(test_data, time_step)
    y_test_bin = np.array([0. if i < 0 else 1. for i in np.diff(y_test)])
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    add_vars_num = 0

    if str(type(additional_data)) == "<class 'pandas.core.frame.DataFrame'>":
        add_vars_num = len(additional_data.columns.tolist())
        for column in additional_data.columns.tolist():
            scaler2 = MinMaxScaler(feature_range=(0,1))
            add_sc = scaler2.fit_transform(np.array(additional_data[str(column)]).reshape(-1,1))
            train_add, test_add = add_sc[0:training_size,:], add_sc[training_size:len(x),:1]

            X_train_add, _ = create_dataset(train_add, time_step)
            X_test_add, _ = create_dataset(test_add, time_step)
            X_train_add = X_train_add.reshape(X_train_add.shape[0],X_train_add.shape[1] , 1)
            X_test_add = X_test_add.reshape(X_test_add.shape[0],X_test_add.shape[1] , 1)

            X_test = np.concatenate([X_test,X_test_add],axis = 2)
            X_train = np.concatenate([X_train,X_train_add],axis = 2)

    else:
        print('No additional data')
    

    X_train_bin = np.delete(X_train, 0, 0)
    X_test_bin = np.delete(X_test, 0, 0)

    y_train_bin = to_categorical(y_train_bin)
    y_test_bin = to_categorical(y_test_bin)

    X_train_bin.shape
    y_train_bin.shape

    model = Sequential()
    model.add(LSTM(500, input_shape=(lag, 1 + add_vars_num)))
    model.add(Dropout(0.1))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))
    #opt = keras.optimizers.adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
    #model.summary()
    
    model.fit(X_train_bin, y_train_bin, epochs=2000, verbose=1)
    
    train_predict = model.predict(X_train_bin)
    
    np.array([0. if i < 0 else 1. for i in train_predict.reshape(2,455)[1]])

    

    argmax(train_predict,axis=1)


    scores = model.evaluate(X_train_bin, y_train_bin, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
    test_predict= model.predict(X_test_bin)
    argmax(test_predict,axis=1)
    scores2 = model.evaluate(X_test_bin, y_test_bin, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    


    

    return argmax(train_predict,axis=1), argmax(test_predict,axis=1) , argmax(y_train_bin,axis=1) , argmax(y_test_bin,axis=1), scores2[1]





def XGB_tuning_ud(X,additional_data = None, lag = 10, layers = 2, train_ratio = 0.7):
    
    
    scaler = MinMaxScaler(feature_range=(0,1))
    x = scaler.fit_transform(np.array(X).reshape(-1,1))
    training_size = int(len(x) * train_ratio)
    train_data, test_data = x[0:training_size,:], x[training_size:len(x),:1]
    

    def create_dataset(dataset, time_step=1):
      dataX, dataY = [], []
      for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
      return np.array(dataX), np.array(dataY)
    


    time_step = lag
    X_train, y_train = create_dataset(train_data, time_step)
    y_train_bin = np.array([0. if i < 0 else 1. for i in np.diff(y_train)])

    X_test, y_test = create_dataset(test_data, time_step)
    y_test_bin = np.array([0. if i < 0 else 1. for i in np.diff(y_test)])
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    add_vars_num = 0

    if str(type(additional_data)) == "<class 'pandas.core.frame.DataFrame'>":
        add_vars_num = len(additional_data.columns.tolist())
        for column in additional_data.columns.tolist():
            scaler2 = MinMaxScaler(feature_range=(0,1))
            add_sc = scaler2.fit_transform(np.array(additional_data[str(column)]).reshape(-1,1))
            train_add, test_add = add_sc[0:training_size,:], add_sc[training_size:len(x),:1]

            X_train_add, _ = create_dataset(train_add, time_step)
            X_test_add, _ = create_dataset(test_add, time_step)
            X_train_add = X_train_add.reshape(X_train_add.shape[0],X_train_add.shape[1] , 1)
            X_test_add = X_test_add.reshape(X_test_add.shape[0],X_test_add.shape[1] , 1)

            X_test = np.concatenate([X_test,X_test_add],axis = 2)
            X_train = np.concatenate([X_train,X_train_add],axis = 2)

    else:
        print('No additional data')
    

    X_train_bin = np.delete(X_train, 0, 0)
    X_test_bin = np.delete(X_test, 0, 0)

    X_train_bin = X_train_bin.reshape(X_train_bin.shape[0], X_train_bin.shape[1] * X_train_bin.shape[2])
    X_test_bin = X_test_bin.reshape(X_test_bin.shape[0], X_test_bin.shape[1] * X_test_bin.shape[2])

    xgb_cl = xgboost.XGBClassifier()
    xgb_cl.fit(X_train_bin, y_train_bin)

    xgb_train_pred =  xgb_cl.predict(X_train_bin)

    xgb_test_pred =  xgb_cl.predict(X_test_bin)

    return xgb_train_pred, xgb_test_pred, y_train_bin, y_test_bin



def XGB_tuning(X,additional_data = None, lag = 10, layers = 2, train_ratio = 0.7):
    
    
    scaler = MinMaxScaler(feature_range=(0,1))
    x = scaler.fit_transform(np.array(X).reshape(-1,1))
    training_size = int(len(x) * train_ratio)
    train_data, test_data = x[0:training_size,:], x[training_size:len(x),:1]
    

    def create_dataset(dataset, time_step=1):
      dataX, dataY = [], []
      for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
      return np.array(dataX), np.array(dataY)
    


    time_step = lag
    X_train, y_train = create_dataset(train_data, time_step)

    X_test, y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    add_vars_num = 0

    if str(type(additional_data)) == "<class 'pandas.core.frame.DataFrame'>":
        add_vars_num = len(additional_data.columns.tolist())
        for column in additional_data.columns.tolist():
            scaler2 = MinMaxScaler(feature_range=(0,1))
            add_sc = scaler2.fit_transform(np.array(additional_data[str(column)]).reshape(-1,1))
            train_add, test_add = add_sc[0:training_size,:], add_sc[training_size:len(x),:1]

            X_train_add, _ = create_dataset(train_add, time_step)
            X_test_add, _ = create_dataset(test_add, time_step)
            X_train_add = X_train_add.reshape(X_train_add.shape[0],X_train_add.shape[1] , 1)
            X_test_add = X_test_add.reshape(X_test_add.shape[0],X_test_add.shape[1] , 1)

            X_test = np.concatenate([X_test,X_test_add],axis = 2)
            X_train = np.concatenate([X_train,X_train_add],axis = 2)

    else:
        print('No additional data')
    

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    X_train.shape

    xgb_lin = xgboost.XGBRegressor(n_estimators=1000)
    xgb_lin.fit(X_train, y_train)

    xgb_train_pred =  xgb_lin.predict(X_train)


    xgb_test_pred =  xgb_lin.predict(X_test)


    train_predict_norm = scaler.inverse_transform(xgb_train_pred.reshape(-1,1))
    test_predict_norm = scaler.inverse_transform(xgb_test_pred.reshape(-1,1))

    y_train_norm = scaler.inverse_transform(train_data[lag:])
    y_test_norm = scaler.inverse_transform(test_data[lag:])



    return np.concatenate(train_predict_norm).ravel().tolist(),np.concatenate(test_predict_norm).ravel().tolist() ,np.concatenate(y_train_norm).ravel().tolist() ,np.concatenate(y_test_norm).ravel().tolist()




def profit(xgb_test_pred, y_test_bin, X):
    real_prices = X.tolist()[-len(y_test_bin)-1:]
    real_prices_diff = np.diff(real_prices)
    bank = 0
    for t in xgb_test_pred:
        if t == 0:
            bank += (-real_prices_diff[t])
        else: 
            bank += (real_prices_diff[t])
    return bank
    

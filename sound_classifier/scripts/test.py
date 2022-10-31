import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix
import seaborn as sn

from data import *
from model import *


def test_model(model, test_data):
    '''Testing the data'''
    correctly_predicted = 0
    for num, data in enumerate(test_data):
        label = data[1]
        signal = data[0].reshape(np.shape(test_data[0][0])[0], np.shape(test_data[0][0])[1], 1)
        
        model_out = model.predict([signal])[0]
        print(model_out.round(2))
        y_true = np.argmax(label)
        y_pred = np.argmax(model_out)

        if y_pred == y_true:
            correctly_predicted += 1

    return correctly_predicted/len(test_data)

model_path = '/home/miguel/sound_classification/src/MobileRobots/sound_tools/sound_classifier/model_test'
train_path = '/home/miguel/sound_classification/src/datasets/dcase2016_task2_train_dev/train_data'
test_path = '/home/miguel/sound_classification/src/datasets/dcase2016_task2_train_dev/test_data'
train_data = create_data_features(train_path, False)
test_data = create_data_features(test_path, False)
model = create_model(train_data, model_path)
accuracy = test_model(model, test_data)
print('ACCURACY: {}'.format(accuracy))
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix
import seaborn as sn

from data import *
from model import *

TRAIN_PART = 0.9  # the training dataset represents 90% of the whole dataset
CODE_PATH = '/home/miguel/sound_classification/src/MobileRobots/sound_tools/sound_classifier'
MODEL_PATH = CODE_PATH + '/complex_models/models10_aug_amp0_cough'  # models10_aug_amp0_cough
FEATURES_PATH = CODE_PATH + '/data_features'
DATA_PATH = '/home/miguel/sound_classification/src/datasets/dcase2016_task2_train_dev'
RAND_SEED = 4


def get_data_features(data_path, augmented):
    '''Running the training and the testing in the train and test dataset for our model'''
    # Check if data file already exists
    if augmented: dataset_file_name = 'data{}_aug_amp{}.npy'.format(len(labelID), MIN_AMP_RATIO_THRESHOLD)
    else: dataset_file_name = 'data{}_amp{}.npy'.format(len(labelID), MIN_AMP_RATIO_THRESHOLD)
    dataset_file = FEATURES_PATH + '/' + dataset_file_name
    if os.path.exists(dataset_file):
        data_features = np.load(dataset_file, allow_pickle=True)
        print('\nData features loaded!')
    else:
        start = time.time()
        data_features = create_data_features(data_path, augmented)
        np.save(dataset_file, data_features)
        print('\nTIME create_train_test_data: {}'.format(time.time() - start))
        # It takes 2281 seconds to load all UrbanSound8K dataset (8732 files) with all freq spectrum
    return data_features, dataset_file_name[:-4]


def train_test_split(features, augmented, n):
    train_data = []
    test_data = []
    if augmented:
        data_per_class = int( len(features) / len(labelID) )
        augmented_data = int( data_per_class / SAMPLES_PER_CLASS )
        # loop through each class
        for id in range(len(labelID)):
            test_data.append( features[id*data_per_class + n*augmented_data] )
            for i in range(SAMPLES_PER_CLASS):
                if i == n:
                    continue
                train_data.extend( features[id*data_per_class+i*augmented_data : id*data_per_class+(i+1)*augmented_data] )
    else:
        # loop through each class
        for id in range(len(labelID)):
            test_data.append( features[id*SAMPLES_PER_CLASS + n] )
            for i in range(SAMPLES_PER_CLASS):
                if i == n:
                    continue
                # print(id*SAMPLES_PER_CLASS+i)
                train_data.append( features[id*SAMPLES_PER_CLASS+i] )  

    return train_data, test_data


def test_model(model, test_data):
    '''Testing the data'''
    correctly_predicted = 0
    for num, data in enumerate(test_data):
        label = data[1]
        signal = data[0].reshape(np.shape(test_data[0][0])[0], np.shape(test_data[0][0])[1], 1)
        
        model_out = model.predict([signal])[0]
        y_true = np.argmax(label)
        y_pred = np.argmax(model_out)

        if y_pred == y_true:
            correctly_predicted += 1

        y_true_list.append(np.argmax(label))
        y_pred_list.append(np.argmax(model_out))

    return correctly_predicted/len(test_data)


def complete_test():
    augmented = True
    data_path = DATA_PATH + '/dcase2016_task2_train'
    features, train_file = get_data_features(data_path, augmented)
    acc = []
    for n in range(20):
    # n=19
        print('\nITERATION: {}\n'.format(n))
        train_data, test_data = train_test_split(features, augmented, n)
        # Loading the saved model
        model_path = MODEL_PATH + '/model_{}epochs_n{}_{}'.format(N_EPOCH, n, train_file)
        start = time.time()
        model = create_model(train_data, model_path)
        print('\nTIME create_model: {}'.format(time.time() - start))
        accuracy = test_model(model, test_data)
        acc.append(accuracy)
        print('ACCURACY: {}'.format(accuracy))
    return sum(acc)/len(acc)


def final_test():
    train_path = '/home/miguel/sound_classification/src/datasets/dcase2016_task2_train_dev/train_data'
    test_path = '/home/miguel/sound_classification/src/datasets/dcase2016_task2_train_dev/test_data'
    train_data = create_data_features(train_path, augment=True)
    test_data = create_data_features(test_path, augment=False)
    model = create_model(train_data, MODEL_PATH)
    return test_model(model, test_data)


if __name__=="__main__":
    y_true_list = []
    y_pred_list = []

    accuracy = complete_test()
    # accuracy = final_test()
    print('AVERAGE ACCURACY: {}'.format(accuracy))

    cm = confusion_matrix(y_true_list, y_pred_list, normalize='pred')*100
    # plt.figure(figsize = (10,7))
    plt.figure()
    sn.heatmap(cm, annot=True, xticklabels=labelID, yticklabels=labelID, fmt='.3g', cmap="gray_r")
    # sn.heatmap(cm, annot=True, xticklabels=labelID, yticklabels=labelID, fmt='.3g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # plt.subplots_adjust(bottom=0.1, left=0.12, top = 0.9, right=0.75, hspace=0.2, wspace=0.05)
    plt.tight_layout()
    plt.show()




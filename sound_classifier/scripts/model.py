'''Creating the neural network using tensorflow'''
#Importing the required libraries
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data import *

'''Setting up the model which will help with tensorflow models'''
LR = 1e-3
N_EPOCH = 15

def get_model(shape1, shape2):
    tf.compat.v1.reset_default_graph()
    # None - many or a number of or how many images of
    # shape1 X shape2 - dimensions of the image
    # 1 - with one color channel
    convnet = input_data(shape=[None, shape1, shape2, 1], name='input')
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, len(labelID), activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    return tflearn.DNN(convnet, tensorboard_dir='log')


def create_model(train_data, model_path):
    shape1 = np.shape(train_data[0][0])[0]
    shape2 = np.shape(train_data[0][0])[1]
    model = get_model(shape1, shape2)

    #Loading the saved model
    if os.path.exists(model_path +'.meta'):
        model.load(model_path)
        print('\nModel loaded!')

    else:
        '''Setting up the features and lables'''
        # X = np.array([i[0] for i in train_data])
        X = np.array([i[0] for i in train_data]).reshape(-1, shape1, shape2, 1)
        Y = [i[1] for i in train_data]

        '''Fitting the data into our model'''
        model.fit({'input': X}, {'targets': Y}, n_epoch=N_EPOCH, shuffle=True, validation_set=0.05, 
            snapshot_step=500, show_metric=True, run_id=model_path, batch_size=16) # batch_size=80 worked fine for the whole data...
        model.save(model_path)
        print('\nModel saved!')

    return model
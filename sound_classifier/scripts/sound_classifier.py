#!/usr/bin/env python3

import rospy
import numpy as np
import librosa
import actionlib

import sound_utils
from utils import *
from sound_msgs.msg import *
from geometry_msgs.msg import *
from model import *

MODEL_NAME = '/home/miguel/sound_classification/src/MobileRobots/sound_tools/sound_classifier/models/model10_aug_amp0.05_cough'
labelID = ['clearthroat', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock', 'laughter', 'pageturn', 'phone', 'speech']
spec_count = 0

class SoundClassifier():
    def __init__(self):
        self._as = actionlib.SimpleActionServer("classifier", ClassifierAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()

    def execute_cb(self, goal):
        rospy.loginfo("Received Spectrogram length: " + str(len(goal.data.frames)))
        spectrogram = self.matrix_from_msg(goal.data)
        global spec_count
        spec_count += 1
        save_spectrogram(spectrogram, str(spec_count))
        # rospy.logerr(np.max(spectrogram))
        classes_prediction = self.predict_class(spectrogram) # return predicted class or array of probabilities ?
        predicted_class = labelID[np.argmax(classes_prediction)]
        rospy.loginfo(classes_prediction)
        rospy.logerr(predicted_class) # speech, phone, keyboard, clearthroat, knock
        #self._as.set_preempted()
        #self._as.is_preempt_requested()
        feedback = ClassifierFeedback()
        feedback.feedback = "Providing Feedback"
        self._as.publish_feedback(feedback)
        result = ClassifierResult()
        result.classified_data = str(spectrogram.shape) # (10, 1025)
        self._as.set_succeeded(result)
        #self._as.set_aborted(self.result)

    def run(self):
        rospy.spin()

    
    def matrix_from_msg(self, data):
        # Convert type SoundFrameArray to numpy matrix
        stft = []
        for frame in data.frames:
            spec_frame = [point.amplitude for point in frame.points]
            stft.append(spec_frame)
        stft = np.array(stft).T
        # stft = stft.T[:400, :] # cut until ~4000Hz
        stft = np.where( stft < MIN_AMP_RATIO_THRESHOLD*np.max(stft), 0, stft )
        spectrogram = librosa.amplitude_to_db(stft, ref=HEARING_THRESHOLD)
        return spectrogram


    def predict_class(self, spectrogram):
        shape1 = spectrogram.shape[0]
        shape2 = spectrogram.shape[1]
        model = get_model(shape1, shape2)
        model.load(MODEL_NAME)
        signal = spectrogram.reshape(shape1, shape2, 1)
        # model = tf.keras.models.load_model(MODEL_NAME)
        # signal = spectrogram.reshape(-1, shape1, shape2, 1)
        classes = model.predict([signal])[0]
        return classes


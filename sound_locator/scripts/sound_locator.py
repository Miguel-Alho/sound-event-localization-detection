import rospy
import numpy as np
import math
import itertools
import actionlib
import matplotlib.pyplot as plt
import pyroomacoustics as pra

import sound_utils
from utils import *
from sound_msgs.msg import *
from geometry_msgs.msg import *
from visualization_msgs.msg import *


class SoundLocator():
    def __init__(self):
        self.count = 0
        self.previous_coords = dict()
        self.sound_sources = parse_sources_base()
        self.mics = parse_microphones_base()
        self.tf_mics = self.mics_transform()
        self.mics_area = self.get_mics_area()
        self.distances = self.get_distances()
        if CLASSIFIER:
            self.classifier_client = actionlib.SimpleActionClient('classifier', ClassifierAction)
            rospy.loginfo('waiting for classifier server...')
            self.classifier_client.wait_for_server()
        self.marker_publisher = rospy.Publisher('poses_markers', MarkerArray, queue_size=10)
        rospy.Subscriber('raw_data', SoundSample, self.mic_data_callback)

    def mic_data_callback(self, data:SoundSample):
        self.process_data(data, rospy.get_time(), self.count)
        self.count += 1

    def feedbackCB(self, feedback):
        rospy.loginfo('Feedback was ' + str(feedback))

    def mics_transform(self):
        tf_mics = parse_microphones_base()
        for mic in tf_mics:
            mic.x += MICS_CENTER[0]
            mic.y += MICS_CENTER[1]
            mic.z += MICS_CENTER[2]
        return tf_mics

    def get_mics_markers(self):
        markers = []
        index = 99
        for mic in self.mics:
            index += 1
            pose = Pose()
            marker = Marker()
            marker.id = index
            marker.action = Marker.ADD
            marker.type = Marker.CUBE
            # marker.scale.x = 0.1
            # marker.scale.y = 0.1
            # marker.scale.z = 0.1
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.header.frame_id = 'world'
            pose.position.x = mic.x
            pose.position.y = mic.y
            pose.position.z = mic.z
            pose.orientation.w = 1.0
            marker.pose = pose
            markers.append(marker)
        return markers


    def process_data(self, data:SoundSample, start_time, count):  
        #
        # Process of localization
        #
        # Average all frames of the sample
        avg_data = self.average_data(data)
        # Compare mics amplitudes by freq to filter noise and sounds coming from too far away
        imp_data = self.get_important_data(avg_data)
        if not imp_data:
            rospy.logerr('No significant data in this time frame...')
        else:
            srcsDict = self.estimate_poses(imp_data, count)
            # for k,v in srcsDict.items():
            #     rospy.logerr('{}: {}'.format(k,v))
            rospy.loginfo('\nTIME Localization: {}'.format(rospy.get_time() - start_time))

            # Gather frequencies coming from similar positions
            avg_srcs_freqDict, errorDict = self.gather_close_coords(srcsDict)
            for k,v in avg_srcs_freqDict.items():
                k = np.array(k) - np.array(MICS_CENTER)
                rospy.logerr('{}: {}'.format(k,v))

            # Create filtered spectrogram
            spectrograms = self.get_spectrograms(avg_srcs_freqDict, data)

            
            #
            # Classify each spectrogram
            #
            if CLASSIFIER:
                classification_time = rospy.get_time()
                for spectrogram in spectrograms.frame_arrays:
                    goal = ClassifierGoal()
                    goal.data = spectrogram
                    self.classifier_client.send_goal(goal, feedback_cb=self.feedbackCB)
                    self.classifier_client.wait_for_result()
                    result = self.classifier_client.get_result()
                    rospy.loginfo('Provided Result : ' + str(result))
                rospy.loginfo('\nTIME Classification: {}'.format(rospy.get_time() - classification_time))
                print('\n')


            #
            # RVIZ
            #
            markers = MarkerArray()

            # pose = Pose()
            # marker = Marker()
            # marker.id = 1000
            # marker.action = Marker.ADD
            # marker.type = Marker.MESH_RESOURCE
            # marker.mesh_resource = "package://sound_bring_up/mesh/chassis_base.0.dae"
            # marker.scale.x = 1.0
            # marker.scale.y = 1.0
            # marker.scale.z = 1.0
            # marker.mesh_use_embedded_materials = True
            # marker.header.frame_id = 'world'
            # pose.position.x = 0.0
            # pose.position.y = 0.0
            # pose.position.z = 0.0637 + 0.313/2
            # pose.orientation.w = 1.0
            # marker.pose = pose
            # markers.markers.append(marker)

            # pose = Pose()
            # marker = Marker()
            # marker.id = 1001
            # marker.action = Marker.ADD
            # marker.type = Marker.MESH_RESOURCE
            # marker.mesh_resource = "package://sound_bring_up/mesh/topbox.0.dae"
            # marker.scale.x = 1.0
            # marker.scale.y = 1.0
            # marker.scale.z = 1.0
            # marker.mesh_use_embedded_materials = True
            # marker.header.frame_id = 'world'
            # pose.position.x = 0.0
            # pose.position.y = 0.0
            # pose.position.z = 0.5325 + 0.0637 + 0.313/2
            # pose.orientation.w = 1.0
            # marker.pose = pose
            # markers.markers.append(marker)

            index = 0
            for k,v in errorDict.items():                
                # SS marker
                tf_pos = np.array(k) - np.array(MICS_CENTER)
                index += 1
                pose = Pose()
                marker = Marker()
                marker.id = index
                marker.action = Marker.ADD
                marker.type = Marker.SPHERE
                marker.scale.x = 0.3+v
                marker.scale.y = 0.3+v
                marker.scale.z = 0.3+v
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 0.4
                marker.lifetime = rospy.Duration(2)
                marker.header.frame_id = 'world'
                pose.position.x = tf_pos[0]
                pose.position.y = tf_pos[1]
                pose.position.z = tf_pos[2]
                pose.orientation.w = 1.0
                marker.pose = pose
                markers.markers.append(marker)


                # Label marker of SS marker
                # index += 1
                # pose = Pose()
                # marker = Marker()
                # marker.id = index
                # marker.action = Marker.ADD
                # marker.type = Marker.TEXT_VIEW_FACING
                # marker.text = '{}: {}Hz'.format(tf_pos, np.array(avg_srcs_freqDict[k]).round(1))
                # marker.scale.z = 0.2
                # marker.color.r = 1.0
                # marker.color.g = 1.0
                # marker.color.b = 1.0
                # marker.color.a = 1.0
                # marker.lifetime = rospy.Duration(2)
                # marker.header.frame_id = 'world'
                # pose.position.x = tf_pos[0]
                # pose.position.y = tf_pos[1]
                # pose.position.z = tf_pos[2]+1.0
                # pose.orientation.w = 1.0
                # marker.pose = pose
                # markers.markers.append(marker)

            mics_markers = self.get_mics_markers()
            for mic in mics_markers:
                markers.markers.append(mic)

            self.marker_publisher.publish(markers)

        if rospy.get_time()-start_time > TIME_LIMIT:
            rospy.logwarn('Took ' + str(rospy.get_time()-start_time) + ' to process frame array')

    def run(self):
        rospy.spin()


    def average_data(self, data: SoundSample):
        # Sum all data of the same frequencies
        avg_data = []
        for mic in data.frame_arrays:
            freqDict = dict()
            for frame in mic.frames:
                for point in frame.points:
                    if not point.frequency in freqDict.keys():
                        freqDict[point.frequency] = 0
                    freqDict[point.frequency] += point.amplitude
            avg_data.append(freqDict) # there is a freqDict for each mic in avg_data

        # Calculate average dividing the sum by the number of frame_arrays in the sample
        for i, mic in enumerate(avg_data):
            n_frames = len(data.frame_arrays[i].frames)
            for freq in mic.keys():
                avg_data[i][freq] /= n_frames

        return avg_data


    def find_max_amp(self, data):
        max_amp = 0
        mic_max = 9
        for i, mic in enumerate(data):
            for amp in mic.values(): 
                if amp > max_amp:
                    max_amp = amp
                    mic_max = i
        rospy.logerr('mic max amp: {}'.format(mic_max))
        return max_amp


    # def get_important_data(self, data):
    #     '''
    #     It finds which amplitudes are below a threshold and then computes de difference between amplitudes
    #     and applies another threshold to not use sources too far away (small amplitude difference between mics)
    #     '''
    #     # Find important frequencies (the ones whose amplitude difference between mics is significant)
    #     pointsDict = dict()
    #     # min_amp_threshold = self.find_max_amp(data) * MIN_AMP_THRESHOLD
    #     for freq in data[0].keys():
    #         if freq < FREQ_RANGE[0] or freq > FREQ_RANGE[1]:
    #             continue
    #         max_amp = 0
    #         for mic in data: 
    #             if mic[freq] > MIN_AMP_THRESHOLD:
    #                 if mic[freq] > max_amp:
    #                     max_amp = mic[freq]
    #         if max_amp:
    #             out_of_range = 1
    #             for i, j in itertools.combinations( np.arange(len(data)), 2 ):
    #                 # dif = np.abs(data[i][freq] - data[j][freq]) / max_amp #TODO: Does it make a difference to do as below?
    #                 dif = data[i][freq] - data[j][freq]
    #                 if dif > 0:
    #                     dif /= data[i][freq]
    #                 else:
    #                     dif /= -data[j][freq]
    #                 if dif > MIN_AMP_DIF_THRESHOLD:
    #                     freq_mics = []
    #                     for mic in data:
    #                         freq_mics.append(mic[freq])
    #                     pointsDict[freq] = freq_mics
    #                     out_of_range = 0
    #                     break
    #             if out_of_range:
    #                 rospy.logerr('{}Hz coming from a sound source too far away.'.format(freq))
    #     return pointsDict


    def get_important_data(self, data):
        '''
        It only considers frequencies within a chosen range.
        It discards frequencies whose magnitudes are below a certain threshold.
        '''
        pointsDict = dict()
        for freq in data[0].keys():
            if freq < FREQ_RANGE[0] or freq > FREQ_RANGE[1]:
                continue
            for mic in data: 
                if mic[freq] > MIN_AMP_THRESHOLD:
                    freq_mics = []
                    for mic_ in data:
                        freq_mics.append(mic_[freq])
                    pointsDict[freq] = freq_mics
        return pointsDict


    def gather_close_coords(self, srcsDict):
        avg_srcsDict = dict()
        for src in srcsDict.keys():
            if avg_srcsDict.keys() == []:
                avg_srcsDict[src] = [src]
            else:
                min_dist = 99
                for av_src in avg_srcsDict.keys():
                    dist = math.dist((src[0], src[1], src[2]), (av_src[0], av_src[1], av_src[2]))
                    if dist < min_dist:
                        min_dist = dist
                        closest = av_src
                if min_dist < DIST_THRESHOLD:
                    # TODO: See which method is faster
                    # size = len(avg_srcsDict[closest])
                    # average = np.around( (size*np.array(closest) + np.array(src)) / (size+1), 2 )
                    # average = (average[0], average[1])
                    # # average = [sum(x)/len(x) for x in zip(*avg_srcsDict[closest])]
                    # # Rename key in dictionary
                    # avg_srcsDict[average] = avg_srcsDict.pop(closest)
                    # avg_srcsDict[average].append(src)

                    avg_srcsDict[closest].append(src)
                    # average = np.average(avg_srcsDict[closest])
                    average = np.around( [sum(np.array(x))/len(x) for x in zip(*avg_srcsDict[closest])], 2 )
                    average = (average[0], average[1], average[2])
                    avg_srcsDict[average] = avg_srcsDict.pop(closest)  # rename key in dictionary
                else:
                    avg_srcsDict[src] = [src]

        # Add frequencies to dictionary of average sources
        avg_srcs_freqDict = dict()
        errorDict = dict()
        for av_src, srcs in avg_srcsDict.items():
            freqs = []
            max_error = 0
            for src in srcs:
                freqs.extend(srcsDict[src])
                error = math.dist((src[0], src[1], src[2]), (av_src[0], av_src[1], av_src[2]))
                if error > max_error:
                    max_error = error
            avg_srcs_freqDict[av_src] = freqs
            errorDict[av_src] = max_error
        return avg_srcs_freqDict, errorDict


    def find_closest_mic(self, src):
        min_dist = 99
        for mic_index, mic in enumerate(self.tf_mics):
            dist = math.dist((src[0], src[1], src[2]), (mic.x, mic.y, mic.z))
            # print(dist)
            if dist < min_dist:
                min_dist = dist
                closest_mic = mic_index
        return closest_mic


    def get_spectrograms(self, srcsDict, data: SoundSample):
        '''
        For each sound source estimated, gather all data of the frequencies coming from
        the same position (puting other frequencies to zero). The only data that matters
        is the one from the microphone closest to that sound source.
        '''
        spectrograms = SoundSample()
        for src in srcsDict.keys():
            # find the mic closer to this src
            spectrogram = SoundFrameArray()
            mic_index = self.find_closest_mic(src)
            # rospy.logerr('closest mic: {}'.format(mic_index))
            for frame in data.frame_arrays[mic_index].frames:
                spec_frame = SoundFrame()
                for point in frame.points:
                    if point.frequency < FREQ_RANGE[0] or point.frequency > FREQ_RANGE[1]:
                        continue
                    spec_point = SoundPoint()
                    spec_point.frequency = point.frequency
                    if not point.frequency in srcsDict[src]:
                        spec_point.amplitude = 0
                    else:
                        spec_point.amplitude = point.amplitude
                    spec_frame.points.append(spec_point)
                spectrogram.frames.append(spec_frame)
            spectrograms.frame_arrays.append(spectrogram)
        return spectrograms


    def get_spectrograms2(self, srcsDict, data: SoundSample):
        '''
        For each sound source estimated, gather all data of the frequencies coming from
        the same position (puting other frequencies to zero). The only data that matters
        is the one from the microphone closest to that sound source.
        '''
        spectrograms = SoundSample()
        for src in srcsDict.keys():
            # find the mic closer to this src
            spectrogram = SoundFrameArray() # each frame array has FRAME_LENGTH
            mic_index = self.find_closest_mic(src)
            # rospy.logerr('closest mic: {}'.format(mic_index))
            # rospy.loginfo(len(data.frame_arrays))
            for frame_array in data.frame_arrays:
                # rospy.loginfo(len(frame_array.frames))
                # rospy.loginfo(len(frame_array.frames[mic_index].points))
                spec_frame = SoundFrame()
                for point in frame_array.frames[mic_index].points:
                    if point.frequency < FREQ_RANGE[0] or point.frequency > FREQ_RANGE[1]:
                        continue
                    spec_point = SoundPoint()
                    spec_point.frequency = point.frequency
                    if not point.frequency in srcsDict[src]:
                        spec_point.amplitude = 0
                    else:
                        spec_point.amplitude = point.amplitude
                    spec_frame.points.append(spec_point)
                spectrogram.frames.append(spec_frame)
            spectrograms.frame_arrays.append(spectrogram)
        return spectrograms


    def remove_mics_area(self, likelihood):
        x_min = int(self.mics_area['x_min'] * DATA_NUM)
        x_max = int(self.mics_area['x_max'] * DATA_NUM)
        y_min = int(self.mics_area['y_min'] * DATA_NUM)
        y_max = int(self.mics_area['y_max'] * DATA_NUM)
        likelihood[x_min:x_max, y_min:y_max] += ADD_TO_MICS_AREA
        # if IN_3D: likelihood[x_min:x_max, y_min:y_max, z_min:z_max] += ADD_TO_MICS_AREA


    def find_max_likelihood(self, likelihood, z):
        index1 = likelihood.argmin()
        index = np.unravel_index(index1, likelihood.shape)
        x = index[1] / DATA_NUM
        y = index[0] / DATA_NUM
        if IN_3D: z = index[2] / DATA_NUM
        return x, y, z


    def remap_likelihood(self, likelihood):
        '''
        Create a likelihood map from the original one, where every pixel is equal to itself plus the sum of the 8 pixels around it.
        '''
        if IN_3D: remap = np.zeros((DATA_NUM*ROOM_SIZE[1], DATA_NUM*ROOM_SIZE[0], DATA_NUM*ROOM_SIZE[2]))
        else: remap = np.zeros((DATA_NUM*ROOM_SIZE[1], DATA_NUM*ROOM_SIZE[0]))
        for i in range(DATA_NUM*ROOM_SIZE[1]):
            for j in range(DATA_NUM*ROOM_SIZE[0]):
                if i==0 or j==0 or i==DATA_NUM*ROOM_SIZE[1]-1 or j==DATA_NUM*ROOM_SIZE[0]-1:
                    remap[i,j] = 100000
                else:
                    remap[i,j] = np.sum(( likelihood[i-1, j-1], likelihood[i-1, j], likelihood[i-1, j+1], 
                                          likelihood[i,   j-1], likelihood[i,   j], likelihood[i,   j+1],
                                          likelihood[i+1, j-1], likelihood[i+1, j], likelihood[i+1, j+1] ))
        return remap


    def map_estimation(self, E1, E2, d1, d2):
        C = np.abs( E1*d1 - E2*d2 ) # compare energies
        if PLOT_CIRCLES and not IN_3D:
            min = get_ith_min(C)
            C = np.where(C < min, C, 1000) #np.square(np.square(C)) #np.square(C) #np.where(C < min, C, 1000) #0.01
        return C


    def compute_ILD(self, data, freq, count):
        if IN_3D: likelihood = np.zeros((DATA_NUM*ROOM_SIZE[1], DATA_NUM*ROOM_SIZE[0], DATA_NUM*ROOM_SIZE[2]))
        else: likelihood = np.zeros((DATA_NUM*ROOM_SIZE[1], DATA_NUM*ROOM_SIZE[0]))

        # For every combination of mics, estimate the sound source localization (circle or line)
        for i, j in itertools.combinations( np.arange(len(self.mics)), 2 ):
            current_map = self.map_estimation(data[freq][i]**2, data[freq][j]**2, self.distances[i], self.distances[j])
            likelihood += current_map
            if PLOT_MAP and PLOT_CURRENT_MAP and not IN_3D:
                # if freq == 500.0:
                fig, ax = plt.subplots() # fig : figure object, ax : Axes object
                fig.set_figheight(5)
                fig.set_figwidth(6)
                im = ax.imshow(current_map, extent=[0,ROOM_SIZE[0],0,ROOM_SIZE[1]], origin='lower')
                ax.scatter([mic.x for mic in self.tf_mics], [mic.y for mic in self.tf_mics], color='#000000', marker='x')
                ax.set_xlim(0, ROOM_SIZE[0])
                ax.set_ylim(0, ROOM_SIZE[1])
                ax.set_title('Comparing mics ({},{})'.format(i+1, j+1))
                fig.colorbar(im, ax=ax)
                fig.savefig('/home/miguel/likelihood/likelihood_{}_{}_({},{}).png'.format(count, freq, i+1, j+1))
                plt.close(fig)
        self.remove_mics_area(likelihood)

        coord = None
        if MAP_OPTION == 0:
            coord = self.find_max_likelihood(likelihood, self.tf_mics[0].z)
        elif MAP_OPTION == 1:
            sum_map = self.remap_likelihood(likelihood)
            coord = self.find_max_likelihood(sum_map, self.tf_mics[0].z)

        # FIXME: Não faz sentido estar a comparar as posições freq a freq com as frames anteriores, porque muitos sons não mantêm as mesmas
        # frequências. Mais vale confiar na localização estimada, e depois no classificador é que se verifica se isso melhor.
        elif MAP_OPTION == 2:
            sum_map = self.remap_likelihood(likelihood) # TODO in this map_option it is not ncessary to compute the entire sum_map
            coords = []
            coords_sum = []
            for i in range(10):
                c = self.find_max_likelihood(likelihood, self.tf_mics[0].z)
                print(c)
                coord_sum = sum_map[c[1]*DATA_NUM, c[0]*DATA_NUM]
                coords.append(c)
                coords_sum.append(coord_sum)
            # for i in range(10):
            index = np.array(coords_sum).argmin()
            coord = coords[index]

        if PLOT_MAP and not IN_3D:
            # if freq == 500.0:
            fig, ax = plt.subplots() # fig : figure object, ax : Axes object
            fig.set_figheight(5)
            fig.set_figwidth(6)
            im = ax.imshow(likelihood, extent=[0,ROOM_SIZE[0],0,ROOM_SIZE[1]], origin='lower')
            ax.scatter(coord[0], coord[1], color='#ffffff')
            for source in self.sound_sources:
                ax.scatter(source.x+MICS_CENTER[0], source.y+MICS_CENTER[1], color='#000000')
            ax.scatter([mic.x for mic in self.tf_mics], [mic.y for mic in self.tf_mics], color='#000000', marker='x')
            ax.set_xlim(0, ROOM_SIZE[0])
            ax.set_ylim(0, ROOM_SIZE[1])
            ax.set_title('Frame {}   |   {} Hz'.format(count, freq))
            fig.colorbar(im, ax=ax)
            fig.savefig('/home/miguel/likelihood/likelihood_{}_{}.png'.format(count, freq))
            plt.close(fig)
        return coord


    def estimate_poses(self, data, count):
        srcsDict =  dict()
        for freq in data.keys():
            coord = self.compute_ILD(data, freq, count)
            if not coord in srcsDict.keys():
                srcsDict[coord] = []
            srcsDict[coord].append(freq)
        return srcsDict


    def get_mics_area(self):
        '''
        The area between the microphones should be ignored by the locator, since it is
        supposed to detect sound around them. This method creates that boundary and
        surpasses it by a little (MIC_AREA_THRESHOLD), to make sure there are no incorrect estimations
        '''
        mics_area = dict()
        mics_area['x_min'] = ROOM_SIZE[0]
        mics_area['x_max'] = 0
        mics_area['y_min'] = ROOM_SIZE[1]
        mics_area['y_max'] = 0
        # mics_area['z_min'] = ROOM_SIZE[2]
        # mics_area['z_max'] = 0     
        for mic in self.tf_mics:
            if mic.x < mics_area['x_min']: mics_area['x_min'] = mic.x
            if mic.x > mics_area['x_max']: mics_area['x_max'] = mic.x
            if mic.y < mics_area['y_min']: mics_area['y_min'] = mic.y
            if mic.y > mics_area['y_max']: mics_area['y_max'] = mic.y
            # if mic.z < mics_area['z_min']: mics_area['z_min'] = mic.z
            # if mic.z > mics_area['z_max']: mics_area['z_max'] = mic.z
        mics_area['x_min'] -= MIC_AREA_THRESHOLD
        mics_area['x_max'] += MIC_AREA_THRESHOLD
        mics_area['y_min'] -= MIC_AREA_THRESHOLD
        mics_area['y_max'] += MIC_AREA_THRESHOLD
        # mics_area['z_min'] -= MIC_AREA_THRESHOLD
        # mics_area['z_max'] += MIC_AREA_THRESHOLD
        return mics_area


    def get_distance_map(self, mic):
        '''
        Compute the likelihood map, based on possible distances between the microphone and the sound sources,
        and in case the microphones are unidirectional, it also takes into account the directionality of the
        microphone and the possible angles it does with the sound sources.
        '''
        coord_mic = np.array([mic.x, mic.y, mic.z])
        if IN_3D:
            x, y, z = create_map_3D()
            d = (x - mic.x)**2 + (y - mic.y)**2 + (z - mic.z)**2  # squared distance
            if PRA: coord = np.array([x-mic.x, y-mic.y, z-mic.z]) / np.square(d)
            coord_src = np.array([x, y, z])
        else:
            x, y = create_map_2D()
            d = (x - mic.x)**2 + (y - mic.y)**2 # squared distance
            if PRA: coord = np.array([x-mic.x, y-mic.y, z-mic.z]) / np.square(d)
            coord_src = np.array([x, y])

        direct = get_mic_direction([mic.rx, mic.ry, mic.rz, mic.rw]) # direction of mic

        if PRA:
            # coord = []
            # for ax_src, ax_mic in zip(coord_src, coord_mic): # coord = np.array([x-mic.x, y-mic.y, z-mic.z]) / np.square(d)
            #     coord.append( (ax_src - ax_mic) / np.square(d) )
            # coord = np.array(coord)
            coord = np.reshape(coord, (coord.shape[0], np.prod(coord.shape[1:])))  # coord.shape = (2,6400) each column is a map coord
            cardioid = pra.cardioid_func(x=coord, direction=direct, coef=POLAR_PATTERN_COEF, magnitude=True)
            polar_pattern = np.reshape(cardioid, d.shape)
        else:
            polar_pattern = polar_patter_func(coord_mic, coord_src, direct)

        return d / polar_pattern**2
        # return d/(1+cos_theta)**2


    def get_distances(self):
        '''
        Initialize the likelihood map for each microphone.
        '''
        distances = []
        for mic in self.tf_mics:
            distances.append(self.get_distance_map(mic))
        # rospy.loginfo('Likelihood maps computed')
        return distances
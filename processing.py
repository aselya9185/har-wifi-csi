import numpy as np
from classification import classify, feat_selection
import re

def compare_strings(string1, string2):
	pattern = re.compile(string2)
	match = re.search(pattern, string1)
	return match


#CLASS FEATURES
class Feature():
    def __init__(self, activity, window, mean_var, integral, distance):
        self.activity = activity
        self.window = window
        self.mean_var = mean_var
        self.spectrum_integral = integral
        self.distance = distance

features = []
n_obj = 0


## ACTIVITIES
#room 1 + room 2
#activity_files = ['saved_csi/r1_empty.npy','saved_csi/r1_sitting_1.npy','saved_csi/r1_sitting_2.npy','saved_csi/r1_sitting_3.npy','saved_csi/r1_standing_1.npy','saved_csi/r1_standing_2.npy','saved_csi/r1_standing_3.npy','saved_csi/r1_walking_1.npy','saved_csi/r1_walking_3.npy','saved_csi/r1_walking_4.npy','saved_csi/r1_walking_5.npy','saved_csi/r2_empty_1.npy','saved_csi/r2_empty_2.npy','saved_csi/r2_sit_1.npy','saved_csi/r2_sit_2.npy','saved_csi/r2_standing_1.npy','saved_csi/r2_standing_2.npy','saved_csi/r2_walk_1.npy','saved_csi/r2_walk_2.npy']

#room 1 
#activity_files = ['saved_csi/r1_empty.npy','saved_csi/r1_sitting_1.npy','saved_csi/r1_sitting_2.npy','saved_csi/r1_sitting_3.npy','saved_csi/r1_standing_1.npy','saved_csi/r1_standing_2.npy','saved_csi/r1_standing_3.npy','saved_csi/r1_walking_1.npy','saved_csi/r1_walking_3.npy','saved_csi/r1_walking_4.npy','saved_csi/r1_walking_5.npy']

#room 2 
activity_files = ['saved_csi/r2_empty_1.npy','saved_csi/r2_empty_2.npy','saved_csi/r2_sit_1.npy','saved_csi/r2_sit_2.npy','saved_csi/r2_standing_1.npy','saved_csi/r2_standing_2.npy','saved_csi/r2_walk_1.npy','saved_csi/r2_walk_2.npy']


## PARAMETERS DEFINITION

# normalisation: 0 not normalised, 1 mean normalisation
norm = 0

# 0 for db, 1 for linear
scale = 1

# 0 for plot, 1 maximum integral, 2 maximum vector distance, 3 amplitude variance
output = [1,2,3]

#0 for time plot, 1 for quantities
#graph = 1

# Digital beamforming weights
n_weights = 40
deltaTheta = 2*np.pi/n_weights

#dimensioni finestra
slide = 1

#dataset initialization
dataset = []

#PROCESSING REPETEAD FOR EACH ACTIVITY
for activity in range(len(activity_files)):
    print('activity: ', activity)

    #load csi matrices from file
    csi = np.load(activity_files[activity])

    #select target channels
    csi_ch1 = csi[0]
    csi_ch2 = csi[1]

    #csi amplitudes
    csi_ch1_amp = abs(csi_ch1)
    csi_ch2_amp = abs(csi_ch2)

    #n_samples = np.min(np.array([csi_ch1.shape[0],csi_ch2.shape[0]]))
    n_samples = csi_ch1.shape[0]
    print('numero di campioni: ',n_samples)

    #normalization
    if norm == 1:
        mean_ch1 = np.mean(csi_ch1_amp)
        mean_ch2 = np.mean(csi_ch2_amp)

        csi_ch1 = csi_ch1 / mean_ch1
        csi_ch2 = csi_ch2 / mean_ch2


    # activity theta star list
    theta_star_out1 = []
    theta_star_out2 = []
    theta_star_out3 = []

    # activity maximum integral list
    max_int = []

    #activity maximum distance list
    max_distance = []

    #activity minimum variance list
    min_variance = [] 

    ################ Methods Cycles ################
    for method in output:

        print('method: ', method)

        #window parameters initialization
        if method == 1:
            window_size = 8
        if method == 2:
            window_size = 16
        if method == 3:
            window_size = 1
        start_id = 1
        stop_id = start_id + window_size
        n_window = 0

        # Processing of windowed csi
        while stop_id <= n_samples:

            #window indices definition
            window_idx = np.arange(start_id,stop_id)

            #window sliding
            start_id = start_id + slide
            stop_id = start_id + window_size
            n_window = n_window + 1
            
            #winodow feature object creation
            features.append(Feature(activity, n_window, np.zeros((n_weights,2)), np.zeros((n_weights,2)), np.zeros((n_weights,2))))

            #windowed csi
            w_csi_ch1 = csi_ch1[window_idx,:]
            w_csi_ch2 = csi_ch2[window_idx,:]

            #combination parameters initialisation
            comb_count = 0
            theta1 = 0
            
            for i in np.arange(0,n_weights):
                theta1 = theta1 + deltaTheta

                #digital beamforming combination
                csi_comb = w_csi_ch1 + w_csi_ch2*np.exp(theta1*1j)

                #amplitude computation
                amp_csi_comb = abs(csi_comb)

                #phase modification in order to represent it between -pi and pi
                theta_save = theta1
                if theta_save > np.pi:
                    theta_save = theta_save - (2 * np.pi)

                #Doppler Spectrum computation
                if scale == 1:
                    DopplerSpec_comb = np.mean(abs(np.fft.fft(amp_csi_comb, axis=0)),axis=1)
                if scale == 0:
                    DopplerSpec_comb = np.log10(np.mean(abs(np.fft.fft(amp_csi_comb, axis=1)),axis=0))


                ####### OUTPUT COMPUTATIONS #######

                # Doppler spectrum integrals computation + associated weights
                if method == 1:
                    f_int = sum(DopplerSpec_comb)
                    features[n_obj].spectrum_integral[comb_count,:] = np.array([f_int, theta_save])
                    comb_count = comb_count + 1

                # Euler distances computation + associated weights
                if method == 2:
                    distance = []
                    for sample in range(csi_comb.shape[0]-1):
                        vector1 = amp_csi_comb[sample,:]
                        for next in range(1,csi_comb.shape[0]):
                            vector2 = amp_csi_comb[next,:]
                            distance.append(np.sqrt(np.sum((vector2-vector1)**2)))
                    features[n_obj].distance[comb_count,:] = np.array([np.mean(distance), theta_save])
                    comb_count = comb_count + 1

                # variances computation + associated weights
                if method == 3:
                    variance = np.var(amp_csi_comb,axis=1)
                    mean_variance = np.mean(variance)
                    features[n_obj].mean_var[comb_count,:] = np.array([mean_variance, theta_save])
                    comb_count = comb_count + 1
            
            ####### saving results #######
            
            # maximum integral selected with correspondent weight
            if method == 1:
                theta_star_index = np.argmax(features[n_obj].spectrum_integral[:,0])
                theta_star_out1.append(features[n_obj].spectrum_integral[theta_star_index,1])
                max_int.append(np.max(features[n_obj].spectrum_integral[:,0]))

            # maximum distance selected with correspondent weight
            if method == 2:
                theta_star_index = np.argmax(features[n_obj].distance[:,0])
                theta_star_out2.append(features[n_obj].distance[theta_star_index,1])
                max_distance.append(np.max(features[n_obj].distance[:,0]))
            
            # minimum variance selected with correspondent weight
            if method == 3:
                theta_star_index = np.argmin(features[n_obj].mean_var[:,0])
                theta_star_out3.append(features[n_obj].mean_var[theta_star_index,1])
                min_variance.append(np.min(features[n_obj].mean_var[:,0]))

            #object number update
            n_obj = n_obj + 1

    
    ####### METRIC WINDOWING and FEATURES COMPUTATION #######

    #from lists to numpy arrays
    theta_star_out1 = np.array(theta_star_out1)
    theta_star_out2 = np.array(theta_star_out2)
    theta_star_out3 = np.array(theta_star_out3)
    max_int = np.array(max_int)
    max_distance = np.array(max_distance)
    min_variance = np.array(min_variance)

    # metrics window initialization
    window_size_2 = 10
    start_id_2 = 0
    stop_id_2 = start_id_2 + window_size_2
    n_window_2 = 0

    # feature vectors initialization
    feature1 = []
    feature2 = []
    feature3 = []
    feature4 = []
    feature5 = []
    feature6 = []

    # features computation
    while stop_id_2 <= len(theta_star_out2):
        
        #window indices definition
        window_idx_2 = np.arange(start_id_2,stop_id_2)

        #feature vectors update
        feature1.append(np.var(theta_star_out1[window_idx_2]))
        feature2.append(np.mean(max_int[window_idx_2]))
        feature3.append(np.var(theta_star_out2[window_idx_2]))
        feature4.append(np.mean(max_distance[window_idx_2]))
        feature5.append(np.var(theta_star_out3[window_idx_2]))
        feature6.append(np.mean(min_variance[window_idx_2]))

        #window sliding
        start_id_2 = start_id_2 + 1
        stop_id_2 = start_id_2 + window_size_2
        n_window_2 = n_window_2 + 1

    #from lists to numpy array
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    feature3 = np.array(feature3)
    feature4 = np.array(feature4)
    feature5 = np.array(feature5)
    feature6 = np.array(feature6)


    ####### DATASET CREATION #######

    #class definition
    if compare_strings(activity_files[activity],'empty'):
        activity_class = 0
    if compare_strings(activity_files[activity],'sit'):
        activity_class = 1
    if compare_strings(activity_files[activity],'standing'):
        activity_class = 2
    if compare_strings(activity_files[activity],'walk'):
        activity_class = 3

    # class vector creation
    class_vector = np.full(feature1.shape, activity_class)
    class_vector = class_vector[:, np.newaxis]

    # feature vector from row to column
    feature1 = feature1[:, np.newaxis]
    feature2 = feature2[:, np.newaxis]
    feature3 = feature3[:, np.newaxis]
    feature4 = feature4[:, np.newaxis]
    feature5 = feature5[:, np.newaxis]
    feature6 = feature6[:, np.newaxis]
    
    # dataset update
    dataset.append(np.concatenate((feature1, feature2, feature3, feature4, feature5, feature6, class_vector),axis=1))

#from list to numpy matrix
dataset = np.concatenate((dataset))

#save the dataset
np.save('saved_dataset/dataset_r2_w2_10.npy', dataset)

#load the dataset
#loaded_dataset = np.load('saved_dataset/dataset_r1r2_w2_10.npy')


####### CLASSIFICATION #######
#classify(dataset)
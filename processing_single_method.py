import matplotlib.pyplot as plt
import numpy as np
from classification import classify
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
#activity_files = ['saved_csi/r2_empty_1.npy','saved_csi/r2_empty_2.npy','saved_csi/r2_sit_1.npy','saved_csi/r2_sit_2.npy','saved_csi/r2_standing_1.npy','saved_csi/r2_standing_2.npy','saved_csi/r2_walk_1.npy','saved_csi/r2_walk_2.npy']

#SELECTED ACTIVITIES
activity_files = ['saved_csi/r1_empty.npy','saved_csi/r1_sitting_3.npy','saved_csi/r1_standing_1.npy','saved_csi/r1_walking_1.npy']


#PARAMETERS DEFINITION

# normalisation: 0 not normalised, 1 mean normalisation
norm = 0

# 0 for db, 1 for linear
scale = 1

# 0 for plot, 1 maximum integral, 2 maximum vector distance, 3 amplitude variance
output = 2

#0 for time plot, 1 for quantities
graph = 0

# Digital beamforming weights
#n_weights = 20
n_weights = 40
deltaTheta = 2*np.pi/n_weights


#dimensioni finestra
slide = 1
window_size = 16


#PLOT PARAMETERS
#theta star list of lists
theta_stars = []
#maximum integrals list of lists, in case of output == 1
max_ints = []
#maximum distances list of lists, in case of output == 2
max_dists = []
#minimum variances list of lists, in case of output == 3
min_vars = []
#subplots definition
fig, axs = plt.subplots(2,2)
#axis parameters
ranges = [0,0,0,0]
x_up_lims = [0,0,0,0]
x_low_lims = [0,0,0,0]

#dataset
dataset = []
class_labels = []

#PROCESSING REPETEAD FOR EACH ACTIVITY
for activity in range(len(activity_files)):
    print(activity)

    #load csi matrices from file
    csi = np.load(activity_files[activity])

    #select target channels
    csi_ch1 = csi[0]
    csi_ch2 = csi[1]

    csi_ch1_amp = abs(csi_ch1)
    csi_ch2_amp = abs(csi_ch2)

    n_samples = np.min(np.array([csi_ch1.shape[0],csi_ch2.shape[0]]))
    print(n_samples)

    #normalisation
    if norm == 1:
        mean_ch1 = np.mean(csi_ch1_amp)
        mean_ch2 = np.mean(csi_ch2_amp)
    
        csi_ch1 = csi_ch1 / mean_ch1
        csi_ch2 = csi_ch2 / mean_ch2

    start_id = 1
    stop_id = start_id + window_size
    n_window = 0

    # activity theta star list
    theta_star = []

    # activity maximum integral list
    max_int = []

    #activity maximum distance list
    max_distance = []

    #activity minimum variance list
    min_variance = [] 

    while stop_id <= n_samples:
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

            amp_csi_comb = abs(csi_comb)

            #phase modification in order to represent it between -pi and pi
            theta_save = theta1
            # if theta_save > np.pi:
            #     theta_save = theta_save - (2 * np.pi)

            #Doppler Spectrum computation
            if scale == 1:
                DopplerSpec_comb = np.mean(abs(np.fft.fft(amp_csi_comb, axis=0)),axis=1)
            if scale == 0:
                DopplerSpec_comb = np.log10(np.mean(abs(np.fft.fft(amp_csi_comb, axis=1)),axis=0))

            ####### OUTPUT COMPUTATIONS

            if output == 1:
                f_int = sum(DopplerSpec_comb)
                features[n_obj].spectrum_integral[comb_count,:] = np.array([f_int, theta_save])
                comb_count = comb_count + 1
            
            if output == 2:
                distance = []
                for sample in range(csi_comb.shape[0]-1):
                    vector1 = amp_csi_comb[sample,:]
                    for next in range(1,csi_comb.shape[0]):
                        vector2 = amp_csi_comb[next,:]
                        distance.append(np.sqrt(np.sum((vector2-vector1)**2)))
                features[n_obj].distance[comb_count,:] = np.array([np.mean(distance), theta_save])
                comb_count = comb_count + 1

            if output == 3:
                variance = np.var(amp_csi_comb,axis=1)
                mean_variance = np.mean(variance)
                features[n_obj].mean_var[comb_count,:] = np.array([mean_variance, theta_save])
                comb_count = comb_count + 1
        
        ############ saving results
           
        if output == 1:
            theta_star_index = np.argmax(features[n_obj].spectrum_integral[:,0])
            theta_star.append(features[n_obj].spectrum_integral[theta_star_index,1])
            max_int.append(np.max(features[n_obj].spectrum_integral[:,0]))

        if output == 2:
            theta_star_index = np.argmax(features[n_obj].distance[:,0])
            theta_star.append(features[n_obj].distance[theta_star_index,1])
            max_distance.append(np.max(features[n_obj].distance[:,0]))
        
        if output == 3:
            theta_star_index = np.argmin(features[n_obj].mean_var[:,0])
            theta_star.append(features[n_obj].mean_var[theta_star_index,1])
            min_variance.append(np.min(features[n_obj].mean_var[:,0]))

        n_obj = n_obj + 1

    
    ######## METRIC WINDOWING

    theta_star = np.array(theta_star)
    #theta_star[theta_star > np.pi] = theta_star[theta_star > np.pi] - (2 * np.pi)

    if output == 1:
        metric = np.array(max_int)

    if output == 2:
        metric = np.array(max_distance)
 
    if output == 3:
        metric = np.array(min_variance)
        

    window_size_2 = 100
    start_id_2 = 0
    stop_id_2 = start_id_2 + window_size_2
    n_window_2 = 0
    feature1 = []
    feature2 = []

    while stop_id_2 <= len(metric):
        
        window_idx_2 = np.arange(start_id_2,stop_id_2)
        feature1.append(np.var(theta_star[window_idx_2]))
        feature2.append(np.mean(metric[window_idx_2]))

        #window sliding
        start_id_2 = start_id_2 + 1
        stop_id_2 = start_id_2 + window_size_2
        n_window_2 = n_window_2 + 1

    feature1 = np.array(feature1)
    feature2 = np.array(feature2)


    ############################# DATASET CREATION #############################


    if compare_strings(activity_files[activity],'empty'):
        activity_class = 0
        activity_class_label = 'Empty'
    if compare_strings(activity_files[activity],'sit'):
        activity_class = 1
        activity_class_label = 'Sitting'
    if compare_strings(activity_files[activity],'standing'):
        activity_class = 2
        activity_class_label = 'Standing'
    if compare_strings(activity_files[activity],'walk'):
        activity_class = 3
        activity_class_label = 'Walking'

    class_vector = np.full(feature1.shape, activity_class)
    class_vector = class_vector[:, np.newaxis]
    feature1 = feature1[:, np.newaxis]
    feature2 = feature2[:, np.newaxis]

    class_labels.append(activity_class_label)
    dataset.append(np.concatenate((feature1, feature2, class_vector),axis=1))



    #PLOT PARAMETERS DEFINITION

    # list of lists, one list for each activity
    #convertion from list to numpy array
    theta_stars.append(np.array(theta_star))

    if output == 1:
        max_ints.append(np.array(max_int))
    if  output == 2:
        max_dists.append(np.array(max_distance))
    if  output == 3:
        min_vars.append(np.array(min_variance))


    # axis definition
    if graph == 0:
        ranges[activity] = np.arange(1,n_window+1)
        x_up_lims[activity] = n_window+1
    if graph == 1:
        if output == 1:
            x_up_lims[activity] = max(max_int)
            x_low_lims[activity] = min(max_int)
        if output == 2:
            x_up_lims[activity] = max(max_distance)
            x_low_lims[activity] = min(max_distance)
        if output == 3:
            x_up_lims[activity] = max(min_variance)
            x_low_lims[activity] = min(min_variance)
        

#axis limits
if graph == 0:
    x_lim = [0,max(x_up_lims)]
if graph == 1:
    x_lim = [min(x_low_lims),max(x_up_lims)]

#PLOT

#first activity
if output == 1:
    if graph==0:
        axs[0,0].scatter(ranges[0], theta_stars[0])
        axs[0,0].set_xlabel('time')
    if graph==1:
        axs[0,0].scatter(max_ints[0], theta_stars[0])
        axs[0,0].set_xlabel('maximum integral of Doppler spectrum')
    axs[0,0].set_ylabel('theta_star')
if output == 2:
    if graph==0:
        axs[0,0].scatter(ranges[0], theta_stars[0])
        axs[0,0].set_xlabel('time')
    if graph==1:
        axs[0,0].scatter(max_dists[0],theta_stars[0])
        axs[0,0].set_xlabel('maximum vectors distance')
    axs[0,0].set_ylabel('theta_star')
if output == 3:
    if graph==0:
        axs[0,0].scatter(ranges[0], theta_stars[0])
        axs[0,0].set_xlabel('time')
    if graph==1:
        axs[0,0].scatter(min_vars[0],theta_stars[0])
        axs[0,0].set_xlabel('minimum variance')
    axs[0,0].set_ylabel('theta_star')
axs[0,0].set_title(class_labels[0])
#axs[0,0].set_ylim([-4,4])
axs[0,0].set_ylim([0,6])
axs[0,0].set_xlim(x_lim)

#second activity
if output == 1:
    if graph==0:
        axs[0,1].scatter(ranges[1], theta_stars[1])
        axs[0,1].set_xlabel('time')
    if graph==1:
        axs[0,1].scatter(max_ints[1],theta_stars[1])
        axs[0,1].set_xlabel('maximum integral of DS')
    axs[0,1].set_ylabel('theta_star')
    
if output == 2:
    if graph==0:
        axs[0,1].scatter(ranges[1], theta_stars[1])
        axs[0,1].set_xlabel('time')
    if graph==1:
        axs[0,1].scatter(max_dists[1],theta_stars[1])
        axs[0,1].set_xlabel('maximum vectors distance')
    axs[0,1].set_ylabel('theta_star')
    
if output == 3:
    if graph==0:
        axs[0,1].scatter(ranges[1], theta_stars[1])
        axs[0,1].set_xlabel('time')
    if graph==1:
        axs[0,1].scatter(min_vars[1],theta_stars[1])
        axs[0,1].set_xlabel('minimum variance')
    axs[0,1].set_ylabel('theta_star')
axs[0,1].set_title(class_labels[1])
#axs[0,1].set_ylim([-4,4])
axs[0,1].set_ylim([0,6])
axs[0,1].set_xlim(x_lim)


#third activity
if output == 1:
    if graph==0:
        axs[1,0].scatter(ranges[2], theta_stars[2])
        axs[1,0].set_xlabel('time')
    if graph==1:
        axs[1,0].scatter(max_ints[2],theta_stars[2])
        axs[1,0].set_xlabel('maximum integral of DS')
    axs[1,0].set_ylabel('theta_star')
if output == 2:
    if graph==0:
        axs[1,0].scatter(ranges[2], theta_stars[2])
        axs[1,0].set_xlabel('time')
    if graph==1:
        axs[1,0].scatter(max_dists[2],theta_stars[2])
        axs[1,0].set_xlabel('maximum vectors distance')
    axs[1,0].set_ylabel('theta_star')
if output == 3:
    if graph==0:
        axs[1,0].scatter(ranges[2], theta_stars[2])
        axs[1,0].set_xlabel('time')
    if graph==1:
        axs[1,0].scatter(min_vars[2],theta_stars[2])
        axs[1,0].set_xlabel('minimum variance')
    axs[1,0].set_ylabel('theta_star')
axs[1,0].set_title(class_labels[2])
#axs[1,0].set_ylim([-4,4])
axs[1,0].set_ylim([0,6])
axs[1,0].set_xlim(x_lim)

#forth activity
if output == 1:
    if graph==0:
        axs[1,1].scatter(ranges[3], theta_stars[3])
        axs[1,1].set_xlabel('time')
    if graph==1:
        axs[1,1].scatter(max_ints[3],theta_stars[3])
        axs[1,1].set_xlabel('maximum integral of DS')
    axs[1,1].set_ylabel('theta_star')
if output == 2:
    if graph==0:
        axs[1,1].scatter(ranges[3], theta_stars[3])
        axs[1,1].set_xlabel('time')
    if graph==1:
        axs[1,1].scatter(max_dists[3],theta_stars[3])
        axs[1,1].set_xlabel('maximum vectors distance')
    axs[1,1].set_ylabel('theta_star')
if output == 3:
    if graph==0:
        axs[1,1].scatter(ranges[3], theta_stars[3])
        axs[1,1].set_xlabel('time')
    if graph==1:
        axs[1,1].scatter(min_vars[3],theta_stars[3])
        axs[1,1].set_xlabel('minimum variance')
    axs[1,1].set_ylabel('theta_star')   
axs[1,1].set_title(class_labels[3])
#axs[1,1].set_ylim([-4,4])
axs[1,1].set_ylim([0,6])
axs[1,1].set_xlim(x_lim)

# #fifth activity
# if output == 1:
#     if graph==0:
#         axs[2,0].scatter(ranges[4], theta_stars[4])
#         axs[2,0].set_xlabel('time')
#     if graph==1:
#         axs[2,0].scatter(max_ints[4],theta_stars[4])
#         axs[2,0].set_xlabel('maximum integral of DS')
#     axs[2,0].set_ylabel('theta_star')
# if output == 2:
#     if graph==0:
#         axs[2,0].scatter(ranges[4], theta_stars[4])
#         axs[2,0].set_xlabel('time')
#     if graph==1:
#         axs[2,0].scatter(max_dists[4],theta_stars[4])
#         axs[2,0].set_xlabel('maximum vectors distance')
#     axs[2,0].set_ylabel('theta_star')
# if output == 3:
#     if graph==0:
#         axs[2,0].scatter(ranges[4], theta_stars[4])
#         axs[2,0].set_xlabel('time')
#     if graph==1:
#         axs[2,0].scatter(min_vars[4],theta_stars[4])
#         axs[2,0].set_xlabel('minimum variance')
#     axs[2,0].set_ylabel('theta_star')
# axs[2,0].set_title(class_labels[4])
# axs[2,0].set_ylim([-4,4])
# #axs[2,0].set_ylim([0,6])
# axs[2,0].set_xlim(x_lim)

# #sixth activity 
# if output == 1:
#     if graph==0:
#         axs[2,1].scatter(ranges[5], theta_stars[5])
#         axs[2,1].set_xlabel('time')
#     if graph==1:
#         axs[2,1].scatter(max_ints[5],theta_stars[5])
#         axs[2,1].set_xlabel('maximum integral of DS')
#     axs[2,1].set_ylabel('theta_star')
# if output == 2:
#     if graph==0:
#         axs[2,1].scatter(ranges[5], theta_stars[5])
#         axs[2,1].set_xlabel('time')
#     if graph==1:
#         axs[2,1].scatter(max_dists[5],theta_stars[5])
#         axs[2,1].set_ylabel('maximum vectors distance')
#     axs[2,1].set_xlabel('theta_star')
# if output == 3:
#     if graph==0:
#         axs[2,1].scatter(ranges[5], theta_stars[5])
#         axs[2,1].set_xlabel('time')
#     if graph==1:
#         axs[2,1].scatter(min_vars[5],theta_stars[5])
#         axs[2,1].set_xlabel('minimum variance')
#     axs[2,1].set_ylabel('theta_star')
# axs[2,1].set_title(class_labels[5])
# axs[2,1].set_ylim([-4,4])
# #axs[2,1].set_ylim([0,6])
# axs[2,1].set_xlim(x_lim)


plt.subplots_adjust(left=0.125, bottom=0.08, right=0.9, top=0.942, wspace=0.2, hspace=0.405)

plt.show()



                
########## CLASSIFICATION ##########

# dataset = np.concatenate((dataset))
# classify(dataset)
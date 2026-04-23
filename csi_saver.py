import numpy as np
from csi_decoder import csi_extract

csi = csi_extract('activities_pcap_files/r1_empty.pcap')

# print(csi)

# undersampled_csi = csi
# #[::5] #r2 stand 2
# #[::3] #r2 stand 1
# #[::2] #r1 sitting
# undersampled_csi[0] = undersampled_csi[0][::5]
# undersampled_csi[1] = undersampled_csi[1][::5]
# undersampled_csi[2] = undersampled_csi[2][::5]
# undersampled_csi[3] = undersampled_csi[3][::5]

#save csi
np.save('saved_csi_2/r1_empty.npy', csi)
#np.save('saved_csi/r2_standing_2.npy', undersampled_csi)

#load saved csi
loaded_csi = np.load('dataset/saved_csi_raw/r1_empty.npy')

#print('extracted csi:\n',csi)
print('loaded csi:\n',loaded_csi)
print(loaded_csi.shape)

from nexcsi import decoder
import numpy as np
import plot_csi


def csi_extract(pcap_file):

    device = "rtac86u" # nexus5, nexus6p, rtac86u

    # Data retrieval
    samples = decoder(device).read_pcap(pcap_file)

    # samples is a Numpy Structured Array
    #print(samples.dtype)
    #
    # [
    #     ('ts_sec', '<u4'), ('ts_usec', '<u4'), ('saddr', '>u4'), 
    #     ('daddr', '>u4'), ('sport', '>u2'), ('dport', '>u2'),
    #     ('magic', '<u2'), ('rssi', 'i1'), ('fctl', 'u1'),
    #     ('mac', 'u1', (6,)), ('seq', '<u2'), ('css', '<u2'),
    #     ('csp', '<u2'), ('cvr', '<u2'), ('csi', '<i2', (512,))
    # ]
    #

    # #################### Timestamp ####################

    # t0 = samples['ts_sec'][0]+samples['ts_usec'][0]*1e-6

    # timestamp = samples['ts_sec']+samples['ts_usec']*1e-6 - t0

    # #print deltaT
    # deltaT = np.zeros(timestamp.shape)
    # for i in range(len(samples)-1):
    #     deltaT[i] = timestamp[i+1]-timestamp[i]

    #################### Accessing CSI as type complex64 ####################

    ## Null and Pilots subcarriers elimination
    csi = decoder(device).unpack(samples['csi'], zero_nulls=True, zero_pilots=True)
    #csi = np.delete(csi, csi.dtype.metadata['nulls'], axis=1)
    #csi = np.delete(csi, csi.dtype.metadata['pilots'], axis=1)
    csi = np.delete(csi, [0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231, 251, 252, 253, 254, 255], axis=1)

   

    #################### CSI packets assignation to each antenna ####################
    #Antenna monitoring: -C 1 -> 0, -C 2 -> 256, -C 4 -> 512 , -C 8 -> 768
    antenna_idx = np.unique(samples['css'], return_counts=False)
    antenna_csi = []
    for a in range(len(antenna_idx)):
        ant_indices = np.where(samples['css'] == antenna_idx[a])[0]
        antenna_csi.append(csi[ant_indices])


    ## CSI PLOT
    #plot_csi.plot_csi(antenna_idx, antenna_csi, timestamp)

    #return the list of csi matrices
    return antenna_csi


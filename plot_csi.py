import matplotlib.pyplot as plt
import numpy as np
import time

def plot_csi(antenna_idx, antenna_csi, t):

    #upper csi limit computation
    ylim = []
    for idx in range(len(antenna_idx)):  
        ylim.append(np.abs(antenna_csi[idx]).max())
    
    ylim = np.array(ylim)
    ylim_op = ylim.max()


    if len(antenna_idx)>1:

        lines = []
        plt.ion()
        fig, axs = plt.subplots(len(antenna_csi))        

        for idx, antenna in enumerate(antenna_csi):
            line, = axs[idx].plot(np.abs(antenna[0]))
            lines.append(line)

        for i in range(0, len(antenna_csi[0])):

            for idx, antenna in enumerate(antenna_csi):
                lines[idx].set_ydata(np.abs(antenna[i]))
                axs[idx].set_ylim([0, ylim_op])
                axs[idx].title.set_text('Antenna '+str(antenna_idx[idx]))

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.5)

    else:
        plt.ion()
        fig, ax = plt.subplots(1)
        line1, = ax.plot(np.abs(antenna_csi[0])[0])
        
        for i in range(0, len(antenna_csi[0])):

            line1.set_ydata(np.abs(antenna_csi[0])[i])
            ax.set_ylim([0, ylim])
            ax.set_title('Time:'+str(t[i])+', Antenna:'+str(antenna_idx[0]))
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(2)

    plt.subplots_adjust(left=0.125, bottom=0.071, right=0.9, top=0.94, wspace=0.2, hspace=0.362)
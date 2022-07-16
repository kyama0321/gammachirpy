# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time

import utils
import gcfb_v211 as gcfb


class GCparamDefault:
    fs = 48000 # samping frequency
    num_ch = 100 # number of channels
    f_range = np.array([100, 6000]) # range of center frequency
    out_mid_crct = 'No' # outer & middle ear correlations
    # out_mid_crct = 'ELC' # equal loudness contour (ELC)
    # ctrl = "dynamic" # used to be 'tyme-varying'
    # ctrl = "static" # or 'fixed'


def main():
    # stiuli : a simple pulse train
    fs = 48000
    t_pulse = 10 # (ms) 100 Hz pulse train
    snd = np.array(([1]+[0]*int(t_pulse*fs/1000-1))*10)
    t_snd = len(snd)/fs
    t = np.arange(0, len(snd), 1)/fs
    print(f"Duration of sound = {t_snd*1000} (ms)")

    # signal levels
    list_dbspl = np.arange(40, 100, 20)
    
    for sw_ctrl in range(2): # 1: only dynamic, 2: dynamic and static
        fig = plt.subplots()

        for sw_dbspl in range(len(list_dbspl)): # each dbspl
            # calibrate the signal level
            dbspl = list_dbspl[sw_dbspl]
            snd_eq, _ = utils.eqlz2meddis_hc_level(snd, dbspl)

            # set paramteres for dcGC
            gc_param = GCparamDefault() # reset all
            if sw_ctrl == 0: 
                ctrl = "dynamic"
            else: 
                ctrl = "static"
            gc_param.ctrl = ctrl
            
            # dcGC processing
            t_start = time.time()
            cgc_out, _, _, _ = gcfb.gcfb_v211(snd_eq, gc_param)
            t_end = time.time()
            print(f"Elapsed time is {np.round(t_end-t_start, 4)} (sec) = " \
                  + f"{np.round((t_end-t_start)/t_snd, 4)} times RealTime.")
            
            # plot
            ax = plt.subplot(len(list_dbspl), 1, sw_dbspl+1)
            plt.imshow(np.maximum(cgc_out, 0), extent=[min(t), max(t), 0, 100], \
                       aspect='auto', origin='lower', cmap='jet')
            ax.set_title(f"GCFB control = {ctrl}; Signal Level = {dbspl} dB SPL")
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_ylabel("channel")
            plt.tight_layout()
            plt.pause(0.05)

        ax.set_xlabel("time (s)")
        plt.tight_layout()
        plt.pause(0.05)
    
    plt.show()


if __name__ == '__main__':
    main()
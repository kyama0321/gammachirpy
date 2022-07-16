# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import utils
import gcfb_v211 as gcfb


class GCparamDefault:
    fs = 48000 # samping frequency
    num_ch = 100 # number of channels
    f_range = np.array([100, 6000]) # range of center frequency
    # out_mid_crct = 'No' # outer & middle ear correlations
    out_mid_crct = 'ELC' # equal loudness contour (ELC)
    # ctrl = "dynamic" # used to be 'tyme-varying'
    # ctrl = "static" # or 'fixed'


def main():
    # get directory path of this file
    path_dir = os.path.dirname(os.path.abspath(__file__))

    # stiuli : a sample speech
    file_name = '/sample/snd_gammachirpy.wav'
    path_file = path_dir + file_name

    # read sample speech
    snd, fs = utils.audioread(path_file)
    t_snd = len(snd)/fs
    t = np.arange(0, len(snd), 1)/fs
    print(f"Duration of sound = {t_snd*1000} (ms)")

    # signal levels (dB)
    list_dbspl = [40, 60, 80]

    max_aud_spec = np.zeros([2, len(list_dbspl)])    
    for sw_ctrl in range(2): # 1: only dynamic, 2: dynamic and static
        fig = plt.subplots()

        for sw_dbspl in range(len(list_dbspl)): # each dbspl
            # calibrate the signal level
            dbspl = list_dbspl[sw_dbspl]
            snd_eq, _ = utils.eqlz2meddis_hc_level(snd, dbspl)

            # set paramteres for dcGC
            gc_param = GCparamDefault() # reset all
            if sw_ctrl == 0: 
                ctrl = "static"
            else: 
                ctrl = "dynamic"
            gc_param.ctrl = ctrl
            
            # dcGC processing
            t_start = time.time()
            cgc_out, _, _, _ = gcfb.gcfb_v211(snd_eq, gc_param)
            t_end = time.time()
            print(f"Elapsed time is {np.round(t_end-t_start, 4)} (sec) = " \
                  + f"{np.round((t_end-t_start)/t_snd, 4)} times RealTime.")
            
            gcfb_param = GCparamDefault()
            gcfb_param.fs = fs # using default. See inside cal_smooth_spec for parameters
            aud_spec, _ = gcfb.cal_smooth_spech(np.maximum(cgc_out, 0), gcfb_param)

            # plot
            ax = plt.subplot(len(list_dbspl), 1, sw_dbspl+1)
            max_aud_spec[sw_ctrl, sw_dbspl] = np.max(np.max(aud_spec))
            print(max_aud_spec)

            if sw_ctrl == 1:
                # Normalized by max value. 
                # It is a data specific value. 
                # Please change it.
                amp_img = (64*1.2)/49
            else:
                amp_img = (64*1.2)/166

            plt.imshow(amp_img*aud_spec, extent=[min(t), max(t), 1, 100], \
                       aspect='auto', origin='lower', cmap='jet')
            ax.set_title(f"GCFB control = {ctrl}; Signal Level = {dbspl} dB SPL")
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            plt.tight_layout()
            plt.pause(0.05)

    plt.show()


if __name__ == '__main__':
    main()
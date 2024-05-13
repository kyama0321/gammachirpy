# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

import utils
import gcfb_v234 as gcfb

class GCparamDefault:
    # --- basic paramters
    fs = 48000 # samping frequency
    num_ch = 100 # number of channels
    f_range = np.array([100, 6000]) # range of center frequency
    field2cochlea = [] # outer- and mid-ear compensation
    
    # --- outer & middle ear correlations
    out_mid_crct = 'No'
    # out_mid_crct = 'ELC' # equal loudness contour (ELC)
    # out_mid_crct = 'FreeField'
    # out_mid_crct = 'DiffuseField'
    # out_mid_crct = 'EarDrum'

    # --- time-varying setting
    ctrl = "dynamic" # used to be 'time-varying'

    # --- frame-base or sample-base processing
    dyn_hpaf_str_prc = 'frame-base'

    # --- hearing-loss patterns and compression health \alpha
    hloss_type = 'NH' # normal hearing listeners

    # hloss_type = 'HL3'
    # hloss_compression_health = 0.5

    # hloss_type = 'HL0' # manual settings
    # hloss_hearing_level_db = np.array([5, 5, 6, 7, 12, 28, 39]) + 5 # HL4+5dB
    # hloss_compression_health = 0.5

class EMparamDefault:
    # --- basic parameters
    reduce_db = 5 # reduction of TMTF
    f_cutoff = 128 # cutoff frequency of LPF in envelope domation
    fc_mod_list = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]) # modulation filterbank


def main():

    # get directory path of this file
    path_dir = os.path.dirname(os.path.abspath(__file__))

    # stiuli : a simple pulse train
    fs = 48000
    t_pulse = 10 # (ms) 100 Hz pulse train
    snd = np.array(([1]+[0]*int(t_pulse*fs/1000-1))*10)
    t_snd = len(snd)/fs
    t = np.arange(0, len(snd), 1)/fs
    print(f"Duration of sound = {t_snd*1000} (ms)")

    snd_src = snd

    # sound pressure level of the input signal (dB)
    list_dbspl = np.arange(40, 100, 20)
    # list_dbspl = [60]

    # time-varying setting of dcGC-FB
    # list_ctrl = ['dynamic', 'static']
    list_ctrl = ['dynamic']

    # calibration method
    # sw_eqlz_mds = 1 # conventional
    sw_eqlz_mds = 2 # new feature

    for ctrl in list_ctrl: # each time-varying setting

        for sw_dbspl, dbspl in enumerate(list_dbspl): # each dbspl

            # calibrate the signal level
            if sw_eqlz_mds == 1:
                # Conventional method  ~v231
                # You do not need to calibrate the sound level in advance.
                # You'd better use the other one if you know the correspondence between the digital level and SPL.
                snd_eq_m, _ = utils.eqlz2meddis_hc_level(snd, dbspl)
            else:
                # Alternative method v233~
                # You can set snd_eq_m precisely if you know SPL (dB) when rms(digital_s(t)) == 1 (i.e., DigitalRms1SPLdB).
                digital_rms1_dbspl = 90
                snd_digital_level_db = dbspl - digital_rms1_dbspl
                snd_src1 = 10**(snd_digital_level_db/20)/utils.rms(snd) * snd_src
                snd_eq_m, _ = utils.eqlz2meddis_hc_level(snd_src1, [], digital_rms1_dbspl)

            """
            dcGC processing
            """
            # NH: normal hearing listener
            gc_param = GCparamDefault() # reset
            gc_param.hloss_type = 'NH'
            dcgc_out_nh, _, gc_param_nh, gc_resp_nh = gcfb.gcfb_v234(snd_eq_m, gc_param)

            # HL: hearing loss (hearing impared) listener
            gc_param_hl3 = GCparamDefault() # reset
            gc_param_hl3.hloss_type = 'HL3'
            gc_param_hl3.hloss_compression_health = 0.5
            dcgc_out_hl3, _, gc_param_hl3, _ = gcfb.gcfb_v234(snd_eq_m, gc_param_hl3)

            """
            reduction of TMTF
            """
            # HL+TMTF: hearing loss and TMTF reduction
            em_param = EMparamDefault()
            gcem_loss, em_param = gcfb.gcfb_v23_env_mod_loss(dcgc_out_hl3, gc_param_hl3, em_param)
            gcem_frame, em_param = gcfb.gcfb_v23_ana_env_mod(dcgc_out_hl3, gc_param_hl3, em_param)

            """
            Nomalizing output level of dcGC (absolute threshold 0 dB == rms of 1)
            """
            dcgc_out_nh = utils.eqlz_gcfb2rms1_at_0db(dcgc_out_nh)
            dcgc_out_hl3 = utils.eqlz_gcfb2rms1_at_0db(dcgc_out_hl3)
            gcem_loss = utils.eqlz_gcfb2rms1_at_0db(gcem_loss)
            
            """
            Plot
            """
            fig = plt.figure()
            scaling = np.max(dcgc_out_nh)
            num_ch, num_frame = dcgc_out_nh.shape
            y_ticks = np.arange(0, num_ch+1, 20)

            # NH
            ax1 = fig.add_subplot(3, 1, 1)
            plt.imshow(np.maximum(dcgc_out_nh, 0), vmax=scaling, extent=[0, num_frame, 0, num_ch], \
                       aspect='auto', origin='lower', cmap='jet')
            ax1.set_title('Type: ' + gc_param_nh.hloss.type + f', Level: {dbspl}dB SPL')
            ax1.set_yticks(y_ticks)
            ax1.set_ylabel("# of channel")

            # HL
            ax2 = fig.add_subplot(3, 1, 2)
            plt.imshow(np.maximum(dcgc_out_hl3, 0), vmax=scaling, extent=[0, num_frame, 0, num_ch], \
                       aspect='auto', origin='lower', cmap='jet')
            ax2.set_title('Type: ' + gc_param_hl3.hloss.type + f', Level: {dbspl}dB SPL')
            ax2.set_yticks(y_ticks)
            ax2.set_ylabel("# of channel")

            # HL+TMTF
            ax3 = fig.add_subplot(3, 1, 3)
            plt.imshow(np.maximum(gcem_loss, 0), vmax=scaling, extent=[0, num_frame, 0, num_ch], \
                       aspect='auto', origin='lower', cmap='jet')
            ax3.set_title('Type: ' + gc_param_hl3.hloss.type + '+TMTF reduct.' \
                          + f', Level: {dbspl}dB SPL')
            ax3.set_yticks((y_ticks))
            ax3.set_xlabel("# of frame")
            ax3.set_ylabel("# of channel")

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)

            plt.savefig(f'{path_dir}/figs/spec_{ctrl}_pulse_{dbspl}dB.png')

            """
            Plot resuts: power distributions in each frame
            """
            fig = plt.figure()
            n_ch_all = np.arange(0, gc_param.num_ch)

            for cnt, n_slice in enumerate(np.arange(100-1, 200, 50)):
                ax = fig.add_subplot(2, 2, cnt+1)
                ax.plot(n_ch_all, dcgc_out_nh[n_ch_all, n_slice], 'b-', label='NH') 
                ax.plot(n_ch_all, dcgc_out_hl3[n_ch_all, n_slice], 'r--',label='HL3')
                ax.plot(n_ch_all, gcem_loss[n_ch_all, n_slice], 'y-.', label='HL+TMTF')
                ax.set_xlim([0, 100])
                ax.set_xlabel('# of channel')
                ax.set_ylabel('power')
                ax.set_title(f'# of frame: {n_slice+1}')
                
                if cnt == 0:
                    ax.legend()

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)

            plt.savefig(f'{path_dir}/figs/power_dist_pulse_{dbspl}dB.png')
    
    """
    Plot results: audiograms
    """
    sw_plot = True
    #sw_plot = False

    if sw_plot:
        plt.rcParams['figure.figsize'] = (5, 10)
        fig = plt.figure()

        # hearing loss type
        gc_param_nh.hloss_type = 'HL3'
        # compression helth factor \alpha
        ch_list = np.array([1, 0.5, 0])

        for n_ch, ch in enumerate(ch_list):
            ax = fig.add_subplot(3, 1, n_ch+1)
            gc_param_nh.hloss_compression_health = ch

            gc_param = gcfb.gcfb_v23_hearing_loss(gc_param_nh, gc_resp_nh)

            # plot audiogram
            log_fag = np.log2(gc_param.hloss.f_audgram_list)
            ax.plot(log_fag, np.zeros(gc_param.hloss.f_audgram_list.shape), 'bx-', label='NH')
            ax.plot(log_fag, -gc_param.hloss.pin_loss_db_act, 'r*--', label='HL(ACT)')
            ax.plot(log_fag, -gc_param.hloss.pin_loss_db_pas, 'y^--', label='HL(PAS)')
            ax.plot(log_fag, -gc_param.hloss.hearing_level_db, 'mo--', label='HearingLevel')
            ax.plot(log_fag, -gc_param.hloss.pin_loss_db_act_init, 'g:', label='HL(ACT)_Init')

            ax.set_xticks(log_fag)
            ax.set_xticklabels(gc_param.hloss.f_audgram_list)
            for n_fag, fag in enumerate(gc_param.hloss.f_audgram_list):
                # show compression health factor at each frequency
                ax.text(np.log2(fag), -gc_param.hloss.pin_loss_db_act[n_fag]+3,
                        f'{gc_param.hloss.compression_health[n_fag]:4.2f}',
                        horizontalalignment='center')
            ax.legend()
            ax.grid()
            ax.set_xlim([np.log2(100), np.log2(10000)])
            ax.set_ylim([-80, 10])
            ax.set_yticks(ticks=range(-80, 10+1, 10))
            ax.set_yticklabels(range(80, -10-1, -10))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Hearing Level (dB)')
            ax.set_title(f'Audiogram(ACT&PAS): {gc_param.hloss.type}\n'
                         + f'compression factor:{ch}')

        plt.tight_layout()
        plt.savefig(f'{path_dir}/figs/audiograms_{gc_param_nh.hloss_type}.png')
        plt.show()

if __name__ == '__main__':
    main()
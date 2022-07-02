# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
from scipy import signal
import utils
import GCFBv211_SetParam as gcfb_set_param
import GammaChirp as gcfb


def gcfb_v211(snd_in, gc_param, *args):
    """Dynamic Compressive Gammachirp Filterbank (dcGC-FB)

    Args:
        snd_in (float): Input sound
        gc_param (struct): Parameters of dcGC-FB
            .fs: Sampling rate (default: 48000)
            .NumCh: Number of Channels (default: 100)
            .FRange: Frequency Range of GCFB (default: [100, 6000]) s
                     pecifying asymtopic freq. of passive GC (Fr1)

    Returns:
        cgc_out: ompressive GammaChirp Filter Output
        pgc_out: Passive GammaChirp Filter Output
        Ppgc: Power at the output of passive GC

    Note: 
        1)  This version is completely different from GCFB v.1.04 (obsolete).
            We introduced the "compressive gammachirp" to accomodate both the 
            psychoacoustical simultaneous masking and the compressive 
            characteristics (Irino and Patterson, 2001). The parameters were 
            determined from large dataset (See Patterson, Unoki, and Irino, 2003.)   

    References:
        Irino,T. and Unoki,M.:  IEEE ICASSP98, pp.3653-3656, May 1998.
        Irino,T. and Patterson,R.D. :  JASA, Vol.101, pp.412-419, 1997.
        Irino,T. and Patterson,R.D. :  JASA, Vol.109, pp.2008-2022, 2001.
        Patterson,R.D., Unoki,M. and Irino,T. :  JASA, Vol.114,pp.1529-1542,2003.
        Irino,T. and and Patterson,R.D. : IEEE Trans.ASLP, Vol.14, Nov. 2006.
    """

    # Handling Input Parameters
    if len(args) > 0:
        help(gcfb_v211)
        sys.exit()

    size = np.shape(snd_in)
    if not len(size) == 1:
        print("Check snd_in. It should be 1 ch (Monaural) and  a single row vector.", file=sys.stderr)
        sys.exit(1)
    len_snd = len(snd_in)
    
    gc_param, gc_resp = gcfb_set_param.SetParam(gc_param)
    fs = gc_param.fs
    num_ch = gc_param.NumCh

    """
    Outer-Mid Ear Compensation
    for inverse filter, use Out utits.OutMidCrctFilt('ELC', fs, 0, 1)
    """
    if gc_param.OutMidCrct == 'No':
        print("*** No Outer/Middle Ear correction ***")
        Snd = snd_in
    else:
        # if gc_param.OutMidCrct in ["ELC", "MAF", "MAP"]:
        print(f"*** Outer/Middle Ear correction (minimum phase) : {gc_param.OutMidCrct} ***")
        cmpn_out_mid, _ = utils.OutMidCrctFilt(gc_param.OutMidCrct, fs, 0, 2) # 2) minimum phase
        # 1kHz: -4 dB, 2kHz: -1 dB, 4kHz: +4 dB (ELC)
        # Now we use Minimum phase version of OutMidCrctFilt (modified 16 Apr. 2006).
        # No compensation is necessary.  16 Apr. 2006

    
    """
    Gammachirp
    """
    print("*** Gammachirp Calculation ***")

    sw_fast_prcs = 1 # ON: fast processing for static filter
    if not sw_fast_prcs == 1:
        print("sw_fast_prcs should be 1.", file=sys.stderr)
        sys.exit(1)
    if sw_fast_prcs == 1 and gc_param.Ctrl == "static":
        # 'Fast processing for linear cGC gain at gc_param.LeveldBscGCFB'
        """
        for HP-AF
        """
        lvl_db = gc_param.LeveldBscGCFB
        fratVal = gc_param.frat[0,0] + gc_param.frat[0,1] * gc_resp.Ef \
            + (gc_param.frat[1,0] + gc_param.frat[1,1] * gc_resp.Ef) * lvl_db
        fr2val = fratVal * gc_resp.Fp1
        gc_resp.Fr2 = fr2val.copy()
        acf_coef_fast_prcs = utils.MakeAsymCmpFiltersV2(fs, fr2val, gc_resp.b2val, gc_resp.c2val)
    else:
        # HP-AF for dynamic-GC level estimation path. 18 Dec 2012 Checked
        fr2lvl_est = gc_param.LvlEst.frat * gc_resp.Fp1
        # default gc_param.LvlEst.frat = 1.08 (GCFBv208_SetParam(gc_param))
        # ---> Linear filter for level estimation
        acf_coef_lvl_est = utils.MakeAsymCmpFiltersV2(fs,fr2lvl_est,gc_param.LvlEst.b2, gc_param.LvlEst.c2)

    """
    Start calculation
    """
    """
    Passive Gammachirp & Levfel estimation filtering
    """
    t_start = time.time()
    cgc_out = np.zeros((num_ch, len_snd))
    pgc_out = np.zeros((num_ch, len_snd))
    Ppgc = np.zeros((num_ch, len_snd))
    cgc_out_lvl_est = np.zeros((num_ch, len_snd))

    print("--- Channel-by-channel processing ---")

    for nch in range(num_ch):

        # passive gammachirp
        pgc, _, _, _ = gcfb.GammaChirp(gc_resp.Fr1[nch], fs, gc_param.n, gc_resp.b1val[nch], gc_resp.c1val[nch], 0, '', 'peak')

        # fast FFT-based filtering by the pgc
        pgc_out[nch, 0:len_snd] = utils.fftfilt(pgc[0,:], Snd) 

        # Fast processing for fixed cGC
        if sw_fast_prcs == 1 and gc_param.Ctrl == 'static': # Static
            str_gc = "Static (Fixed) Compressive-Gammachirp"
            gc_out1 = pgc_out[nch, :].copy()
            for n_filt in range(4):
                gc_out1 = signal.lfilter(acf_coef_fast_prcs.bz[nch, :, n_filt], acf_coef_fast_prcs.ap[nch, :, n_filt], gc_out1)
            cgc_out[nch, :] = gc_out1.copy()
            gc_resp.Fp2[nch], _ = utils.Fr1toFp2(gc_param.n, gc_resp.b1val[nch], gc_resp.c1val[nch], \
                                             gc_resp.b2val[nch], gc_resp.c2val[nch], \
                                             fratVal[nch], gc_resp.Fr1[nch])
            if nch == num_ch:
                gc_resp.Fp2 = gc_resp.Fp2

        else: # Level estimation pass for Dynamic.
            str_gc = "Passive-Gammachirp & Level estimation filter"
            gc_out1 = pgc_out[nch, :].copy()
            for n_filt in range(4):
                gc_out1 = signal.lfilter(acf_coef_lvl_est.bz[nch, :, n_filt], acf_coef_lvl_est.ap[nch, :, n_filt], gc_out1)
            cgc_out_lvl_est[nch, :] = gc_out1.copy()

        if nch == 0 or np.mod(nch+1, 20) == 0: # "rem" is not defined in the original code! 
        # if nch == 0:
            t_now = time.time()
            print(str_gc + " ch #{}".format(nch+1) + " / #{}.   ".format(num_ch) \
                 + "elapsed time = {} (sec)".format(np.round(t_now-t_start, 1)))

    # added level estimation circuit only, 25 Nov. 2013
    if gc_param.Ctrl == 'level':
            cgc_out = cgc_out_lvl_est.copy()
            lvl_db = []

    # Passive filter (static/level estimation) -->  jump to Gain Normalization


    """
    Dynamic Compressive Gammachirp filtering
    Sample-by-sample processing
    """

    if gc_param.Ctrl == 'dynamic':

        # Initial settings
        num_disp = int(np.fix(len_snd/10)) # display 10 times per Snd
        cgc_out = np.zeros((num_ch, len_snd))
        gc_resp.Fr2 = np.zeros((num_ch, len_snd))
        gc_resp.fratVal = np.zeros((num_ch, len_snd))
        gc_resp.Fp2 = []
        lvl_db = np.zeros((num_ch, len_snd))
        lvl_lin = np.zeros((num_ch, 2))
        lvl_lin_prev = np.zeros((num_ch, 2))

        # Sample-by-sample processing
        print("--- Sample-by-sample processing ---")
        t_start = time.time()

        for nsmpl in range(len_snd):

            """
            Level estimation circuit
            """
            lvl_lin[0:num_ch, 0] = \
                np.maximum(np.max(pgc_out[gc_param.LvlEst.NchLvlEst.astype(int)-1, nsmpl], initial=0, axis=1), \
                    lvl_lin_prev[:, 0]*gc_param.LvlEst.ExpDecayVal)
            lvl_lin[0:num_ch, 1] = \
                np.maximum(np.max(cgc_out_lvl_est[gc_param.LvlEst.NchLvlEst.astype(int)-1, nsmpl], initial=0, axis=1), \
                    lvl_lin_prev[:, 1]*gc_param.LvlEst.ExpDecayVal)

            lvl_lin_prev = lvl_lin.copy()

            lvl_lin_ttl = gc_param.LvlEst.Weight \
                * gc_param.LvlEst.LvlLinRef * (lvl_lin[:, 0] / gc_param.LvlEst.LvlLinRef)**gc_param.LvlEst.Pwr[0] \
                    + (1 - gc_param.LvlEst.Weight) \
                        * gc_param.LvlEst.LvlLinRef * (lvl_lin[:, 1] / gc_param.LvlEst.LvlLinRef)**gc_param.LvlEst.Pwr[1]
                
            lvl_db[:, [nsmpl]] = np.array([20 * np.log10(np.maximum(lvl_lin_ttl, gc_param.LvlEst.LvlLinMinLim)) \
                + gc_param.LvlEst.RMStoSPLdB]).T

            """
            Signal path
            """
            # Filtering High-Pass Asymmetric Comressive Filter
            fratVal = gc_param.frat[0, 0] + gc_param.frat[0, 1] * gc_resp.Ef[:] + \
                (gc_param.frat[1, 0] + gc_param.frat[1, 1] * gc_resp.Ef[:]) * lvl_db[:, [nsmpl]]
            fr2val = gc_resp.Fp1[:] * fratVal

            if np.mod(nsmpl, gc_param.NumUpdateAsymCmp) == 0: # update periodically
                acf_coef = utils.MakeAsymCmpFiltersV2(fs, fr2val, gc_resp.b2val, gc_resp.c2val)

            if nsmpl == 0:
                _, acf_status = utils.ACFilterBank(acf_coef, []) # initialization

            sig_out, acf_status = utils.ACFilterBank(acf_coef, acf_status, pgc_out[:, nsmpl])
            cgc_out[:, [nsmpl]] = sig_out.copy()
            gc_resp.Fr2[:, [nsmpl]] = fr2val.copy()
            gc_resp.fratVal[:, [nsmpl]] = fratVal.copy()

            if nsmpl == 0 or np.mod(nsmpl+1, num_disp) == 0:
                t_now = time.time()
                print("Dynamic Compressive-Gammachirp: Time {} (ms) / {} (ms). elapsed time = {} (sec)"\
                    .format(round(nsmpl/fs*1000, 1), len_snd/fs*1000, np.round(t_now-t_start, 1)))

        """
        End of Dynamic Compressive Gammachirp filtering
        """

        """
        Signal path Gain Normalization at Reference Level (GainRefdB) for static dynamic filters
        """

        fratRef = gc_param.frat[0, 0] + gc_param.frat[0, 1] * gc_resp.Ef[:] \
            + (gc_param.frat[1, 0] + gc_param.frat[1, 1] * gc_resp.Ef[:]) * gc_param.GainRefdB

        cgc_ref = utils.CmprsGCFrsp(gc_resp.Fr1, fs, gc_param.n, gc_resp.b1val, \
                                   gc_resp.c1val, fratRef, gc_resp.b2val, gc_resp.c2val)
        gc_resp.cGCRef = cgc_ref
        gc_resp.LvldB = lvl_db

        gc_resp.GainFactor = 10**(gc_param.GainCmpnstdB/20) * cgc_ref.NormFctFp2
        cgc_out = (gc_resp.GainFactor * np.ones((1, len_snd))) * cgc_out

    return cgc_out, pgc_out, gc_param, gc_resp
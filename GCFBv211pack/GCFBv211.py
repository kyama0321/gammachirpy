# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
from scipy import signal
import utils
import GCFBv211_SetParam as gcfb_SetParam
import GammaChirp as gcfb
from thirdparty.overlapadd.olafilt import olafilt
import matplotlib.pyplot as plt


def GCFBv211(SndIn, GCparam, *args):
    """Dynamic Compressive Gammachirp Filterbank (dcGC-FB)

    Args:
        SndIn (float): Input sound
        GCparam (struct): Parameters of dcGC-FB
            .fs: Sampling rate (default: 48000)
            .NumCh: Number of Channels (default: 100)
            .FRange: Frequency Range of GCFB (default: [100, 6000]) specifying asymtopic freq. of passive GC (Fr1)

    Returns:
        cGCout: ompressive GammaChirp Filter Output
        pGCout: Passive GammaChirp Filter Output
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
        help(GCFBv211)
        sys.exit()

    size = np.shape(SndIn)
    if not len(size) == 1:
        print("Check SndIn. It should be 1 ch (Monaural) and  a single row vector.", file=sys.stderr)
        sys.exit(1)
    LenSnd = len(SndIn)
    
    GCparam, GCresp = gcfb_SetParam.SetParam(GCparam)
    fs = GCparam.fs
    NumCh = GCparam.NumCh

    """
    Outer-Mid Ear Compensation
    for inverse filter, use Out utits.OutMidCrctFilt('ELC', fs, 0, 1)
    """
    if GCparam.OutMidCrct == 'No':
        print("*** No Outer/Middle Ear correction ***")
        Snd = SndIn
    else:
        # if GCparam.OutMidCrct in ["ELC", "MAF", "MAP"]:
        print("*** Outer/Middle Ear correction (minimum phase) : {} ***".format(GCparam.OutMidCrct))
        CmpnOutMid, _ = utils.OutMidCrctFilt(GCparam.OutMidCrct, fs, 0, 2) # 2) minimum phase
        # 1kHz: -4 dB, 2kHz: -1 dB, 4kHz: +4 dB (ELC)
        # Now we use Minimum phase version of OutMidCrctFilt (modified 16 Apr. 2006).
        # No compensation is necessary.  16 Apr. 2006

    
    """
    Gammachirp
    """
    print("*** Gammachirp Calculation ***")

    SwFastPrcs = 1 # ON: fast processing for static filter
    if not SwFastPrcs == 1:
        print("SwFastPrcs should be 1.", file=sys.stderr)
        sys.exit(1)
    if SwFastPrcs == 1 and GCparam.Ctrl == "static":
        # 'Fast processing for linear cGC gain at GCparam.LeveldBscGCFB'
        """
        for HP-AF
        """
        LvldB =GCparam.LeveldBscGCFB
        fratVal = GCparam.frat[0,0] + GCparam.frat[0,1] * GCresp.Ef \
            + (GCparam.frat[1,0] + GCparam.frat[1,1] * GCresp.Ef) * LvldB
        Fr2val = fratVal * GCresp.Fp1
        GCresp.Fr2 = Fr2val.copy()
        ACFcoefFastPrcs = utils.MakeAsymCmpFiltersV2(fs, Fr2val, GCresp.b2val, GCresp.c2val)
    else:
        # HP-AF for dynamic-GC level estimation path. 18 Dec 2012 Checked
        Fr2LvlEst = GCparam.LvlEst.frat * GCresp.Fp1
        # default GCparam.LvlEst.frat = 1.08 (GCFBv208_SetParam(GCparam))
        # ---> Linear filter for level estimation
        ACFcoefLvlEst = utils.MakeAsymCmpFiltersV2(fs,Fr2LvlEst,GCparam.LvlEst.b2, GCparam.LvlEst.c2)

    """
    Start calculation
    """
    """
    Passive Gammachirp & Levfel estimation filtering
    """
    Tstart = time.time()
    cGCout = np.zeros((NumCh, LenSnd))
    pGCout = np.zeros((NumCh, LenSnd))
    Ppgc = np.zeros((NumCh, LenSnd))
    cGCoutLvlEst = np.zeros((NumCh, LenSnd))

    print("--- Channel-by-channel processing ---")

    for nch in range(NumCh):

        # passive gammachirp
        pgc, _, _, _ = gcfb.GammaChirp(GCresp.Fr1[nch], fs, GCparam.n, GCresp.b1val[nch], GCresp.c1val[nch], 0, '', 'peak')

        pGCout[nch, 0:LenSnd] = olafilt(pgc[0,:], Snd) # fast FFT-based filtering

        # Fast processing for fixed cGC
        if SwFastPrcs == 1 and GCparam.Ctrl == 'static': # Static
            StrGC = "Static (Fixed) Compressive-Gammachirp"
            GCout1 = pGCout[nch, :].copy()
            for Nfilt in range(4):
                GCout1 = signal.lfilter(ACFcoefFastPrcs.bz[nch, :, Nfilt], ACFcoefFastPrcs.ap[nch, :, Nfilt], GCout1)
            cGCoutLvlEst[nch, :] = GCout1.copy()
            GCresp.Fp2[nch], _ = utils.Fr1toFp2(GCparam.n, GCresp.b1val[nch], GCresp.c1val[nch], \
                                             GCresp.b2val[nch], GCresp.c2val[nch], \
                                             fratVal[nch], GCresp.Fr1[nch])
            if nch == NumCh:
                GCresp.Fp2 = GCresp.Fp2

        else: # Level estimation pass for Dynamic.
            StrGC = "Passive-Gammachirp & Level estimation filter"
            GCout1 = pGCout[nch, :].copy()
            for Nfilt in range(4):
                GCout1 = signal.lfilter(ACFcoefLvlEst.bz[nch, :, Nfilt], ACFcoefLvlEst.ap[nch, :, Nfilt], GCout1)
            cGCoutLvlEst[nch, :] = GCout1.copy()

        if nch == 0 or np.mod(nch+1, 20) == 0: # "rem" is not defined in the original code! 
        # if nch == 0:
            Tnow = time.time()
            print(StrGC + " ch #{}".format(nch+1) + " / #{}.   ".format(NumCh) \
                 + "elapsed time = {} (sec)".format(np.round(Tnow-Tstart, 1)))

    # added level estimation circuit only, 25 Nov. 2013
    if GCparam.Ctrl == 'level':
            cGCout = cGCoutLvlEst.copy()
            LvldB = []

    # Passive filter (static/level estimation) -->  jump to Gain Normalization


    """
    Dynamic Compressive Gammachirp filtering
    Sample-by-sample processing
    """

    if GCparam.Ctrl == 'dynamic':

        # Initial settings
        nDisp = int(np.fix(LenSnd/10)) # display 10 times per Snd
        cGCout = np.zeros((NumCh, LenSnd))
        GCresp.Fr2 = np.zeros((NumCh, LenSnd))
        GCresp.fratVal = np.zeros((NumCh, LenSnd))
        GCresp.Fp2 = []
        LvldB = np.zeros((NumCh, LenSnd))
        LvlLin = np.zeros((NumCh, 2))
        LvlLinPrev = np.zeros((NumCh, 2))

        # Sample-by-sample processing
        print("--- Sample-by-sample processing ---")
        Tstart = time.time()

        for nsmpl in range(LenSnd):

            """
            Level estimation circuit
            """
            LvlLin[0:NumCh, 0] = \
                np.maximum(np.max(pGCout[GCparam.LvlEst.NchLvlEst.astype(int)-1, nsmpl], initial=0, axis=1), \
                    LvlLinPrev[:, 0]*GCparam.LvlEst.ExpDecayVal)
            LvlLin[0:NumCh, 1] = \
                np.maximum(np.max(cGCoutLvlEst[GCparam.LvlEst.NchLvlEst.astype(int)-1, nsmpl], initial=0, axis=1), \
                    LvlLinPrev[:, 1]*GCparam.LvlEst.ExpDecayVal)

            LvlLinPrev = LvlLin.copy()

            LvlLinTtl = GCparam.LvlEst.Weight \
                * GCparam.LvlEst.LvlLinRef * (LvlLin[:, 0] / GCparam.LvlEst.LvlLinRef)**GCparam.LvlEst.Pwr[0] \
                    + (1 - GCparam.LvlEst.Weight) \
                        * GCparam.LvlEst.LvlLinRef * (LvlLin[:, 1] / GCparam.LvlEst.LvlLinRef)**GCparam.LvlEst.Pwr[1]
                
            LvldB[:, [nsmpl]] = np.array([20 * np.log10(np.maximum(LvlLinTtl, GCparam.LvlEst.LvlLinMinLim)) \
                + GCparam.LvlEst.RMStoSPLdB]).T

            """
            Signal path
            """
            # Filtering High-Pass Asymmetric Comressive Filter
            fratVal = GCparam.frat[0, 0] + GCparam.frat[0, 1] * GCresp.Ef[:] + \
                (GCparam.frat[1, 0] + GCparam.frat[1, 1] * GCresp.Ef[:]) * LvldB[:, [nsmpl]]
            Fr2Val = GCresp.Fp1[:] * fratVal

            if np.mod(nsmpl, GCparam.NumUpdateAsymCmp) == 0: # update periodically
                ACFcoef = utils.MakeAsymCmpFiltersV2(fs, Fr2Val, GCresp.b2val, GCresp.c2val)

            if nsmpl == 0:
                _, ACFstatus = utils.ACFilterBank(ACFcoef, []) # initialization

            SigOut, ACFstatus = utils.ACFilterBank(ACFcoef, ACFstatus, pGCout[:, nsmpl])
            cGCout[:, [nsmpl]] = SigOut.copy()
            GCresp.Fr2[:, [nsmpl]] = Fr2Val.copy()
            GCresp.fratVal[:, [nsmpl]] = fratVal.copy()

            if nsmpl == 0 or np.mod(nsmpl+1, nDisp) == 0:
                Tnow = time.time()
                print("Dynamic Compressive-Gammachirp: Time {} (ms) / {} (ms). elapsed time = {} (sec)"\
                    .format(round(nsmpl/fs*1000, 1), LenSnd/fs*1000, np.round(Tnow-Tstart, 1)))

        """
        End of Dynamic Compressive Gammachirp filtering
        """

        """
        Signal path Gain Normalization at Reference Level (GainRefdB) for static dynamic filters
        """

        fratRef = GCparam.frat[0, 0] + GCparam.frat[0, 1] * GCresp.Ef[:] \
            + (GCparam.frat[1, 0] + GCparam.frat[1, 1] * GCresp.Ef[:]) * GCparam.GainRefdB

        cGCRef = utils.CmprsGCFrsp(GCresp.Fr1, fs, GCparam.n, GCresp.b1val, GCresp.c1val, fratRef, GCresp.b2val, GCresp.c2val)
        GCresp.cGCRef = cGCRef
        GCresp.LvldB = LvldB

        GCresp.GainFactor = 10**(GCparam.GainCmpnstdB/20) * cGCRef.NormFctFp2
        cGCout = (GCresp.GainFactor * np.ones((1, LenSnd))) * cGCout

    return cGCout, pGCout, GCparam, GCresp
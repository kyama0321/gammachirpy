# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
import utils
import GCFBv211_SetParam as gcfb_SetParam
import GammaChirp as gcfb


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
        GCresp.Fr2 = Fr2val
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
    # Tstart = time.perf_counter()
    cGCout = np.zeros((NumCh, LenSnd))
    pGCout = np.zeros((NumCh, LenSnd))
    Ppgc = np.zeros((NumCh, LenSnd))
    cGCoutLvlEst = np.zeros((NumCh, LenSnd))

    print("--- Channel-by-channel processing ---")

    for nch in range(NumCh):

        # passive gammachirp
        pgc, _, _, _ = gcfb.GammaChirp(GCresp.Fr1[nch], fs, GCparam.n, GCresp.b1val[nch], GCresp.c1val[nch], 0, '', 'peak')


    return cGCout, pGCout, GCparam, GCresp
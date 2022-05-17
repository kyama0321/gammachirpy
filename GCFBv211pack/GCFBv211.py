# -*- coding: utf-8 -*-
import numpy as np
import datetime
import sys
import utils
import GCFBv211_SetParam as gcfb_SetParam


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

    Tstart0 = datetime.datetime.now()

    size = SndIn.shape
    if len(size) != 1:
        print("Check SndIn. It should be 1 ch (Monaural) and  a single row vector.", file=sys.stderr)
        sys.exit(1)
    
    GCparam, GCresp = gcfb_SetParam.SetParam(GCparam)
    fs = GCparam.fs
    NumCh = GCparam.NumCh

    """
    Outer-Mid Ear Compensation
    """
    if GCparam.OutMidCrct == 'No':
        print("*** No Outer/Middle Ear correction ***")
        Snd = SndIn
    else:
        # if GCparam.OutMidCrct in ["ELC", "MAF", "MAP"]:
        print("*** Outer/Middle Ear correction (minimum phase) : {} ***".format(GCparam.OutMidCrct))
        CmpnOutMid, _ = utils.OutMidCrctFilt(GCparam.OutMidCrct, fs, 0, 2) # 2) minimum phase

    return cGCout, pGCout, GCparam, GCresp
# -*- coding: utf-8 -*-
import numpy as np
import sys
import utils


class GCresp:
    Fr1 = []
    Fr2 = []
    ERBspace1 = []
    Ef = []
    b1val = []
    c1val = []
    Fp1 = []
    Fp2 = []
    b2val = []
    c1val = []
    fratVal = []

class LvlEst:
    LctERB = []
    DecayHL = []
    b2 = []
    c2 = []
    frat = []
    RMStoSPLdB = []
    Weight = []
    RefdB = []
    Pwr = []
    ExpDecayVal = []
    NchShift = []
    NchLvlEst = []
    LvlLinMinLim = []
    LvlLinRef = []


def set_param(GCparam):
    """Setting Default Parameters for GCFBv2

    Args:
        GCparam (struct): Your preset gammachirp parameters
            .fs: Sampling rate (default: 48000)
            .NumCh: Number of Channels (default: 100)
            .FRange: Frequency Range of GCFB (default: [100, 6000]) 
                     specifying asymtopic freq. of passive GC (Fr1)

    Returns:
        GCparam (struct): GCparam values
    """
    if not hasattr(GCparam, 'fs'):
        GCparam.fs = 48000

    if not hasattr(GCparam, 'OutMidCrct'):
        GCparam.OutMidCrct = 'ELC'
        # if no OutMidCrct is not necessary, specify GCparam.OutMidCrct = 'no'

    if not hasattr(GCparam, 'NumCh'):
        GCparam.NumCh = 100

    if not hasattr(GCparam, 'FRange'):
        GCparam.FRange = np.array([100, 6000])
    
    # Gammachirp parameters
    if not hasattr(GCparam, 'n'):
        GCparam.n = 4 # default gammatone & gammachirp

    # Convention
    if not hasattr(GCparam, 'b1'):
        GCparam.b1 = np.array([1.81, 0]) # b1 becomes two coeffs in v210 (18 Apr. 2015). Frequency independent by 0. 

    if len(GCparam.b1) == 1:
        GCparam.b1.append(0) # frequency independent by 0

    if not hasattr(GCparam, 'c1'):
        GCparam.c1 = np.array([-2.96, 0]) # c1 becomes two coeffs. in v210 (18 Apr. 2015). Frequency independent by 0.

    if len(GCparam.c1) == 1:
        GCparam.c1.append(0) # frequency independent by 0
    
    if not hasattr(GCparam, 'frat'):
        GCparam.frat = np.array([[0.4660, 0], [0.0109, 0]])
    
    if not hasattr(GCparam, 'b2'):
        GCparam.b2 = np.array([[2.17, 0], [0, 0]]) # no level-dependency  (8 Jul 05)

    if not hasattr(GCparam, 'c2'):
        GCparam.c2 = np.array([[2.20, 0], [0, 0]]) # no level-dependency; no freq-dependency (3 Jun 05)

    if not hasattr(GCparam, 'Ctrl'):
        GCparam.Ctrl = 'static'      

    if not hasattr(GCparam, 'Ctrl'):
        GCparam.Ctrl = 'static'
    if 'fix' in GCparam.Ctrl:
        GCparam.Ctrl = 'static'
    if 'tim' in GCparam.Ctrl:
        GCparam.Ctrl = 'dynamic'
    if not 'sta' in GCparam.Ctrl and not 'dyn' in GCparam.Ctrl and not 'lev' in GCparam.Ctrl:
        print("Specify GCparam.Ctrl:  'static', 'dynamic', or 'level(-estimation). \
               (old version 'fixed'/'time-varying')", file=sys.stderr)
        sys.exit(1)

    if not hasattr(GCparam, 'GainCmpnstdB'):
        GCparam.GainCmpnstdB = -1 # in dB. when LvlEst.c2==2.2, 1 July 2005

    """
    Parameters for level estimation
    """
    if hasattr(GCparam, 'PpgcRef') or hasattr(GCparam, 'LvlRefdB'):
        print("The parameter 'GCparam.PpgcRef' is obsolete.")
        print("The parameter 'GCparam.LvlRefdB' is obsolete.")
        print("Please change it to 'GCparam.GainRefdB'", file=sys.stderr)
        sys.exit(1)
    
    if not hasattr(GCparam, 'GainRefdB'):
        GCparam.GainRefdB = 50 # reference Ppgc level for gain normalization

    if not hasattr(GCparam, 'LeveldBscGCFB'):
        GCparam.LeveldBscGCFB = GCparam.GainRefdB # use it as default

    if not hasattr(GCparam, 'LvlEst'):
        GCparam.LvlEst = LvlEst()

    if len(GCparam.LvlEst.LctERB) == 0:
        #GCparam.LvlEst.LctERB = 1.0
        # Location of Level Estimation pGC relative to the signal pGC in ERB
        # see testGC_LctERB.py for fitting result. 10 Sept 2004
        GCparam.LvlEst.LctERB = 1.5;   # 16 July 05

    if len(GCparam.LvlEst.DecayHL) == 0:
        GCparam.LvlEst.DecayHL = 0.5; # 18 July 2005

    if len(GCparam.LvlEst.b2) == 0:
        GCparam.LvlEst.b2 = GCparam.b2[0, 0]

    if len(GCparam.LvlEst.c2) == 0:
        GCparam.LvlEst.c2 = GCparam.c2[0, 0]

    if len(GCparam.LvlEst.frat) == 0:
        # GCparam.LvlEst.frat = 1.1 #  when b=2.01 & c=2.20
        GCparam.LvlEst.frat = 1.08 # peak of cGC ~= 0 dB (b2=2.17 & c2=2.20)

    if len(GCparam.LvlEst.RMStoSPLdB) == 0:
        GCparam.LvlEst.RMStoSPLdB = 30 # 1 rms == 30 dB SPL for Meddis IHC

    if len(GCparam.LvlEst.Weight) == 0:
        GCparam.LvlEst.Weight = 0.5

    if len(GCparam.LvlEst.RefdB) == 0:
        GCparam.LvlEst.RefdB = 50 # 50 dB SPL

    if len(GCparam.LvlEst.Pwr) == 0:
        GCparam.LvlEst.Pwr = np.array([1.5, 0.5]) # Weight for pGC & cGC

    # new 19 Dec 2011
    if not hasattr(GCparam, 'NumUpdateAsymCmp'):
        # GCparam.NumUpdateAsymCmp = 3 # updte every 3 samples (== 3*GCFBv207)
        GCparam.NumUpdateAsymCmp = 1 # samply-by-sample (==GCFBv207)

    """
    GCresp
    """
    Fr1, ERBrate1 = utils.EqualFreqScale('ERB', GCparam.NumCh, GCparam.FRange)
    GCresp.Fr1 = np.array([Fr1]).T
    GCresp.ERBspace1 = np.mean(np.diff(ERBrate1))
    ERBrate, ERBw = utils.Freq2ERB(GCresp.Fr1)
    ERBrate1kHz, ERBw1kHz = utils.Freq2ERB(1000)
    GCresp.Ef = ERBrate/ERBrate1kHz - 1

    OneVec = np.ones([GCparam.NumCh, 1])
    GCresp.b1val = GCparam.b1[0]*OneVec + GCparam.b1[1]*GCresp.Ef
    GCresp.c1val = GCparam.c1[0]*OneVec + GCparam.c1[1]*GCresp.Ef

    GCresp.Fp1, _ = utils.fr2fpeak(GCparam.n, GCresp.b1val, GCresp.c1val, GCresp.Fr1)
    GCresp.Fp2 = np.zeros(np.shape(GCresp.Fp1))

    GCresp.b2val = GCparam.b2[0, 0]*OneVec + GCparam.b2[0, 1]*GCresp.Ef
    GCresp.c2val = GCparam.c2[0, 0]*OneVec + GCparam.c2[0, 1]*GCresp.Ef
    
    """
    Set Params estimation circuit
    """
    # keep LvlEst params  3 Dec 2013
    ExpDecayVal = np.exp(-1/(GCparam.LvlEst.DecayHL*GCparam.fs/1000)*np.log(2)) # decay exp
    NchShift = np.round(GCparam.LvlEst.LctERB/GCresp.ERBspace1)
    NchLvlEst = np.minimum(np.maximum(1, np.array([np.arange(GCparam.NumCh)+1]).T+NchShift), \
                           GCparam.NumCh) # shift in NumCh [1:NumCh]
    LvlLinMinLim = 10**(-GCparam.LvlEst.RMStoSPLdB/20) # minimum sould be SPL 0 dB
    LvlLinRef = 10**((GCparam.LvlEst.RefdB - GCparam.LvlEst.RMStoSPLdB)/20)

    GCparam.LvlEst.ExpDecayVal = ExpDecayVal
    GCparam.LvlEst.ERBspace1 = GCresp.ERBspace1
    GCparam.LvlEst.NchShift = NchShift
    GCparam.LvlEst.NchLvlEst = NchLvlEst
    GCparam.LvlEst.LvlLinMinLim = LvlLinMinLim
    GCparam.LvlEst.LvlLinRef = LvlLinRef

    return GCparam, GCresp
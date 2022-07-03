# -*- coding: utf-8 -*-
import numpy as np
import sys
import utils


class GCresp:
    def __init__(self):
        self.Fr1 = []
        self.Fr2 = []
        self.ERBspace1 = []
        self.Ef = []
        self.b1val = []
        self.c1val = []
        self.Fp1 = []
        self.Fp2 = []
        self.b2val = []
        self.c1val = []
        self.fratVal = []

class LvlEst:
    def __init__(self):
        self.LctERB = []
        self.DecayHL = []
        self.b2 = []
        self.c2 = []
        self.frat = []
        self.RMStoSPLdB = []
        self.Weight = []
        self.RefdB = []
        self.Pwr = []
        self.ExpDecayVal = []
        self.NchShift = []
        self.NchLvlEst = []
        self.LvlLinMinLim = []
        self.LvlLinRef = []


def set_param(gc_param):
    """Setting Default Parameters for GCFBv2

    Args:
        gc_param (struct): Your preset gammachirp parameters
            .fs: Sampling rate (default: 48000)
            .NumCh: Number of Channels (default: 100)
            .FRange: Frequency Range of GCFB (default: [100, 6000]) 
                     specifying asymtopic freq. of passive GC (Fr1)

    Returns:
        gc_param (struct): gc_param values
    """
    if not hasattr(gc_param, 'fs'):
        gc_param.fs = 48000

    if not hasattr(gc_param, 'OutMidCrct'):
        gc_param.OutMidCrct = 'ELC'
        # if no OutMidCrct is not necessary, specify gc_param.OutMidCrct = 'no'

    if not hasattr(gc_param, 'NumCh'):
        gc_param.NumCh = 100

    if not hasattr(gc_param, 'FRange'):
        gc_param.FRange = np.array([100, 6000])
    
    # Gammachirp parameters
    if not hasattr(gc_param, 'n'):
        gc_param.n = 4 # default gammatone & gammachirp

    # Convention
    if not hasattr(gc_param, 'b1'):
        gc_param.b1 = np.array([1.81, 0]) # b1 becomes two coeffs in v210 (18 Apr. 2015). Frequency independent by 0. 

    if len(gc_param.b1) == 1:
        gc_param.b1.append(0) # frequency independent by 0

    if not hasattr(gc_param, 'c1'):
        gc_param.c1 = np.array([-2.96, 0]) # c1 becomes two coeffs. in v210 (18 Apr. 2015). Frequency independent by 0.

    if len(gc_param.c1) == 1:
        gc_param.c1.append(0) # frequency independent by 0
    
    if not hasattr(gc_param, 'frat'):
        gc_param.frat = np.array([[0.4660, 0], [0.0109, 0]])
    
    if not hasattr(gc_param, 'b2'):
        gc_param.b2 = np.array([[2.17, 0], [0, 0]]) # no level-dependency  (8 Jul 05)

    if not hasattr(gc_param, 'c2'):
        gc_param.c2 = np.array([[2.20, 0], [0, 0]]) # no level-dependency; no freq-dependency (3 Jun 05)

    if not hasattr(gc_param, 'Ctrl'):
        gc_param.Ctrl = 'static'      

    if not hasattr(gc_param, 'Ctrl'):
        gc_param.Ctrl = 'static'
    if 'fix' in gc_param.Ctrl:
        gc_param.Ctrl = 'static'
    if 'tim' in gc_param.Ctrl:
        gc_param.Ctrl = 'dynamic'
    if not 'sta' in gc_param.Ctrl and not 'dyn' in gc_param.Ctrl and not 'lev' in gc_param.Ctrl:
        print("Specify gc_param.Ctrl:  'static', 'dynamic', or 'level(-estimation). \
               (old version 'fixed'/'time-varying')", file=sys.stderr)
        sys.exit(1)

    if not hasattr(gc_param, 'GainCmpnstdB'):
        gc_param.GainCmpnstdB = -1 # in dB. when LvlEst.c2==2.2, 1 July 2005

    """
    Parameters for level estimation
    """
    if hasattr(gc_param, 'PpgcRef') or hasattr(gc_param, 'LvlRefdB'):
        print("The parameter 'gc_param.PpgcRef' is obsolete.")
        print("The parameter 'gc_param.LvlRefdB' is obsolete.")
        print("Please change it to 'gc_param.GainRefdB'", file=sys.stderr)
        sys.exit(1)
    
    if not hasattr(gc_param, 'GainRefdB'):
        gc_param.GainRefdB = 50 # reference Ppgc level for gain normalization

    if not hasattr(gc_param, 'LeveldBscGCFB'):
        gc_param.LeveldBscGCFB = gc_param.GainRefdB # use it as default

    if not hasattr(gc_param, 'LvlEst'):
        gc_param.LvlEst = LvlEst()

    if len(gc_param.LvlEst.LctERB) == 0:
        #gc_param.LvlEst.LctERB = 1.0
        # Location of Level Estimation pGC relative to the signal pGC in ERB
        # see testGC_LctERB.py for fitting result. 10 Sept 2004
        gc_param.LvlEst.LctERB = 1.5;   # 16 July 05

    if len(gc_param.LvlEst.DecayHL) == 0:
        gc_param.LvlEst.DecayHL = 0.5; # 18 July 2005

    if len(gc_param.LvlEst.b2) == 0:
        gc_param.LvlEst.b2 = gc_param.b2[0, 0]

    if len(gc_param.LvlEst.c2) == 0:
        gc_param.LvlEst.c2 = gc_param.c2[0, 0]

    if len(gc_param.LvlEst.frat) == 0:
        # gc_param.LvlEst.frat = 1.1 #  when b=2.01 & c=2.20
        gc_param.LvlEst.frat = 1.08 # peak of cGC ~= 0 dB (b2=2.17 & c2=2.20)

    if len(gc_param.LvlEst.RMStoSPLdB) == 0:
        gc_param.LvlEst.RMStoSPLdB = 30 # 1 rms == 30 dB SPL for Meddis IHC

    if len(gc_param.LvlEst.Weight) == 0:
        gc_param.LvlEst.Weight = 0.5

    if len(gc_param.LvlEst.RefdB) == 0:
        gc_param.LvlEst.RefdB = 50 # 50 dB SPL

    if len(gc_param.LvlEst.Pwr) == 0:
        gc_param.LvlEst.Pwr = np.array([1.5, 0.5]) # Weight for pGC & cGC

    # new 19 Dec 2011
    if not hasattr(gc_param, 'NumUpdateAsymCmp'):
        # gc_param.NumUpdateAsymCmp = 3 # updte every 3 samples (== 3*GCFBv207)
        gc_param.NumUpdateAsymCmp = 1 # samply-by-sample (==GCFBv207)

    """
    GCresp
    """
    gc_resp = GCresp()
    
    Fr1, ERBrate1 = utils.equal_freq_scale('ERB', gc_param.NumCh, gc_param.FRange)
    gc_resp.Fr1 = np.array([Fr1]).T
    gc_resp.ERBspace1 = np.mean(np.diff(ERBrate1))
    ERBrate, ERBw = utils.freq2erb(gc_resp.Fr1)
    ERBrate1kHz, ERBw1kHz = utils.freq2erb(1000)
    gc_resp.Ef = ERBrate/ERBrate1kHz - 1

    OneVec = np.ones([gc_param.NumCh, 1])
    gc_resp.b1val = gc_param.b1[0]*OneVec + gc_param.b1[1]*gc_resp.Ef
    gc_resp.c1val = gc_param.c1[0]*OneVec + gc_param.c1[1]*gc_resp.Ef

    gc_resp.Fp1, _ = utils.fr2fpeak(gc_param.n, gc_resp.b1val, gc_resp.c1val, gc_resp.Fr1)
    gc_resp.Fp2 = np.zeros(np.shape(gc_resp.Fp1))

    gc_resp.b2val = gc_param.b2[0, 0]*OneVec + gc_param.b2[0, 1]*gc_resp.Ef
    gc_resp.c2val = gc_param.c2[0, 0]*OneVec + gc_param.c2[0, 1]*gc_resp.Ef
    
    """
    Set Params estimation circuit
    """
    # keep LvlEst params  3 Dec 2013
    ExpDecayVal = np.exp(-1/(gc_param.LvlEst.DecayHL*gc_param.fs/1000)*np.log(2)) # decay exp
    NchShift = np.round(gc_param.LvlEst.LctERB/gc_resp.ERBspace1)
    NchLvlEst = np.minimum(np.maximum(1, np.array([np.arange(gc_param.NumCh)+1]).T+NchShift), \
                           gc_param.NumCh) # shift in NumCh [1:NumCh]
    LvlLinMinLim = 10**(-gc_param.LvlEst.RMStoSPLdB/20) # minimum sould be SPL 0 dB
    LvlLinRef = 10**((gc_param.LvlEst.RefdB - gc_param.LvlEst.RMStoSPLdB)/20)

    gc_param.LvlEst.ExpDecayVal = ExpDecayVal
    gc_param.LvlEst.ERBspace1 = gc_resp.ERBspace1
    gc_param.LvlEst.NchShift = NchShift
    gc_param.LvlEst.NchLvlEst = NchLvlEst
    gc_param.LvlEst.LvlLinMinLim = LvlLinMinLim
    gc_param.LvlEst.LvlLinRef = LvlLinRef

    return gc_param, gc_resp
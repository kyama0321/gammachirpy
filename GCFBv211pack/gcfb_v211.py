# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
from scipy import signal
import matplotlib.pyplot as plt

import utils
import gcfb_v211_set_param as gcfb_set_param
import gammachirp as gcfb


class classACFstatus:
    def __init__(self):
        self.NumCh = []
        self.NumFilt = []
        self.Lbz = []
        self.Lap = []
        self.SigInPrev = []
        self.SigOutPrev = []
        self.Count = []

class ACFcoef:
    def __init__(self):
        self.fs = []
        self.ap = np.array([])
        self.bz = np.array([])


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
    
    # Call default parametes
    gc_param, gc_resp = gcfb_set_param.set_param(gc_param)
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
        cmpn_out_mid, _ = utils.out_mid_crct_filt(gc_param.OutMidCrct, fs, 0, 2) # 2) minimum phase
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
        acf_coef_fast_prcs = make_asym_cmp_filters_v2(fs, fr2val, gc_resp.b2val, gc_resp.c2val)
    else:
        # HP-AF for dynamic-GC level estimation path. 18 Dec 2012 Checked
        fr2lvl_est = gc_param.LvlEst.frat * gc_resp.Fp1
        # default gc_param.LvlEst.frat = 1.08 (GCFBv208_SetParam(gc_param))
        # ---> Linear filter for level estimation
        acf_coef_lvl_est = make_asym_cmp_filters_v2(fs,fr2lvl_est,gc_param.LvlEst.b2, gc_param.LvlEst.c2)

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
        pgc, _, _, _ = gcfb.gammachirp(gc_resp.Fr1[nch], fs, gc_param.n, \
                                       gc_resp.b1val[nch], gc_resp.c1val[nch], 0, '', 'peak')

        # fast FFT-based filtering by the pgc
        pgc_out[nch, 0:len_snd] = utils.fftfilt(pgc[0,:], Snd) 

        # Fast processing for fixed cGC
        if sw_fast_prcs == 1 and gc_param.Ctrl == 'static': # Static
            str_gc = "Static (Fixed) Compressive-Gammachirp"
            gc_out1 = pgc_out[nch, :].copy()
            for n_filt in range(4):
                gc_out1 = signal.lfilter(acf_coef_fast_prcs.bz[nch, :, n_filt], \
                                         acf_coef_fast_prcs.ap[nch, :, n_filt], gc_out1)

            cgc_out[nch, :] = gc_out1.copy()
            gc_resp.Fp2[nch], _ = fr1_to_fp2(gc_param.n, gc_resp.b1val[nch], gc_resp.c1val[nch], \
                                                 gc_resp.b2val[nch], gc_resp.c2val[nch], \
                                                 fratVal[nch], gc_resp.Fr1[nch])
            if nch == num_ch:
                gc_resp.Fp2 = gc_resp.Fp2

        else: # Level estimation pass for Dynamic.
            str_gc = "Passive-Gammachirp & Level estimation filter"
            gc_out1 = pgc_out[nch, :].copy()
            for n_filt in range(4):
                gc_out1 = signal.lfilter(acf_coef_lvl_est.bz[nch, :, n_filt], \
                                         acf_coef_lvl_est.ap[nch, :, n_filt], gc_out1)
            cgc_out_lvl_est[nch, :] = gc_out1.copy()

        if nch == 0 or np.mod(nch+1, 20) == 0: # "rem" is not defined in the original code! 
        # if nch == 0:
            t_now = time.time()
            print(str_gc + f" ch #{nch+1}" + f" / #{num_ch}.   " \
                  + f"elapsed time = {np.round(t_now-t_start, 1)} (sec)")

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
                acf_coef = make_asym_cmp_filters_v2(fs, fr2val, gc_resp.b2val, gc_resp.c2val)

            if nsmpl == 0:
                _, acf_status = acfilterbank(acf_coef, []) # initialization

            sig_out, acf_status = acfilterbank(acf_coef, acf_status, pgc_out[:, nsmpl])
            cgc_out[:, [nsmpl]] = sig_out.copy()
            gc_resp.Fr2[:, [nsmpl]] = fr2val.copy()
            gc_resp.fratVal[:, [nsmpl]] = fratVal.copy()

            if nsmpl == 0 or np.mod(nsmpl+1, num_disp) == 0:
                t_now = time.time()
                print(f"Dynamic Compressive-Gammachirp: Time {np.round(nsmpl/fs*1000, 1)} (ms) / "\
                      + f"{np.round(len_snd/fs*1000, 1)} (ms). elapsed time = {np.round(t_now-t_start, 1)} (sec)")

        """
        End of Dynamic Compressive Gammachirp filtering
        """
        """
        Signal path Gain Normalization at Reference Level (GainRefdB) for static dynamic filters
        """
        fratRef = gc_param.frat[0, 0] + gc_param.frat[0, 1] * gc_resp.Ef[:] \
            + (gc_param.frat[1, 0] + gc_param.frat[1, 1] * gc_resp.Ef[:]) * gc_param.GainRefdB

        cgc_ref = cmprs_gc_frsp(gc_resp.Fr1, fs, gc_param.n, gc_resp.b1val, \
                                      gc_resp.c1val, fratRef, gc_resp.b2val, gc_resp.c2val)
        gc_resp.cGCRef = cgc_ref
        gc_resp.LvldB = lvl_db

        gc_resp.GainFactor = 10**(gc_param.GainCmpnstdB/20) * cgc_ref.NormFctFp2
        cgc_out = (gc_resp.GainFactor * np.ones((1, len_snd))) * cgc_out

    return cgc_out, pgc_out, gc_param, gc_resp


def make_asym_cmp_filters_v2(fs, frs, b, c):
    """Computes the coefficients for a bank of Asymmetric Compensation Filters
    This is a modified version to fix the round off problem at low freqs
    Use this with ACFilterBank.m
    See also asym_cmp_frsp_v2 for frequency response

    Args:
        fs (int): Sampling frequency
        frs (array_like): array of the center frequencies, frs
        b (array_like): array or scalar of a bandwidth coefficient, b
        c (float): array or scalar of asymmetric parameters, c

    Returns:
        ACFcoef: 
        - fs (int): Sampling frequency
        - bz (array_like): MA coefficients  (num_ch*3*num_filt)
        - ap (array_like): AR coefficients  (num_ch*3*num_filt)

    Notes:
        [1] Ref for p1-p4: Unoki,M , Irino,T. , and Patterson, R.D. , "Improvement of an IIR asymmetric compensation gammachirp filter," Acost. Sci. & Tech. (ed. by the Acoustical Society of Japan ), 22 (6), pp. 426-430, Nov. 2001.
        [2] Conventional setting was removed.
            fn = frs + nfilt* p3 .*c .*b .*erbw/n;
            This frequency fn is for normalizing GC(=GT*Hacf) filter to be unity at the peak, frequnecy. But now we use Hacf as a highpass filter as well. cGC = pGC *Hacf. In this case, this normalization is useless. 
            So, it was set as the gain at frs is unity.  (4. Jun 2004 )
        [3] Removed
            ACFcoef.fn(:,nff) = fn;
            n : scalar of order t^(n-1) % used only in normalization 
    """
    num_ch, len_frs = np.shape(frs)
    if len_frs > 1:
        print("frs should be a column vector frs.", file=sys.stderr)
        sys.exit(1)
    
    _, erbw = utils.freq2erb(frs)

    acf_coef = ACFcoef()
    acf_coef.fs = fs

    # New coefficients. See [1]
    num_filt = 4
    p0 = 2
    p1 = 1.7818 * (1-0.0791*b) * (1-0.1655*np.abs(c))
    p2 = 0.5689 * (1-0.1620*b) * (1-0.0857*np.abs(c))
    p3 = 0.2523 * (1-0.0244*b) * (1+0.0574*np.abs(c))
    p4 = 1.0724

    if num_filt > 4:
        print("num_filt > 4", file=sys.stderr)
        sys.exit(1) 

    acf_coef.ap = np.zeros((num_ch, 3, num_filt))
    acf_coef.bz = np.zeros((num_ch, 3, num_filt))

    for nfilt in range(num_filt):
        r  = np.exp(-p1*(p0/p4)**(nfilt) * 2*np.pi*b*erbw / fs)
        del_frs = (p0*p4)**(nfilt)*p2*c*b*erbw;  
        phi = 2*np.pi*(frs+del_frs).clip(0)/fs
        psi = 2*np.pi*(frs-del_frs).clip(0)/fs
        fn = frs # see [2]

        # second order filter
        ap = np.concatenate([np.ones(np.shape(r)), -2*r*np.cos(phi), r**2], axis=1)
        bz = np.concatenate([np.ones(np.shape(r)), -2*r*np.cos(psi), r**2], axis=1)

        vwr = np.exp(1j*2*np.pi*fn/fs)
        vwrs = np.concatenate([np.ones(np.shape(vwr)), vwr, vwr**2], axis=1)
        nrm = np.array([np.abs(np.sum(vwrs*ap, axis=1) / np.sum(vwrs*bz, axis=1))]).T
        bz = bz * (nrm*np.ones((1, 3)))

        acf_coef.ap[:,:,nfilt] = ap
        acf_coef.bz[:,:,nfilt] = bz

    return acf_coef



def fr1_to_fp2(n, b1, c1, b2, c2, frat, fr1, sr=24000, n_fft=2048, sw_plot=0):
    """Convert fr1 (for passive GC; pGC) to fp2 (for compressive GC; cGC)

    Args:
        n (int): Parameter defining the envelope of the gamma distribution (for pGC)
        b1 (float): Parameter defining the envelope of the gamma distribution (for pGC)
        c1 (float): Chirp factor (for pGC)
        b2 (float): Parameter defining the envelope of the gamma distribution (for cGC)
        c2 (float): Chirp factor  (for cGC)
        frat (float): Frequency ratio, the main level-dependent variable
        fr1 (float): Center frequency (for pGC)
        sr (int, optional): Sampling rate. Defaults to 24000.
        n_fft (int, optional): Size of FFT. Defaults to 2048.
        sw_plot (int, optional): Show plot of cgc_frsp and pGC_frsp. Defaults to 0.

    Returns:
        fp2 (float): Peak frequency (for compressive GC)
        fr2 (float): Center Frequency (for compressive GC)
    """
    _, erbw1 = utils.freq2erb(fr1)
    fp1, _ = utils.fr2fpeak(n, b1, c1, fr1)
    fr2 = frat * fp1
    _, erbw2 = utils.freq2erb(fr2)

    bw1 = b1 * erbw1
    bw2 = b2 * erbw2

    # coef1*fp2^3 + coef2*fp2^2 + coef3*fp2 + coef4 = 0 
    coef1 = -n
    coef2 = c1*bw1 + c2*bw2 + n*fr1 + 2*n*fr2
    coef3 = -2*fr2*(c1*bw1+n*fr1) - n*((bw2)**2+fr2**2) - 2*c2*bw2*fr1
    coef4 = c2*bw2*((bw1)**2+fr1**2) + (c1*bw1+n*fr1)*(bw2**2+fr2**2)
    coefs = [coef1, coef2, coef3, coef4]

    p = np.roots(coefs)
    fp2cand = p[np.imag(p)==0]
    if len(fp2cand) == 1:
        fp2 = fp2cand
    else:
        val, ncl = np.min(np.abs(fp2cand - fp1))
        fp2 = fp2cand(ncl) # in usual cGC range, fp2 is close to fp1

    # sw_plot = 1
    if sw_plot == 1: # Check
        fs = 48000
        n_frq_rsl = 2048
        cgc_rsp = cmprs_gc_frsp(fr1, fs, n, b1, c1, frat, b2, c2, n_frq_rsl)

        nfr2 = np.zeros((len(fp2cand), 1))
        for nn in range(len(fp2cand)):
            nfr2[nn] = np.argmin(abs(cgc_rsp.freq - fp2cand[nn]))
        
        fig, ax = plt.subplots()
        plt_freq = np.array(cgc_rsp.freq).T
        plt_cgc_frsp = np.array(cgc_rsp.cGCFrsp/np.max(cgc_rsp.cGCFrsp)).T
        plt_pgc_frsp = np.array(cgc_rsp.pGCFrsp).T

        ax.plot(plt_freq, plt_cgc_frsp, label="cGCFrsp") # compressive GC
        ax.plot(plt_freq, plt_pgc_frsp, label="pGCFrsp") # passive GC
        ax.set_xlim([0, np.max(fp2cand)*2])
        ax.set_ylim([0, 1])
        ax.legend()
        plt.show()

    return fp2, fr2


def cmprs_gc_frsp(fr1, fs=48000, n=4, b1=1.81, c1=-2.96, frat=1, b2=2.01, c2=2.20, n_frq_rsl=1024):
    """Frequency Response of Compressive GammaChirp

    Args:
        fr1 (array-like): Resonance Freqs.
        fs (int, optional): Sampling Freq. Defaults to 48000.
        n (int, optional): Order of Gamma function, t**(n-1). Defaults to 4.
        b1 (float, optional): b1 for exp(-2*pi*b1*erb(f)). Defaults to 1.81.
        c1 (float, optional): c1 for exp(j*2*pi*fr + c1*ln(t)). Defaults to -2.96.
        frat (int, optional): Frequency ratio. fr2 = frat*fp1. Defaults to 1.
        b2 (float, optional): _description_. Defaults to 2.01.
        c2 (float, optional): _description_. Defaults to 2.20.
        n_frq_rsl (int, optional): _description_. Defaults to 1024.

    Returns:
        cGCresp: Struct of cGC response
            pGCFrsp (array-like): Passive GC freq. resp. (num_ch*n_frq_rsl matrix)
            cGCFrsp (array-like): Comressive GC freq. resp. (num_ch*n_frq_rsl matrix)
            cGCNrmFrsp (array-like): Normalized cGCFrsp (num_ch*n_frq_rsl matrix)
            ACFrsp: Asym (array-like). Compensation Filter freq. resp.
            AsymFunc (array-like): Asym Func
            freq (array-like): Frequency (1*n_frq_rsl)
            Fp2 (array-like): Peak freq.
            ValFp2 (array-like): Peak Value
    """
    class cGCresp:
        Fr1 = []
        n = []
        b1 = []
        c1 = []
        frat = []
        b2 = []
        c2 = []
        NfrqRsl = []
        pGCFrsp = []
        cGCFrsp = []
        cGCNrmFrsp = []
        ACFFrsp = []
        AsymFunc = []
        Fp1 = []
        Fr2 = []
        Fp2 = []
        ValFp2 = []
        NormFctFp2 = []
        freq = []

    if utils.isrow(fr1):
        fr1 = np.array([fr1]).T

    num_ch = len(fr1)

    if isinstance(n, (int, float)):
        n = n * np.ones((num_ch, 1))
    if isinstance(b1, (int, float)):
        b1 = b1 * np.ones((num_ch, 1))
    if isinstance(c1, (int, float)):
        c1 = c1 * np.ones((num_ch, 1))
    if isinstance(frat, (int, float)):
        frat = frat * np.ones((num_ch, 1))
    if isinstance(b2, (int, float)):
        b2 = b2 * np.ones((num_ch, 1))
    if isinstance(c2, (int, float)):
        c2 = c2 * np.ones((num_ch, 1))

    pgc_frsp, freq, _, _, _ = gammachirp_frsp(fr1, fs, n, b1, c1, 0.0, n_frq_rsl)
    fp1, _ = utils.fr2fpeak(n, b1, c1, fr1)
    fr2 = frat * fp1
    acf_frsp, freq, asym_func = asym_cmp_frsp_v2(fr2, fs, b2, c2, n_frq_rsl)
    cgc_frsp = pgc_frsp * asym_func # cgc_frsp = pgc_frsp * acf_frsp
    
    val_fp2 = np.max(cgc_frsp, axis=1)
    nchFp2 = np.argmax(cgc_frsp, axis=1)
    if utils.isrow(val_fp2):
        val_fp2 = np.array([val_fp2]).T
    
    norm_fact_fp2 = 1/val_fp2

    # function cGCresp = CmprsGCFrsp(fr1,fs,n,b1,c1,frat,b2,c2,n_frq_rsl)
    cGCresp.Fr1 = fr1
    cGCresp.n = n
    cGCresp.b1 = b1
    cGCresp.c1 = c1
    cGCresp.frat = frat
    cGCresp.b2 = b2
    cGCresp.c2 = c2
    cGCresp.NfrqRsl = n_frq_rsl
    cGCresp.pGCFrsp = pgc_frsp
    cGCresp.cGCFrsp = cgc_frsp
    cGCresp.cGCNrmFrsp = cgc_frsp * (norm_fact_fp2 * np.ones((1,n_frq_rsl)))
    cGCresp.ACFFrsp = acf_frsp
    cGCresp.AsymFunc = asym_func
    cGCresp.Fp1 = fp1
    cGCresp.Fr2 = fr2
    cGCresp.Fp2 = freq[nchFp2]
    cGCresp.ValFp2 = val_fp2
    cGCresp.NormFctFp2 = norm_fact_fp2
    cGCresp.freq = [freq]

    return cGCresp


def gammachirp_frsp(frs, sr=48000, order_g=4, coef_erbw=1.019, coef_c=0.0, phase=0.0, n_frq_rsl=1024):
    """Frequency Response of GammaChirp

    Args:
        frs (array_like, optional): Resonance freq. Defaults to None.
        sr (int, optional): Sampling freq. Defaults to 48000.
        order_g (int, optional): Order of Gamma function t**(order_g-1). Defaults to 4.
        coef_erbw (float, optional): Coeficient -> exp(-2*pi*coef_erbw*ERB(f)). Defaults to 1.019.
        coef_c (int, optional): Coeficient -> exp(j*2*pi*Fr + coef_c*ln(t)). Defaults to 0.0.
        phase (int, optional): Coeficient -> exp(j*2*pi*Fr + coef_c*ln(t)). Defaults to 0.9.
        n_frq_rsl (int, optional): Freq. resolution. Defaults to 1024.

    Returns:
        amp_frsp (array_like): Absolute of freq. resp. (num_ch*n_frq_rsl matrix)
        freq (array_like): Frequency (1*n_frq_rsl)
        f_peak (array_like): Peak frequency (num_ch * 1)
        grp_dly (array_like): Group delay (num_ch*n_frq_rsl matrix)
        phs_frsp (array_like): Angle of freq. resp. (num_ch*n_frq_rsl matrix)
    """
    if utils.isrow(frs):
        frs = np.array([frs]).T

    num_ch = len(frs)

    if isinstance(order_g, (int, float)) or len(order_g) == 1:
        order_g = order_g * np.ones((num_ch, 1))
    if isinstance(coef_erbw, (int, float)) or len(coef_erbw) == 1:
        coef_erbw = coef_erbw * np.ones((num_ch, 1))
    if isinstance(coef_c, (int, float)) or len(coef_c) == 1:
        coef_c = coef_c * np.ones((num_ch, 1))
    if isinstance(phase, (int, float)) or len(phase) == 1:
        phase = phase * np.ones((num_ch, 1))

    if n_frq_rsl < 256:
        print("n_frq_rsl < 256", file=sys.stderr)
        sys.exit(1)

    _, erbw = utils.freq2erb(frs)
    freq = np.arange(n_frq_rsl) / n_frq_rsl * sr / 2
    freq = np.array([freq]).T

    one1 = np.ones((1, n_frq_rsl))
    bh = (coef_erbw * erbw) * one1
    fd = (np.ones((num_ch, 1)) * freq[:,0]) - frs * one1
    cn = (coef_c / order_g) * one1
    n = order_g * one1
    c = coef_c * one1
    phase = phase * one1

    # Analytic form (normalized at f_peak)
    amp_frsp = ((1+cn**2) / (1+(fd/bh)**2))**(n/2) \
                * np.exp(c * (np.arctan(fd/bh)-np.arctan(cn)))
    
    f_peak = frs + coef_erbw * erbw * coef_c / order_g
    grp_dly = 1/(2*np.pi) * (n*bh + c*fd) / (bh**2 + fd**2)
    phs_frsp = -n * np.arctan(fd/bh) - c / 2*np.log((2*np.pi*bh)**2 + (2*np.pi*fd)**2) + phase

    return amp_frsp, freq, f_peak, grp_dly, phs_frsp
    

def asym_cmp_frsp_v2(frs, fs=48000, b=None, c=None, n_frq_rsl=1024, num_filt=4):
    """Amplitude spectrum of Asymmetric compensation IIR filter (ACF) for the gammachirp 
    corresponding to make_asym_cmp_filters_v2

    Args:
        frs (array_like, optional): Center freqs. Defaults to None.
        fs (int, optional): Sampling freq. Defaults to 48000.
        b (array_like, optional): Bandwidth coefficient. Defaults to None.
        c (array_like, optional): Asymmetric paramters. Defaults to None.
        n_frq_rsl (int, optional): Freq. resolution for linear freq. scale for specify renponse at frs
                                (n_frq_rsl>64). Defaults to 1024.
        num_filt (int, optional): Number of 2nd-order filters. Defaults to 4.

    Returns:
        acf_frsp: Absolute values of frequency response of ACF (num_ch * n_frq_rsl)
        freq: freq. (1 * n_frq_rsl)
        asym_func: Original asymmetric function (num_ch * n_frq_rsl)
    """
    if utils.isrow(frs):
        frs = np.array([frs]).T
    if utils.isrow(b):
        b = np.array([b]).T
    if utils.isrow(c):
        c = np.array([c]).T
    num_ch = len(frs)

    if n_frq_rsl >= 64:
        freq = np.arange(n_frq_rsl) / n_frq_rsl * fs/2
    elif n_frq_rsl == 0:
        freq = frs
        n_frq_rsl = len(freq)
    else:
        help(asym_cmp_frsp_v2)
        print("Specify n_frq_rsl 0) for frs or N>=64 for linear-freq scale", file=sys.stderr)
        sys.exit(1)

    # coef.
    sw_coef = 0 # self consistency
    # sw_coef = 1 # reference to make_asym_cmp_filters_v2

    if sw_coef == 0:
        # New Coefficients. num_filt = 4; See [1]
        p0 = 2
        p1 = 1.7818 * (1 - 0.0791*b) * (1 - 0.1655*np.abs(c))
        p2 = 0.5689 * (1 - 0.1620*b) * (1 - 0.0857*np.abs(c))
        p3 = 0.2523 * (1 - 0.0244*b) * (1 + 0.0574*np.abs(c))
        p4 = 1.0724
    else:
        ACFcoef = make_asym_cmp_filters_v2(fs, frs, b, c)

    # filter coef.
    _, erbw = utils.freq2erb(frs)
    acf_frsp = np.ones((num_ch, n_frq_rsl))
    freq2 = np.concatenate([np.ones((num_ch,1))*freq, frs], axis=1)

    for nfilt in range(num_filt):

        if sw_coef == 0:
            r = np.exp(-p1 * (p0/p4)**nfilt * 2 * np.pi * b * erbw / fs)
            delfr = (p0*p4)**nfilt * p2 * c * b * erbw
            phi = 2*np.pi*np.maximum(frs + delfr, 0)/fs
            psi = 2*np.pi*np.maximum(frs - delfr, 0)/fs
            fn = frs
            ap = np.concatenate([np.ones((num_ch, 1)), -2*r*np.cos(phi), r**2], axis=1)
            bz = np.concatenate([np.ones((num_ch, 1)), -2*r*np.cos(psi), r**2], axis=1)
        else:
            ap = ACFcoef.ap[:, :, nfilt]
            bz = ACFcoef.bz[:, :, nfilt]

        cs1 = np.cos(2*np.pi*freq2/fs)
        cs2 = np.cos(4*np.pi*freq2/fs)
        bzz0 = np.array([bz[:, 0]**2 + bz[:, 1]**2 + bz[:, 2]**2]).T * np.ones((1, n_frq_rsl+1))
        bzz1 = np.array([2 * bz[:, 1] * (bz[:, 0] + bz[:, 2])]).T * np.ones((1, n_frq_rsl+1))
        bzz2 = np.array([2 * bz[:, 0] * bz[:, 2]]).T * np.ones((1, n_frq_rsl+1))
        hb = bzz0 + bzz1*cs1 + bzz2*cs2

        app0 = np.array([ap[:, 0]**2 + ap[:, 1]**2 + ap[:, 2]**2]).T * np.ones((1, n_frq_rsl+1))
        app1 = np.array([2 * ap[:, 1] * (ap[:, 0] + ap[:, 2])]).T * np.ones((1, n_frq_rsl+1))
        app2 = np.array([2 * ap[:, 0] * ap[:, 2]]).T * np.ones((1, n_frq_rsl+1))
        ha = app0 + app1*cs1 + app2*cs2

        h = np.sqrt(hb/ha)
        h_norm = np.array([h[:, n_frq_rsl]]).T * np.ones((1, n_frq_rsl)) # Normalizatoin by fn value

        acf_frsp = acf_frsp * h[:,0:n_frq_rsl] / h_norm

    # original Asymmetric Function without shift centering
    fd = np.ones((num_ch, 1))*freq - frs*np.ones((1, n_frq_rsl))
    be = (b * erbw) * np.ones((1, n_frq_rsl))
    cc = (c * np.ones((num_ch, 1)) * np.ones((1, n_frq_rsl))) # in case when c is scalar
    asym_func = np.exp(cc * np.arctan2(fd, be))

    return acf_frsp, freq, asym_func


def acfilterbank(ACFcoef, ACFstatus, sig_in=[], sw_ordr=0):
    """IIR ACF time-slice filtering for time-varing filter

    Args:
        ACFcoef (structure): ACFcoef: coef from make_asym_cmp_filters_v2
            ap: AR coefficents (==a ~= pole) num_ch*lap*num_filt
            fs : sampling rate  (also switch for verbose)
                (The variables named 'a' and 'b' are not used to avoid the
                confusion to the gammachirp parameters.)
            verbose : Not specified) quiet   1) verbose
        ACFstatus (structure):
            NumCh: Number of channels (Set by initialization
            Lbz: size of MA
            Lap: size of AR
            NumFilt: Length of filters
            SigInPrev: Previous status of sig_in
            SigOutPrev: Previous status of SigOut
        sig_in (array_like, optional): Input signal. Defaults to [].
        sw_ordr (int, optional): Switch filtering order. Defaults to 0.

    Returns:
        SigOut (array_like): Filtered signal (num_ch * 1)
        ACFstatus: Current status
    """    
    if len(sig_in) == 0 and len(ACFstatus) != 0:
        help(acfilterbank)
        sys.exit()

    if not hasattr(ACFstatus, 'NumCh'):
        ACFstatus = classACFstatus()

        num_ch, lbz, num_filt = np.shape(ACFcoef.bz)
        num_ch, lap, _ = np.shape(ACFcoef.ap)

        if lbz != 3 or lap !=3:
            print("No gaurantee for usual IIR filters except for AsymCmpFilter.\n"\
                + "Please check make_asym_cmp_filters_v2.")
    
        ACFstatus.NumCh = num_ch
        ACFstatus.NumFilt = num_filt
        ACFstatus.Lbz = lbz # size of MA
        ACFstatus.Lap = lap # size of AR
        ACFstatus.SigInPrev = np.zeros((num_ch, lbz))
        ACFstatus.SigOutPrev = np.zeros((num_ch, lap, num_filt))
        ACFstatus.Count = 0
        print("ACFilterBank: Initialization of ACFstatus")
        sig_out = []

        return sig_out, ACFstatus
    
    if utils.isrow(sig_in):
        sig_in = np.array([sig_in]).T
    
    num_ch_sig, len_sig = np.shape(sig_in)
    if len_sig != 1:
        print("Input signal sould be num_ch*1 vector (1 sample time-slice)", file=sys.stderr)
        sys.exit(1)
    if num_ch_sig != ACFstatus.NumCh:
        print(f"num_ch_sig ({num_ch_sig}) != ACFstatus.NumCh ({ACFstatus.NumCh})")

    # time stamp
    if hasattr(ACFcoef, 'verbose'):
        if ACFcoef.verbose == 1: # verbose when ACFcoef.verbose is specified to 1
            t_disp = 50 # ms
            t_cnt = ACFstatus.Count/(np.fix(ACFcoef.fs/1000)) # ms

            if ACFstatus.Count == 0:
                print("ACFilterBank: Start processing")
                tic = time.time()

            elif np.mod(t_cnt, t_disp) == 0:
                toc = time.time()
                print(f"ACFilterBank: Processed {t_cnt} (ms)." \
                      + f"elapsed Time = {np.round(tic-toc, 1)} (sec)")
    
    ACFstatus.Count = ACFstatus.Count+1
    
    """
    Processing
    """
    ACFstatus.SigInPrev = np.concatenate([ACFstatus.SigInPrev[:, 1:ACFstatus.Lbz], sig_in], axis=1)

    x = ACFstatus.SigInPrev.copy()
    nfilt_list = np.arange(ACFstatus.NumFilt)

    if sw_ordr == 1:
        nfilt_list = np.flip(nfilt_list)

    for nfilt in nfilt_list:

        forward = ACFcoef.bz[:, ACFstatus.Lbz::-1, nfilt] * x
        feedback = ACFcoef.ap[:, ACFstatus.Lap:0:-1, nfilt] * \
            ACFstatus.SigOutPrev[:, 1:ACFstatus.Lap, nfilt]

        fwdSum = np.sum(forward, axis=1)
        fbkSum = np.sum(feedback, axis=1)

        y = np.array([(fwdSum - fbkSum) / ACFcoef.ap[:, 0, nfilt]]).T
        ACFstatus.SigOutPrev[:, :, nfilt] = \
            np.concatenate([ACFstatus.SigOutPrev[:, 1:ACFstatus.Lap, nfilt], y], axis=1)
        x = ACFstatus.SigOutPrev[:, :, nfilt].copy()

    sig_out = y

    return sig_out, ACFstatus
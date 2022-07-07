# -*- coding: utf-8 -*-
from ctypes import util
import sys
import numpy as np
from scipy import signal

import utils


def gammachirp(frs, sr, order_g=4, coef_erbw=1.019, coef_c=0, phase=0, sw_carr='cos', sw_norm='no'):
    """Gammachirp : Theoretical auditory filter
    gc(t) = t^(n-1) exp(-2 pi b ERB(frs)) cos(2*pi*frs*t + c ln t + phase)

    Args:
        frs (array_like): Asymptotic Frequency
        sr (int): Sampling frequency
        order_g (int, optional): Order of Gamma function t^(order_g-1), n. Defaults to 4.
        coef_erbw (float, optional): Coeficient -> exp(-2*pi*coef_erbw*ERB(f)), b. Defaults to 1.019.
        coef_c (int, optional): Coeficient -> exp(j*2*pi*frs + coef_c*ln(t)), c. Defaults to 0.
        phase (int, optional): Start phase(0 ~ 2*pi). Defaults to 0.
        sw_carr (str, optional): Carrier ('cos','sin','complex','envelope' with 3 letters). Defaults to 'cos'.
        sw_norm (str, optional): Normalization of peak spectrum level ('no', 'peak'). Defaults to 'no'.

    Returns:
        gc (array_like): GammaChirp
        len_gc (array_like): Length of gc for each channel
        fps (array_like): Peak frequency
        inst_freq (array_like): Instanteneous frequency
    """    

    if len(sw_carr) == 0:
        sw_carr = 'cos'

    num_ch = len(frs)
    order_g = order_g * np.ones((num_ch, 1))
    coef_erbw = coef_erbw * np.ones((num_ch, 1))
    coef_c = coef_c * np.ones((num_ch, 1))
    phase = phase * np.ones((num_ch, 1))

    _, erbw = utils.freq2erb(frs)
    len_gc_1khz = (40*max(order_g)/max(coef_erbw) + 200) * sr/16000
    _, erbw_1khz = utils.freq2erb(1000)

    if sw_carr == 'sin':
        phase = phase - np.pi/2*np.ones((1,num_ch))

    # Phase compensation
    phase = phase + coef_c * np.log(frs/1000) # relative phase to 1kHz

    len_gc = np.fix(len_gc_1khz*erbw_1khz/erbw)

    """
    Production of GammaChirp
    """
    gc = np.zeros((num_ch, int(max(len_gc))))

    fps = fr2fpeak(order_g, coef_erbw, coef_c, frs) # Peak freq.
    inst_freq = np.zeros((num_ch, int(max(len_gc))))

    for nch in range(num_ch):
        t = np.arange(1, int(len_gc[nch]))/sr

        gamma_env = t**(order_g[nch]-1) * np.exp(-2*np.pi*coef_erbw[nch]*erbw[nch]*t)
        gamma_env = np.array([0] + list(gamma_env/max(gamma_env)))

        if sw_carr == 'env': # envelope
            carrier = np.ones(np.shape(gamma_env))
        elif sw_carr == 'com': # complex
            carrier = np.array([0] + list(np.exp(1j*(2*np.pi*frs[nch]*t + coef_c[nch]*np.log(t) + phase[nch]))))
        else:
            carrier = np.array([0] + list(np.cos(2*np.pi*frs[nch]*t + coef_c[nch]*np.log(t) + phase[nch])))
        
        gc[nch, 0: int(len_gc[nch])] = gamma_env * carrier

        inst_freq[nch, 0: int(len_gc[nch])] = np.array([0] + list(frs[nch] + list(coef_c[nch]/(2*np.pi*t))))

        if sw_norm == 'peak': # peak gain normalization
            freq, frsp = signal.freqz(gc[nch, 0: int(len_gc[nch])], 1, 2**utils.nextpow2(int(len_gc[nch])), fs=sr)
            fp, _ = fr2fpeak(order_g[nch], coef_erbw[nch], coef_c[nch], frs[nch])
            npeak = np.argmin(np.abs(freq - fp))
            gc[nch, :] = gc[nch, :] / np.abs(frsp[npeak])

    return gc, len_gc, fps, inst_freq


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


def fr2fpeak(n, b, c, fr):
    """Estimate fpeak from fr

    Args:
        n (float): a parameter of the gammachirp
        b (float): a parameter of the gammachirp
        c (float): a parameter of the gammachirp
        fr (float): fr

    Returns:
        fpeak (float): peak frequency
        erbw (float): erbwidth at fr
    """
    _, erb_width = utils.freq2erb(fr)
    fpeak = fr + c*erb_width*b/n

    return fpeak, erb_width


def fp2_to_fr1(n, b1, c1, b2, c2, frat, fp2):
    """_summary_

    Args:
        n (_type_): _description_
        b1 (_type_): _description_
        c1 (_type_): _description_
        b2 (_type_): _description_
        c2 (_type_): _description_
        frat (_type_): _description_
        fp2 (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Coefficients: ERBw(fr1) = alp1*fr1+alp0
    _, alp0 = utils.freq2erb(0)
    _, w1 = utils.freq2erb(1)
    alp1 = w1 - alp0

    # Coefficients: fr2=bet1*fr2+bet0
    bet1 = frat*(1+c1*b1*alp1/n)
    bet0 = frat*c1*b1*alp0/n

    # Coefficients: ERB(fr2)=zet1*Fr1+zet0
    zet1=alp1*bet1
    zet0=alp1*bet0+alp0

    # Coef1*Fr1**3 + Coef2*Fr1**2 + Coef3*Fr1 + Coef4 = 0
    coef1 = ((b2**2*zet1**2+bet1**2)*(c1*b1*alp1+n) + (c2*b2*zet1)*(b1**2*alp1**2+1))
    coef2 = ((b2**2*zet1**2+bet1**2)*(c1*b1*alp0-n*fp2) \
            + (2*b2**2*zet1*zet0-2*bet1*(fp2-bet0))*(c1*b1*alp1+n) \
            + (c2*b2*zet1)*(2*b1**2*alp1*alp0-2*fp2) + (c2*b2*zet0)*(b1**2*alp1**2+1))
    coef3 = ((2*b2**2*zet1*zet0-2*bet1*(fp2-bet0))*(c1*b1*alp0-n*fp2) \
            + (b2**2*zet0**2+(fp2-bet0)**2)*(c1*b1*alp1+n) \
            + (c2*b2*zet1)*(b1**2*alp0**2+fp2**2) \
            + (c2*b2*zet0)*(2*b1**2*alp1*alp0-2*fp2))
    coef4 = (b2**2*zet0**2+(fp2-bet0)**2)*(c1*b1*alp0-n*fp2) \
            + (c2*b2*zet0)*(b1**2*alp0**2+fp2**2)

    q = np.roots([coef1, coef2, coef3, coef4])
    fr1cand = q[np.imag(q)==0]

    if len(fr1cand) == 1:
        fr1 = fr1cand
        fp1 = fr2fpeak(n, b1, c1, fr1)
    else:
        fp1cand = fr2fpeak(n, b1, c1, fr1cand)
        ncl = np.argmin(np.abs(fp1cand - fp2)) 
        fp1 = fp1cand[ncl]
        fr1 = fr1cand[ncl]           

    return fr1, fp1
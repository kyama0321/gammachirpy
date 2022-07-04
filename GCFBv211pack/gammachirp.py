# -*- coding: utf-8 -*-
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

    fps = utils.fr2fpeak(order_g, coef_erbw, coef_c, frs) # Peak freq.
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
            fp, _ = utils.fr2fpeak(order_g[nch], coef_erbw[nch], coef_c[nch], frs[nch])
            npeak = np.argmin(np.abs(freq - fp))
            gc[nch, :] = gc[nch, :] / np.abs(frsp[npeak])

    return gc, len_gc, fps, inst_freq
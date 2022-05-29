# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import utils

def GammaChirp(Frs, SR, OrderG=4, CoefERBw=1.019, CoefC=0, Phase=0, SwCarr='cos', SwNorm='no'):
    """Gammachirp : Theoretical auditory filter
    gc(t) = t^(n-1) exp(-2 pi b ERB(Frs)) cos(2*pi*Frs*t + c ln t + phase)

    Args:
        Frs (array_like): Asymptotic Frequency
        SR (int): Sampling frequency
        OrderG (int, optional): Order of Gamma function t^(OrderG-1), n. Defaults to 4.
        CoefERBw (float, optional): Coeficient -> exp(-2*pi*CoefERBw*ERB(f)), b. Defaults to 1.019.
        CoefC (int, optional): Coeficient -> exp(j*2*pi*Frs + CoefC*ln(t)), c. Defaults to 0.
        Phase (int, optional): Start Phase(0 ~ 2*pi). Defaults to 0.
        SwCarr (str, optional): Carrier ('cos','sin','complex','envelope' with 3 letters). Defaults to 'cos'.
        SwNorm (str, optional): Normalization of peak spectrum level ('no', 'peak'). Defaults to 'no'.

    Returns:
        GC (array_like): GammaChirp
        LenGC (array_like): Length of GC for each channel
        Fps (array_like): Peak frequency
        InstFreq (array_like): Instanteneous frequency
    """    

    NumCh = len(Frs)
    OrderG = OrderG * np.ones((NumCh, 1))
    CoefERBw = CoefERBw * np.ones((NumCh, 1))
    CoefC = CoefC * np.ones((NumCh, 1))
    Phase = Phase * np.ones((NumCh, 1))

    ERBrate, ERBw = utils.Freq2ERB(Frs)
    LenGC1kHz = (40*max(OrderG)/max(CoefERBw) + 200) * SR/16000
    _, ERBw1kHz = utils.Freq2ERB(1000)

    if SwCarr == 'sin':
        Phase = Phase - np.pi/2*np.ones((1,NumCh))

    # Phase compensation
    Phase = Phase + CoefC * np.log(Frs/1000) # relative phase to 1kHz

    LenGC = np.fix(LenGC1kHz*ERBw1kHz/ERBw)

    """
    Production of GammaChirp
    """
    GC = np.zeros((NumCh, int(max(LenGC))))

    Fps = utils.Fr2Fpeak(OrderG, CoefERBw, CoefC, Frs) # Peak freq.
    InstFreq = np.zeros((NumCh, int(max(LenGC))))

    for nch in range(NumCh):
        t = np.arange(1, int(LenGC[nch]))/SR

        GammaEnv = t**(OrderG[nch]-1) * np.exp(-2*np.pi*CoefERBw[nch]*ERBw[nch]*t)
        GammaEnv = np.array([0] + list(GammaEnv/max(GammaEnv)))

        if SwCarr == 'env': # envelope
            Carrier = np.ones(np.shape(GammaEnv))
        elif SwCarr == 'com': # complex
            Carrier = np.array([0] + list(np.exp(1j*(2*np.pi*Frs[nch]*t + CoefC[nch]*np.log(t) + Phase[nch]))))
        else:
            Carrier = np.array([0] + list(np.cos(2*np.pi*Frs[nch]*t) + CoefC[nch]*np.log(t) + Phase[nch]))
        
        GC[nch, 0: int(LenGC[nch])] = GammaEnv * Carrier

        InstFreq[nch, 0: int(LenGC[nch])] = np.array([0] + list(Frs[nch] + list(CoefC[nch]/(2*np.pi*t))))

        if SwNorm == 'peak': # peak gain normalization
            freq, frsp = signal.freqz(GC[nch, 0: int(LenGC[nch])], 1, 2**utils.nextpow2(int(LenGC[nch])), fs=SR)
            fp, _ = utils.Fr2Fpeak(OrderG[nch], CoefERBw[nch], CoefC[nch], Frs[nch])
            npeak = np.argmin(np.abs(freq - fp))
            GC[nch, :] = GC[nch, :] / abs(frsp[npeak])

    return GC, LenGC, Fps, InstFreq
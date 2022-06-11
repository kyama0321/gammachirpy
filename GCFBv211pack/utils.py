# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import wave as wave
import time
from scipy.interpolate import UnivariateSpline
from scipy import signal
from functools import lru_cache


def audioread(filepath):
    """Reads a wavfile as a float 
        
        Parameters
        ----------
        filepath: string
            Filepath to the input wav file

        Returns
        ----------
        wavSnd: float
            Sound signal as a float and normalized scale (-1 ~ +1) 
        wavFs: 
            Sampling frequency
    """
    wav = wave.open(filepath)
    wavFs = wav.getframerate() # sampling frequency
    wavSnd = wav.readframes(wav.getnframes())
    wavSnd = np.frombuffer(wavSnd, dtype=np.int16) # int16 (-32768 ~ +32767)
    wav.close()
    wavSnd = wavSnd/abs(np.iinfo(np.int16).min) # float (-1 ~ +1)
    return wavSnd, wavFs


def rms(x):
    """
        Caliculates a root-mean-squared (RMS) value of input signal (1D)

        Parameters
        ----------
        x: float
            Input signal (1D)

        Returns
        ----------
        y: RMS value

    """
    y = np.sqrt(np.mean(x * x))
    return y

def nextpow2(n):
    """Find exponent of next higher power of 2

    Args:
        n (array-like): Input values

    Returns:
        p (array-like): Exponent of next higher power of 2
    """
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    p = int(np.log2(2**m_i))

    return p


def Eqlz2MeddisHCLevel(Snd, OutLeveldB, *args):
    """ Equalizing Signal RMS Level to the Level for MeddisHairCell

    Args:
        Snd (float): Input sound
        OutLeveldB (float): Output level (No default value,  RMS level)

    Returns:
        SndEqM (float): Equalized Sound (rms value of 1 is 30 dB SPL)
        AmpdB (array_like): 3 values in dB, [OutputLevel_dB, CompensationValue_dB, SourceLevel_dB]

    Matlab examples:
        rms(s(t)) == sqrt(mean(s.^2)) == 1   --> 30 dB SPL
        rms(s(t)) == sqrt(mean(s.^2)) == 10  --> 50 dB SPL
        rms(s(t)) == sqrt(mean(s.^2)) == 100 --> 70 dB SPL  

    Reference:
        Meddis (1986), JASA, 79(3),pp.702-711.
    """
    SourceLevel = np.sqrt(np.mean(Snd**2)) * 10**(30/20) # level in terms of Meddis Level

    Amp = (10**(OutLeveldB/20))/SourceLevel
    SndEqM = Amp * Snd

    AmpdB = [OutLeveldB, 20*np.log10(Amp), 20*np.log10(SourceLevel)]

    return SndEqM, AmpdB


def EqualFreqScale(NameScale, NumCh, RangeFreq):
    """Calculation of Equal Frequency scale on ERB/Mel/Log/Linear scale

    Args:
        NameScale (string): 'ERB', 'mel', 'log', 'linear'
        NumCh (int): Number of channels
        RangeFreq (array_like): Frequency Range

    Returns:
        Frs (array_like): Fr vector
        WFval (array_like): Wraped freq. value
    """
    eps = np.finfo(float).eps # epsilon

    if NameScale == 'linear':
        RangeWF = RangeFreq
        dWF = np.diff(RangeWF) / (NumCh-1)
        WFvals = np.linspace(RangeWF[0], RangeWF[1]+eps*1000, dWF)
        Frs = WFvals

    elif NameScale == 'mel':
        RangeWF = Freq2FMel(RangeFreq)
        dWF = np.diff(RangeWF) / (NumCh-1)
        WFvals = np.linspace(RangeWF[0], RangeWF[1]+eps*1000, dWF)
        Frs = Mel2Freq(WFvals)

    elif NameScale == 'ERB':
        RangeWF, _ = Freq2ERB(RangeFreq)
        dWF = np.diff(RangeWF) / (NumCh-1)
        WFvals = np.arange(RangeWF[0], RangeWF[1]+eps*1000, dWF)
        Frs, _ = ERB2Freq(WFvals)

    elif NameScale == 'log':
        if min(RangeFreq) < 50:
            print("min(RangeFreq) < 50. Rplaced by 50.")
            RangeFreq[0] = 50
        RangeWF = np.log10(RangeFreq)
        dWF = np.diff(RangeWF) / (NumCh-1)
        WFvals = np.linspace(RangeWF[0], RangeWF[1]+eps*1000, dWF)
        Frs = 10**(WFvals)
    else:
        help(EqualFreqScale)
        print("Specify NameScale correctly", file=sys.stderr)
        sys.exit(1)
    
    return Frs, WFvals


def Freq2FMel(freq):
    """Convert mel to linear frequency

    Args:
        freq (array_like): linaer-scale frequency [Hz] 

    Returns:
        mel (array_like): mel-scale frequency [mel]

    Note:
        The function was made by the GammachirPy project because there is not original code of "mel2freq" in GCFBv211pack 
    """
    mel = 2595 * np.log10(1+freq/700)
    return mel


def Mel2Freq(mel):
    """Convert mel to linear frequency

    Args:
        mel (array_like): mel-scale frequency [mel] 

    Returns:
        freq (array_like): linear-scale frequency [Hz]

    Note:
        The function was made by the GammachirPy project because there is not original code of "mel2freq" in GCFBv211pack 
    """
    freq = 700 * ((10**(mel/2595))-1)
    return freq


def Freq2ERB(cf=None, warning=0):
    """Convert linear frequency to ERB

    Args:
        cf (array_like): center frequency in linaer-scale [Hz]. Default is None.
        warning (int): check frequency range. Default is 0.

    Returns:
        ERBrate (array_like): ERB_N rate [ERB_N] or [cam] 
        ERBwidth (array_like): ERB_N Bandwidth [Hz]
    """

    if warning == 1:
        # Warnig for frequency range
        cfmin = 50
        cfmax = 12000

        if np.min(cf) < cfmin or np.max(cf) > cfmax:
            print("Warning : Min or max frequency exceeds the proper ERB range: "
                +"{} (Hz) <= Fc <= {} (Hz)".format(cfmin, cfmax), file=sys.stderr)
            sys.exit(1)

    ERBrate = 21.4 * np.log10(4.37*cf/1000+1)
    ERBwidth = 24.7 * (4.37*cf/1000+1)

    return ERBrate, ERBwidth


def ERB2Freq(ERBrate):
    """Convert ERBrate to linear frequency

    Args:
        ERBrate (array_like): ERB_N rate [ERB_N] or [cam] 
    
    Returns:
        cf (array_like): center frequency in linaer-scale [Hz] 
        ERBwidth (array_like): ERB_N Bandwidth [Hz]
    """
    cf = (10**(ERBrate/21.4)-1) / 4.37 * 1000
    ERBwidth = 24.7 * (4.37*cf/1000+1)

    return cf, ERBwidth


def Fr2Fpeak(n, b, c, fr):
    """Estimate fpeak from fr

    Args:
        n (float): a parameter of the gammachirp
        b (float): a parameter of the gammachirp
        c (float): a parameter of the gammachirp
        fr (float): fr

    Returns:
        fpeak (float): peak frequency
        ERBw (float): ERBwidth at fr
    """

    _, ERBw = Freq2ERB(fr)
    fpeak = fr + c*ERBw*b/n

    return fpeak, ERBw


@lru_cache(maxsize=None)
def OutMidCrctFilt(StrCrct, SR, SwPlot=0, SwFilter=0):
    """Outer/middle ear compensation filter

    Args:
        StrCrct (string): String for Correction ELC/MAF/MAP
        SR (int): Sampling rate
        SwPlot (int): Switch of plot (0:OFF/1:ON) (default:0)
        SwFilter (int): Switch of filter type
            0: FIR linear phase filter (default)
            1: FIR linear phase inverse filter filter
            2: FIR mimimum phase filter (length: half of linear phase filter)

    Returns:
        FIRCoef (array_like): FIR filter coefficients
        StrFilt (string): Filter infomation

    Notes:
        In the original Matlab code of OutMidCrctFilt.m, persistent variables 
        are called by "persistent" function. The GammachirPy uses the "lru_cache"  
        instead of the persistent variables to call outputs if args are same 
        to previous one. 

    Reference:
        https://docs.python.org/3/library/functools.html
    """

    if SR > 48000:
        print("OutMidCrctFilt : Sampling rate of {} (Hz) (> 48000 (Hz) is not recommended)".format(SR))
        print("<-- ELC etc. is only defined below 16000 (Hz)")

    if SwFilter == 0:
        StrFilt = "FIR linear phase filter"
    elif SwFilter == 1:
        StrFilt = "FIR linear phase inverse filter"
    elif SwFilter == 2:
        StrFilt = "FIR minimum phase filter"
    else:
        help(OutMidCrctFilt)
        print("Specify filter type", file=sys.stderr)
        sys.exit(1)        

    if not StrCrct in ['ELC', 'MAF', 'MAP', 'MidEar']:
        help(OutMidCrctFilt)
        print("Specifiy correction: ELC/MAF/MAP/MidEar", file=sys.stderr)
        sys.exit(1)

    """
    Generating filter at the first time
    """
    print("*** OutMidCrctFilt: Generating {} {} ***".format(StrCrct, StrFilt))
    Nint = 1024
    # Nint = 0 # No spline interpolation:  NG no convergence at remez

    crctPwr, freq, _ = OutMidCrct(StrCrct, Nint, SR, 0)
    crct = np.sqrt(crctPwr[:,0])
    freq = freq[:,0]

    LenCoef = 200 # ( -45 dB) <- 300 (-55 dB)
    NCoef = int(np.fix(LenCoef/16000*SR/2)*2) # even number only

    if SwFilter == 1:
        crct = 1 / np.max(np.sqrt(crctPwr), 0.1) # Giving up less tan -20 dB : f>15000 Hz
                                                 # if requered, the response becomes worse
    
    LenCoef = 200 # ( -45 dB) <- 300 (-55 dB)
    NCoef = int(np.fix(LenCoef/16000*SR/2)*2) # even number only
    
    """ 
    Calculate the minimax optimal filter with a frequency response
    instead of "FIRCoef = firpm(NCoef,freq/SR*2,crct)" in the original code OutMidCrctFilt.m
    """
    x1 = np.array(np.arange(len(freq))).T * 2
    x2 = np.array(np.arange(len(freq)*2)).T
    freq_interp = np.interp(x2, x1, freq)
    FIRCoef = signal.remez(NCoef+1, freq_interp, crct, fs=SR) # len(freq_interp) must be twice of len(crct)

    Win, _ = TaperWindow(len(FIRCoef),'HAN',LenCoef/10)
    FIRCoef = Win * FIRCoef

    """
    Minimum phase reconstruction
    """
    if SwFilter == 2: 
        _, x_mp = rceps(FIRCoef)
        FIRCoef = x_mp[0:int(np.fix(len(x_mp)/2))]

    """
    Plot
    """
    if SwPlot == 1:
        Nrsl = 1024
        freq2, frsp = signal.freqz(FIRCoef, 1, Nrsl, fs=SR)

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        plt.plot(FIRCoef)
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlim([0, 300])
        ax1.set_ylim([-0.3, 0.3])
        
        ax2 = fig.add_subplot(2, 1, 2)
        plt.plot(freq2, abs(frsp), freq, crct, '--')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude (linear term)')
        ax2.set_xlim([0, 25000])
        ax2.set_ylim([0, 1.8])
        
    return FIRCoef, StrFilt



def OutMidCrct(StrCrct, NfrqRsl=0, fs=32000, SwPlot=1):
    """Correction of ELC, MAF, MAP, MID. 
    It produces interpolated points for the ELC/MAF/MAP/MidEar correction.

    Args:
        StrCrct (string): Correction ELC/MAF/MAP/MidEar
        NfrqRsl (int): Number of data points, if zero, then direct out (default: 0)
        fs (int): Sampling frequency (default: 32000)
        SwPlot (int): Switch for plot (0/1, default:1)

    Returns:
        CrctLinPwr (array_like): Correction value in LINEAR POWER. 
            This is defined as:  CrctLiPwr =10^(-FreqChardB_toBeCmpnstd/10)
        freq (array_like): Corresponding Frequency at the data point
        FreqChardB_toBeCmpnstd (array_like): Frequency char of ELC/MAP dB 
            to be compensated for filterbank (defined By Glassberg and Moore.)

    Note: 
        "ER4B" option in StrCrct was omitted because the option uses a special measurement data. 
    """

    # ER4B: Omitted 
    """  
    if StrCrct == 'ER4B':
        CrctLinPwr, freq, FreqChardB_toBeCmpnstd = OutMidCrct_ER4B(NfrqRsl, fs, SwPlot)
        return CrctLinPwr, freq, FreqChardB_toBeCmpnstd
    """

    """
    Conventional ELC/MAF/MAP/MidEar
    """

    f1 = [	20,   25,  30,     35,  40,    45,  50,   55,   60,   70,  # 1-10
            80,   90,  100,   125,  150,   177, 200,  250,  300,  350,  # 11-20
            400,  450, 500,   550,  600,   700, 800,  900,  1000, 1500,  # 21-30
            2000, 2500, 2828, 3000, 3500, 4000, 4500, 5000, 5500, 6000,  # 31-40
            7000, 8000, 9000, 10000, 12748, 15000]   # 41-46

    ELC = [ 31.8, 26.0, 21.7, 18.8, 17.2, 15.4, 14.0, 12.6, 11.6, 10.6, 
            9.2, 8.2, 7.7, 6.7, 5.3, 4.6, 3.9, 2.9, 2.7, 2.3, 
            2.2, 2.3, 2.5, 2.7, 2.9, 3.4, 3.9, 3.9, 3.9, 2.7, 
            0.9, -1.3, -2.5, -3.2, -4.4, -4.1, -2.5, -0.5, 2.0, 5.0, 
            10.2, 15.0, 17.0, 15.5, 11.0, 22.0]

    MAF = [ 73.4, 65.2, 57.9, 52.7, 48.0, 45.0, 41.9, 39.3, 36.8, 33.0, 
            29.7, 27.1, 25.0, 22.0, 18.2, 16.0, 14.0, 11.4, 9.2, 8.0, 
            6.9,  6.2,  5.7,  5.1,  5.0,  5.0,  4.4,  4.3, 3.9, 2.7, 
            0.9, -1.3, -2.5, -3.2, -4.4, -4.1, -2.5, -0.5, 2.0, 5.0, 
            10.2, 15.0, 17.0, 15.5, 11.0, 22.0]

    f2  = [  125,  250,  500, 1000, 1500, 2000, 3000, 
            4000, 6000, 8000,10000,12000,14000,16000]

    MAP = [ 30.0, 19.0, 12.0,  9.0, 11.0, 16.0, 16.0, 
            14.0, 14.0,  9.9, 24.7, 32.7, 44.1, 63.7]

    # MidEar Correction (little modification at 17000:1000:20000)
    f3 =  [   1,  20,  25, 31.5,   40,   50,   63,   80,  100,  125, 
            160, 200, 250,  315,  400,  500,  630,  750,  800, 1000,
            1250, 1500, 1600,  2000,  2500,  3000,  3150,  4000,  5000,  6000, 
            6300, 8000, 9000, 10000, 11200, 12500, 14000, 15000, 16000,  20000]

    MID =  [  50,  39.15, 31.4, 25.4, 20.9,  18, 16.1, 14.2, 12.5, 11.13,
            9.71,   8.42,  7.2,  6.1,  4.7, 3.7,  3.0,  2.7,  2.6,   2.6,
             2.7,    3.7,  4.6,  8.5, 10.8, 7.3,  6.7,  5.7,  5.7,   7.6,
             8.4,   11.3, 10.6,  9.9, 11.9, 13.9, 16.0, 17.3, 17.8,  20.0] 


    frqTbl = []
    TblFreqChar = []
    if StrCrct == 'ELC':
        frqTbl = np.array([f1]).T
        TblFreqChar = np.array([ELC]).T
        ValHalfFs = 130
    elif StrCrct == 'MAF':
        frqTbl = np.array([f1]).T
        TblFreqChar = np.array([MAF]).T
        ValHalfFs = 130
    elif StrCrct == 'MAF':
        frqTbl = np.array([f2]).T
        TblFreqChar = np.array([MAP]).T
        ValHalfFs = 180
    elif StrCrct == 'MidEar':
        frqTbl = np.array([f3]).T
        TblFreqChar = np.array([MID]).T
        ValHalfFs = 23
    elif StrCrct == 'NO':
        pass
    else:
        print("Specifiy correction: ELC/MAF/MAP/MidEar or NO correction", file=sys.stderr)
        sys.exit(1)

    """
    Additional dummy data for high sampling frequency
    """
    if fs > 32000:
        frqTbl = np.vstack([frqTbl, fs/2])
        TblFreqChar = np.vstack([TblFreqChar, ValHalfFs])
        frqTbl, indx = np.unique(frqTbl, return_index=True)
        frqTbl = np.array([frqTbl]).T
        TblFreqChar = TblFreqChar[indx]

    str1 = ''
    if NfrqRsl <= 0:
        str1 = 'No interpolation. Output: values in original table.'
        freq = frqTbl
        FreqChardB_toBeCmpnstd = TblFreqChar
    else:
        freq = np.array([np.arange(NfrqRsl)/NfrqRsl * fs/2]).T
        if StrCrct == 'NO':
            FreqChardB_toBeCmpnstd = np.zeros(freq.shape)
        else:
            str1 = 'Spline interpolated value in equal frequency spacing.'
            freq_1d = freq.T[0,:]
            frqTbl_1d = frqTbl.T[0,:]
            TblFreqChar_1d = TblFreqChar.T[0,:]
            spl = UnivariateSpline(frqTbl_1d, TblFreqChar_1d, s=0)
            FreqChardB_toBeCmpnstd = spl(freq_1d)
            FreqChardB_toBeCmpnstd = np.array([FreqChardB_toBeCmpnstd]).T
    
    if SwPlot == 1:
        str = "*** Frequency Characteristics (" + StrCrct + "): Its inverse will be corrected. ***"
        print(str) 
        print("{}".format(str1))
        fig, ax = plt.subplots()
        plt.plot(frqTbl, TblFreqChar, 'b-',freq, FreqChardB_toBeCmpnstd, 'r--')
        plt.xlim(0, 25000)
        plt.ylim(-20,140)
        ax.set_title(str)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Level (dB)')
        plt.show()

    CrctLinPwr = 10**(-FreqChardB_toBeCmpnstd/10) # in Linear Power. Checked 19 Apr 2016

    return CrctLinPwr, freq, FreqChardB_toBeCmpnstd


def TaperWindow(LenWin, TypeTaper, LenTaper=None, RangeSigma=3, SwPlot=0):
    """Taper Window Generator for signal onset/offset

    Args:
        LenWin (int): Length of window (number of points)
        TypeTaper (string): Type of Taper (KeyWords of 3 letters)
            - HAM: Hamming
            - HAN/COS: Hanning/Cosine
            - BLA: Blackman
            - GAU: Gaussian
            - (other): Linear
        LenTaper (int, optional): Length of taper. Defaults to None.
        RangeSigma (int, optional): Range in sigma. Defaults to 3.
        SwPlot (int, optional): OFF/ON. Defaults to 0.

    Returns:
        TaperWin (array_like): Taper window points (max: 1)
        TypeTaper (string): Type of taper (full name)
    """

    if LenTaper == None:
        LenTaper = int(np.fix(LenWin/2))
    
    elif LenTaper*2+1 >= LenWin:
        print("Caution (TaperWindow) : No flat part. ")
        
        if not LenTaper == np.fix(LenWin/2):
            print("Caution (TaperWindow) : LenTaper <-- fix(LenWin/2)")
            
        LenTaper = int(np.fix(LenWin/2))

    LenTaper= int(LenTaper)

    if TypeTaper == 'HAM':
        Taper = np.hamming(LenTaper*2+1)
        TypeTaper = 'Hamming'

    elif TypeTaper == 'HAN' or TypeTaper == 'COS':
        Taper = np.hamming(LenTaper*2+1)
        TypeTaper = 'Hanning/Cosine'

    elif TypeTaper == 'BLA':
        Taper = np.blackman(LenTaper*2+1)
        TypeTaper = 'Blackman'

    elif TypeTaper == 'GAU':
        if len(RangeSigma) == 0:
            RangeSigma = 3
        nn = np.arange(-LenTaper, LenTaper, 1)
        Taper = np.exp(-(RangeSigma/LenTaper)**2 / 2)
        TypeTaper == 'Gauss'

    else:
        Taper = np.array(list(np.arange(1,LenTaper+1,1)) + list([LenTaper+1]) + list(np.arange(LenTaper,1-1,-1))) / (LenTaper+1)
        TypeTaper = 'Line'

    LenTaper = int(np.fix(LenTaper))
    TaperWin = list(Taper[0:LenTaper]) + list(np.ones(LenWin-LenTaper*2)) + list(Taper[(LenTaper+1):(LenTaper*2+1)])

    if SwPlot == 1:
        fig, ax = plt.subplots()
        plt.plot(TaperWin)
        ax.set_xlabel('Points')
        ax.set_ylabel('Amplitude')
        plt.title('TypeTaper: {}'.format(TypeTaper))

    return TaperWin, TypeTaper


def rceps(x):
    """returns the real cepstrum of the real sequence X

    Args:
        x (array_like): input signal

    Returns:
        xhat: real cepstrum
        yhat: a unique minimum-phase sequence that has the reame real cepstrum as x

    Note:
        This code is based on "rceps.m" in MATLAB and is under-construction. 

    Examples:
        x = [4 1 5]; % Non-minimum phase sequence
        xhat = array([1.62251148, 0.3400368 , 0.3400368 ])
        yhat = array([5.33205452, 3.49033278, 1.1776127 ])

    References:
    - A.V. Oppenheim and R.W. Schafer, Digital Signal Processing, Prentice-Hall, 1975.
    - Programs for Digital Signal Processing, IEEE Press, John Wiley & Sons, 1979, algorithm 7.2.
    - https://mathworks.com/help/signal/ref/rceps.html
    """

    if isrow(x):
        xT = np.array([x]).T
    else:
        xT = x

    fftxabs = np.abs(np.fft.fft(xT, n=None, axis=0))

    xhatT = np.real(np.fft.ifft(np.log(fftxabs), n=None, axis=0))

    # xhat
    if isrow(x):
        # transform the result to a row vector
        xhat = xhatT[:,0]
    else:
        xhat = xhatT

    # yhat
    nRows = xhatT.shape[0]
    nCols = xhatT.shape[1]
    odd = nRows % 2
    a1 = np.array([1])
    a2 = 2*np.ones((int((nRows+odd)/2)-1, 1))
    a3 = np.zeros((int((nRows+odd)/2)-1,1))
    wn = np.kron(np.ones((1, nCols)), np.vstack((a1,a2,a3)))
    """
    Matlab can use zero and negative numbers for args of ones function, but the np.ones cannot. So, an internal array is removed. The original code is: 
    wn = np.kron(np.ones((1, nCols)), np.array([[1], 2*np.ones((int((nRows+odd)/2)-1, 1)), np.ones(1-odd, 1), np.zeros((int((nRows+odd)/2)-1,1))]))
    """
    yhatT = np.real(np.fft.ifft(np.exp(np.fft.fft((wn*xhatT),n=None, axis=0)), n=None, axis=0))
    if isrow(x):
        # transform the result to a row vector
        yhat = yhatT[:,0]
    else:
        yhat = yhatT

    return xhat, yhat


def isrow(x):
    """returns True if x is a row vector, False otherwise.

    Args:
        x (array_like): verctors

    Returns:
        logical (bool): True/False
    """

    if np.size(np.shape(x)) == 1:
        logical = True
    else:
        logical = False

    return logical
    

def MakeAsymCmpFiltersV2(fs,Frs,b,c):
    """Computes the coefficients for a bank of Asymmetric Compensation Filters
    This is a modified version to fix the round off problem at low freqs
    Use this with ACFilterBank.m
    See also AsymCmpFrspV2 for frequency response

    Args:
        fs (int): Sampling frequency
        Frs (array_like): array of the center frequencies, Frs
        b (array_like): array or scalar of a bandwidth coefficient, b
        c (float): array or scalar of asymmetric parameters, c

    Returns:
        ACFcoef: 
        - fs (int): Sampling frequency
        - bz (array_like): MA coefficients  (NumCh*3*NumFilt)
        - ap (array_like): AR coefficients  (NumCh*3*NumFilt)

    Notes:
        [1] Ref for p1-p4: Unoki,M , Irino,T. , and Patterson, R.D. , "Improvement of an IIR asymmetric compensation gammachirp filter," Acost. Sci. & Tech. (ed. by the Acoustical Society of Japan ), 22 (6), pp. 426-430, Nov. 2001.
        [2] Conventional setting was removed.
            fn = Frs + Nfilt* p3 .*c .*b .*ERBw/n;
            This frequency fn is for normalizing GC(=GT*Hacf) filter to be unity at the peak, frequnecy. But now we use Hacf as a highpass filter as well. cGC = pGC *Hacf. In this case, this normalization is useless. 
            So, it was set as the gain at Frs is unity.  (4. Jun 2004 )
        [3] Removed
            ACFcoef.fn(:,nff) = fn;
            n : scalar of order t^(n-1) % used only in normalization 
    """

    class ACFcoef:
        fs = []
        ap = np.array([])
        bz = np.array([])


    NumCh, LenFrs = np.shape(Frs)
    if LenFrs > 1:
        print("Frs should be a column vector Frs.", file=sys.stderr)
        sys.exit(1)
    
    _, ERBw = Freq2ERB(Frs)
    ACFcoef.fs = fs

    # New coefficients. See [1]
    NumFilt = 4
    p0 = 2
    p1 = 1.7818 * (1-0.0791*b) * (1-0.1655*np.abs(c))
    p2 = 0.5689 * (1-0.1620*b) * (1-0.0857*np.abs(c))
    p3 = 0.2523 * (1-0.0244*b) * (1+0.0574*np.abs(c))
    p4 = 1.0724

    if NumFilt > 4:
        print("NumFilt > 4", file=sys.stderr)
        sys.exit(1) 

    ACFcoef.ap = np.zeros((NumCh, 3, NumFilt))
    ACFcoef.bz = np.zeros((NumCh, 3, NumFilt))

    for Nfilt in range(NumFilt):
        r  = np.exp(-p1*(p0/p4)**(Nfilt) * 2*np.pi*b*ERBw / fs)
        delFrs = (p0*p4)**(Nfilt)*p2*c*b*ERBw;  
        phi = 2*np.pi*(Frs+delFrs).clip(0)/fs
        psi = 2*np.pi*(Frs-delFrs).clip(0)/fs
        fn = Frs # see [2]

        # second order filter
        ap = np.concatenate([np.ones(np.shape(r)), -2*r*np.cos(phi), r**2], axis=1)
        bz = np.concatenate([np.ones(np.shape(r)), -2*r*np.cos(psi), r**2], axis=1)

        vwr = np.exp(1j*2*np.pi*fn/fs)
        vwrs = np.concatenate([np.ones(np.shape(vwr)), vwr, vwr**2], axis=1)
        nrm = np.array([np.abs(np.sum(vwrs*ap, axis=1) / np.sum(vwrs*bz, axis=1))]).T
        bz = bz * (nrm*np.ones((1, 3)))

        ACFcoef.ap[:,:,Nfilt] = ap
        ACFcoef.bz[:,:,Nfilt] = bz

    return ACFcoef



def Fr1toFp2(n, b1, c1, b2, c2, frat, Fr1, SR=24000, Nfft=2048, SwPlot=0):
    """Convert Fr1 (for passive GC; pGC) to Fp2 (for compressive GC; cGC)

    Args:
        n (int): Parameter defining the envelope of the gamma distribution (for pGC)
        b1 (float): Parameter defining the envelope of the gamma distribution (for pGC)
        c1 (float): Chirp factor (for pGC)
        b2 (float): Parameter defining the envelope of the gamma distribution (for cGC)
        c2 (float): Chirp factor  (for cGC)
        frat (float): Frequency ratio, the main level-dependent variable
        Fr1 (float): Center frequency (for pGC)
        SR (int, optional): Sampling rate. Defaults to 24000.
        Nfft (int, optional): Size of FFT. Defaults to 2048.
        SwPlot (int, optional): Show plot of cGCFrsp and pGCFrsp. Defaults to 0.

    Returns:
        Fp2 (float): Peak frequency (for compressive GC)
        Fr2 (float): Center Frequency (for compressive GC)
    """

    _, ERBw1 = Freq2ERB(Fr1)
    Fp1, _ = Fr2Fpeak(n, b1, c1, Fr1)
    Fr2 = frat * Fp1
    _, ERBw2 = Freq2ERB(Fr2)

    Bw1 = b1 * ERBw1
    Bw2 = b2 * ERBw2

    # Coef1*Fp2^3 + Coef2*Fp2^2 + Coef3*Fp2 + Coef4 = 0 
    Coef1 = -n
    Coef2 = c1*Bw1 + c2*Bw2 + n*Fr1 + 2*n*Fr2
    Coef3 = -2*Fr2*(c1*Bw1+n*Fr1) - n*((Bw2)**2+Fr2**2) - 2*c2*Bw2*Fr1
    Coef4 =  c2*Bw2*((Bw1)**2+Fr1**2) + (c1*Bw1+n*Fr1)*(Bw2**2+Fr2**2)
    Coefs = [Coef1, Coef2, Coef3, Coef4]

    p = np.roots(Coefs)
    Fp2cand = p[np.imag(p)==0]
    if len(Fp2cand) == 1:
        Fp2 = Fp2cand
    else:
        val, ncl = np.min(np.abs(Fp2cand - Fp1))
        Fp2 = Fp2cand(ncl) # in usual cGC range, Fp2 is close to Fp1

    # SwPlot = 1
    if SwPlot == 1: # Check
        fs = 48000
        NfrqRsl = 2048
        cGCrsp = CmprsGCFrsp(Fr1, fs, n, b1, c1, frat, b2, c2, NfrqRsl)

        nFr2 = np.zeros((len(Fp2cand), 1))
        for nn in range(len(Fp2cand)):
            nFr2[nn] = np.argmin(abs(cGCrsp.freq - Fp2cand[nn]))
        
        fig, ax = plt.subplots()
        plt_freq = np.array(cGCrsp.freq).T
        plt_cGCFrsp = np.array(cGCrsp.cGCFrsp/np.max(cGCrsp.cGCFrsp)).T
        plt_pGCFrsp = np.array(cGCrsp.pGCFrsp).T

        ax.plot(plt_freq, plt_cGCFrsp, label="cGCFrsp") # compressive GC
        ax.plot(plt_freq, plt_pGCFrsp, label="pGCFrsp") # passive GC
        ax.set_xlim([0, np.max(Fp2cand)*2])
        ax.set_ylim([0, 1])
        ax.legend()
        plt.show()

    return Fp2, Fr2


def CmprsGCFrsp(Fr1, fs=48000, n=4, b1=1.81, c1=-2.96, frat=1, b2=2.01, c2=2.20, NfrqRsl=1024):
    """Frequency Response of Compressive GammaChirp

    Args:
        Fr1 (array-like): Resonance Freqs.
        fs (int, optional): Sampling Freq. Defaults to 48000.
        n (int, optional): Order of Gamma function, t**(n-1). Defaults to 4.
        b1 (float, optional): b1 for exp(-2*pi*b1*ERB(f)). Defaults to 1.81.
        c1 (float, optional): c1 for exp(j*2*pi*Fr + c1*ln(t)). Defaults to -2.96.
        frat (int, optional): Frequency ratio. Fr2 = frat*Fp1. Defaults to 1.
        b2 (float, optional): _description_. Defaults to 2.01.
        c2 (float, optional): _description_. Defaults to 2.20.
        NfrqRsl (int, optional): _description_. Defaults to 1024.

    Returns:
        cGCresp: Struct of cGC response
            pGCFrsp (array-like): Passive GC freq. resp. (NumCh*NfrqRsl matrix)
            cGCFrsp (array-like): Comressive GC freq. resp. (NumCh*NfrqRsl matrix)
            cGCNrmFrsp (array-like): Normalized cGCFrsp (NumCh*NfrqRsl matrix)
            ACFrsp: Asym (array-like). Compensation Filter freq. resp.
            AsymFunc (array-like): Asym Func
            freq (array-like): Frequency (1*NfrqRsl)
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

    if isrow(Fr1):
        Fr1 = np.array([Fr1]).T

    NumCh = len(Fr1)

    if isinstance(n, (int, float)):
        n = n * np.ones((NumCh, 1))
    if isinstance(b1, (int, float)):
        b1 = b1 * np.ones((NumCh, 1))
    if isinstance(c1, (int, float)):
        c1 = c1 * np.ones((NumCh, 1))
    if isinstance(frat, (int, float)):
        frat = frat * np.ones((NumCh, 1))
    if isinstance(b2, (int, float)):
        b2 = b2 * np.ones((NumCh, 1))
    if isinstance(c2, (int, float)):
        c2 = c2 * np.ones((NumCh, 1))

    pGCFrsp, freq, _, _, _ = GammaChirpFrsp(Fr1, fs, n, b1, c1, 0.0, NfrqRsl)
    Fp1, _ = Fr2Fpeak(n, b1, c1, Fr1)
    Fr2 = frat * Fp1
    ACFFrsp, freq, AsymFunc = AsymCmpFrspV2(Fr2, fs, b2, c2, NfrqRsl)
    cGCFrsp = pGCFrsp * AsymFunc # cGCFrsp = pGCFrsp * ACFFrsp
    
    ValFp2 = np.max(cGCFrsp, axis=1)
    nchFp2 = np.argmax(cGCFrsp, axis=1)
    if isrow(ValFp2):
        ValFp2 = np.array([ValFp2]).T
    
    NormFactFp2 = 1/ValFp2

    # function cGCresp = CmprsGCFrsp(Fr1,fs,n,b1,c1,frat,b2,c2,NfrqRsl)
    cGCresp.Fr1 = Fr1
    cGCresp.n = n
    cGCresp.b1 = b1
    cGCresp.c1 = c1
    cGCresp.frat = frat
    cGCresp.b2 = b2
    cGCresp.c2 = c2
    cGCresp.NfrqRsl = NfrqRsl
    cGCresp.pGCFrsp = pGCFrsp
    cGCresp.cGCFrsp = cGCFrsp
    cGCresp.cGCNrmFrsp = cGCFrsp * (NormFactFp2 * np.ones((1,NfrqRsl)))
    cGCresp.ACFFrsp = ACFFrsp
    cGCresp.AsymFunc   = AsymFunc
    cGCresp.Fp1        = Fp1
    cGCresp.Fr2        = Fr2
    cGCresp.Fp2        = freq[nchFp2]
    cGCresp.ValFp2     = ValFp2
    cGCresp.NormFctFp2 = NormFactFp2
    cGCresp.freq       = [freq]

    return cGCresp


def GammaChirpFrsp(Frs, SR=48000, OrderG=4, CoefERBw=1.019, CoefC=0.0, Phase=0.0, NfrqRsl=1024):
    """Frequency Response of GammaChirp

    Args:
        Frs (array_like, optional): Resonance freq. Defaults to None.
        SR (int, optional): Sampling freq. Defaults to 48000.
        OrderG (int, optional): Order of Gamma function t**(OrderG-1). Defaults to 4.
        CoefERBw (float, optional): Coeficient -> exp(-2*pi*CoefERBw*ERB(f)). Defaults to 1.019.
        CoefC (int, optional): Coeficient -> exp(j*2*pi*Fr + CoefC*ln(t)). Defaults to 0.0.
        Phase (int, optional): Coeficient -> exp(j*2*pi*Fr + CoefC*ln(t)). Defaults to 0.9.
        NfrqRsl (int, optional): Freq. resolution. Defaults to 1024.

    Returns:
        AmpFrsp (array_like): Absolute of freq. resp. (NumCh*NfrqRsl matrix)
        freq (array_like): Frequency (1*NfrqRsl)
        Fpeak (array_like): Peak frequency (NumCh * 1)
        GrpDlay (array_like): Group delay (NumCh*NfrqRsl matrix)
        PhsFrsp (array_like): Angle of freq. resp. (NumCh*NfrqRsl matrix)
    """

    if isrow(Frs):
        Frs = np.array([Frs]).T

    NumCh = len(Frs)

    if isinstance(OrderG, (int, float)) or len(OrderG) == 1:
        OrderG = OrderG * np.ones((NumCh, 1))
    if isinstance(CoefERBw, (int, float)) or len(CoefERBw) == 1:
        CoefERBw = CoefERBw * np.ones((NumCh, 1))
    if isinstance(CoefC, (int, float)) or len(CoefC) == 1:
        CoefC = CoefC * np.ones((NumCh, 1))
    if isinstance(Phase, (int, float)) or len(Phase) == 1:
        Phase = Phase * np.ones((NumCh, 1))

    if NfrqRsl < 256:
        print("NfrqRsl < 256", file=sys.stderr)
        sys.exit(1)

    ERBrate, ERBw = Freq2ERB(Frs)
    freq = np.arange(NfrqRsl) / NfrqRsl * SR / 2
    freq = np.array([freq]).T

    one1 = np.ones((1, NfrqRsl))
    bh = (CoefERBw * ERBw) * one1
    fd = (np.ones((NumCh, 1)) * freq[:,0]) - Frs * one1
    cn = (CoefC / OrderG) * one1
    n = OrderG * one1
    c = CoefC * one1
    Phase = Phase * one1

    # Analytic form (normalized at Fpeak)
    AmpFrsp = ((1+cn**2) / (1+(fd/bh)**2))**(n/2) \
                * np.exp(c * (np.arctan(fd/bh)-np.arctan(cn)))
    
    Fpeak = Frs + CoefERBw * ERBw * CoefC / OrderG
    GrpDly = 1/(2*np.pi) * (n*bh + c*fd) / (bh**2 + fd**2)
    PhsFrsp = -n * np.arctan(fd/bh) - c / 2*np.log((2*np.pi*bh)**2 + (2*np.pi*fd)**2) + Phase

    return AmpFrsp, freq, Fpeak, GrpDly, PhsFrsp
    

def AsymCmpFrspV2(Frs, fs=48000, b=None, c=None, NfrqRsl=1024, NumFilt=4):
    """Amplitude spectrum of Asymmetric compensation IIR filter (ACF) for the gammachirp 
    corresponding to MakeAsymCmpFiltersV2

    Args:
        Frs (array_like, optional): Center freqs. Defaults to None.
        fs (int, optional): Sampling freq. Defaults to 48000.
        b (array_like, optional): Bandwidth coefficient. Defaults to None.
        c (array_like, optional): Asymmetric paramters. Defaults to None.
        NfrqRsl (int, optional): Freq. resolution for linear freq. scale for specify renponse at Frs
                                (NfrqRsl>64). Defaults to 1024.
        NumFilt (int, optional): Number of 2nd-order filters. Defaults to 4.

    Returns:
        ACFFrsp: Absolute values of frequency response of ACF (NumCh * NfrqRsl)
        freq: freq. (1 * NfrqRsl)
        AsymFunc: Original asymmetric function (NumCh * NfrqRsl)
    """

    if isrow(Frs):
        Frs = np.array([Frs]).T
    if isrow(b):
        b = np.array([b]).T
    if isrow(c):
        c = np.array([c]).T
    NumCh = len(Frs)

    if NfrqRsl >= 64:
        freq = np.arange(NfrqRsl) / NfrqRsl * fs/2
    elif NfrqRsl == 0:
        freq = Frs
        NfrqRsl = len(freq)
    else:
        help(AsymCmpFrspV2)
        print("Specify NfrqRsl 0) for Frs or N>=64 for linear-freq scale", file=sys.stderr)
        sys.exit(1)

    # coef.
    SwCoef = 0 # self consistency
    # SwCoef = 1 # reference to MakeAsymCmpFiltersV2

    if SwCoef == 0:
        # New Coefficients. NumFilter = 4; See [1]
        p0 = 2
        p1 = 1.7818 * (1 - 0.0791*b) * (1 - 0.1655*np.abs(c))
        p2 = 0.5689 * (1 - 0.1620*b) * (1 - 0.0857*np.abs(c))
        p3 = 0.2523 * (1 - 0.0244*b) * (1 + 0.0574*np.abs(c))
        p4 = 1.0724
    else:
        ACFcoef = MakeAsymCmpFiltersV2(fs, Frs, b, c)

    # filter coef.
    _, ERBw = Freq2ERB(Frs)
    ACFFrsp = np.ones((NumCh, NfrqRsl))
    freq2 = np.concatenate([np.ones((NumCh,1))*freq, Frs], axis=1)

    for Nfilt in range(NumFilt):

        if SwCoef == 0:
            r = np.exp(-p1 * (p0/p4)**Nfilt * 2 * np.pi * b * ERBw / fs)
            delfr = (p0*p4)**Nfilt * p2 * c * b * ERBw
            phi = 2*np.pi*np.maximum(Frs + delfr, 0)/fs
            psi = 2*np.pi*np.maximum(Frs - delfr, 0)/fs
            fn = Frs
            ap = np.concatenate([np.ones((NumCh, 1)), -2*r*np.cos(phi), r**2], axis=1)
            bz = np.concatenate([np.ones((NumCh, 1)), -2*r*np.cos(psi), r**2], axis=1)
        else:
            ap = ACFcoef.ap[:, :, Nfilt]
            bz = ACFcoef.bz[:, :, Nfilt]

        cs1 = np.cos(2*np.pi*freq2/fs)
        cs2 = np.cos(4*np.pi*freq2/fs)
        bzz0 = np.array([bz[:, 0]**2 + bz[:, 1]**2 + bz[:, 2]**2]).T * np.ones((1, NfrqRsl+1))
        bzz1 = np.array([2 * bz[:, 1] * (bz[:, 0] + bz[:, 2])]).T * np.ones((1, NfrqRsl+1))
        bzz2 = np.array([2 * bz[:, 0] * bz[:, 2]]).T * np.ones((1, NfrqRsl+1))
        hb = bzz0 + bzz1*cs1 + bzz2*cs2

        app0 = np.array([ap[:, 0]**2 + ap[:, 1]**2 + ap[:, 2]**2]).T * np.ones((1, NfrqRsl+1))
        app1 = np.array([2 * ap[:, 1] * (ap[:, 0] + ap[:, 2])]).T * np.ones((1, NfrqRsl+1))
        app2 = np.array([2 * ap[:, 0] * ap[:, 2]]).T * np.ones((1, NfrqRsl+1))
        ha = app0 + app1*cs1 + app2*cs2

        H = np.sqrt(hb/ha)
        Hnorm = np.array([H[:, NfrqRsl]]).T * np.ones((1, NfrqRsl)) # Normalizatoin by fn value

        ACFFrsp = ACFFrsp * H[:,0:NfrqRsl] / Hnorm

    # original Asymmetric Function without shift centering
    fd = np.ones((NumCh, 1))*freq - Frs*np.ones((1,NfrqRsl))
    be = (b * ERBw) * np.ones((1, NfrqRsl))
    cc = (c * np.ones((NumCh, 1)) * np.ones((1, NfrqRsl))) # in case when c is scalar
    AsymFunc = np.exp(cc * np.arctan2(fd, be))

    return ACFFrsp, freq, AsymFunc

class classACFstatus:
        NumCh = []
        NumFilt = []
        Lbz = []
        Lap = []
        SigInPrev = []
        SigOutPrev = []
        Count = []


def ACFilterBank(ACFcoef, ACFstatus, SigIn=[], SwOrdr=0):
    """IIR ACF time-slice filtering for time-varing filter

    Args:
        ACFcoef (structure): ACFcoef: coef from MakeAsymCmpFiltersV2
            bz: MA coefficents (==b ~= zero) NumCh*Lbz*NumFilt
            ap: AR coefficents (==a ~= pole) NumCh*Lap*NumFilt
            fs : sampling rate  (also switch for verbose)
                (The variables named 'a' and 'b' are not used to avoid the
                confusion to the gammachirp parameters.)
            verbose : Not specified) quiet   1) verbose
        ACFstatus (structure):
            NumCh: Number of channels (Set by initialization
            Lbz: size of MA
            Lap: size of AR
            NumFilt: Length of filters
            SigInPrev: Previous status of SigIn
            SigOutPrev: Previous status of SigOut
        SigIn (array_like, optional): Input signal. Defaults to [].
        SwOrdr (int, optional): Switch filtering order. Defaults to 0.

    Returns:
        SigOut (array_like): Filtered signal (NumCh * 1)
        ACFstatus: Current status
    """    

    if len(SigIn) == 0 and len(ACFstatus) != 0:
        help(ACFilterBank)
        sys.exit()

    if not hasattr(ACFstatus, 'NumCh'):
        ACFstatus = classACFstatus()

        NumCh, Lbz, NumFilt = np.shape(ACFcoef.bz)
        NumCh, Lap, NumFIlt = np.shape(ACFcoef.ap)

        if Lbz != 3 or Lap !=3:
            print("No gaurantee for usual IIR filters except for AsymCmpFilter.\n"\
                + "Please check MakeAsymCmpFiltersV2.")
    
        ACFstatus.NumCh = NumCh
        ACFstatus.NumFilt = NumFilt
        ACFstatus.Lbz = Lbz # size of MA
        ACFstatus.Lap = Lap # size of AR
        ACFstatus.SigInPrev = np.zeros((NumCh, Lbz))
        ACFstatus.SigOutPrev = np.zeros((NumCh, Lap, NumFilt))
        ACFstatus.Count = 0
        print("ACFilterBank: Initialization of ACFstatus")
        SigOut = []

        return SigOut, ACFstatus

    
    if isrow(SigIn):
        SigIn = np.array([SigIn]).T
    
    NumChSig, LenSig = np.shape(SigIn)
    if LenSig != 1:
        print("Input signal sould be NumCh*1 vector (1 sample time-slice)", file=sys.stderr)
        sys.exit(1)
    if NumChSig != ACFstatus.NumCh:
        print("NumChSig ({}) != ACFstatus.NumCh ({})".format(NumChSig, ACFstatus.NumCh))

    # time stamp
    if hasattr(ACFcoef, 'verbose'):
        if ACFcoef.verbose == 1: # verbose when ACFcoef.verbose is specified to 1
            Tdisp = 50 # ms
            Tcnt = ACFstatus.Count/(np.fix(ACFcoef.fs/1000)) # ms

            if ACFstatus.Count == 0:
                print("ACFilterBank: Start processing")
                Tic = time.time()

            elif np.mod(Tcnt, Tdisp) == 0:
                Toc = time.time()
                print("ACFilterBank: Processed {} (ms). elapsed Time = {} (sec)"\
                    .format(Tcnt, np.round(Tic-Toc, 1)))
    
    ACFstatus.Count = ACFstatus.Count+1
    
    """
    Processing
    """
    ACFstatus.SigInPrev = np.concatenate([ACFstatus.SigInPrev[:, 1:ACFstatus.Lbz], SigIn], axis=1)

    x = ACFstatus.SigInPrev.copy()
    NfiltList = np.arange(ACFstatus.NumFilt)

    if SwOrdr == 1:
        NfiltList = np.flip(NfiltList)

    for Nfilt in NfiltList:

        forward = ACFcoef.bz[:, ACFstatus.Lbz::-1, Nfilt] * x
        feedback = ACFcoef.ap[:, ACFstatus.Lap:0:-1, Nfilt] * \
            ACFstatus.SigOutPrev[:, 1:ACFstatus.Lap, Nfilt]

        fwdSum = np.sum(forward, axis=1)
        fbkSum = np.sum(feedback, axis=1)

        y = np.array([(fwdSum - fbkSum) / ACFcoef.ap[:, 0, Nfilt]]).T
        ACFstatus.SigOutPrev[:, :, Nfilt] = \
            np.concatenate([ACFstatus.SigOutPrev[:, 1:ACFstatus.Lap, Nfilt], y], axis=1)
        x = ACFstatus.SigOutPrev[:, :, Nfilt].copy()

    SigOut = y

    return SigOut, ACFstatus



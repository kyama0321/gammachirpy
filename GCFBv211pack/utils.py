# -*- coding: utf-8 -*-
import sys
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
import wave as wave


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


def Eqlz2MeddisHCLevel(Snd, OutLeveldB, *args):
    """ Equalizing Signal RMS Level to the Level for MeddisHairCell

    Args:
        Snd (float): Input sound
        OutLeveldB (float): Output level (No default value,  RMS level)

    Returns:
        SndEqM (float): Equalized Sound (rms value of 1 is 30 dB SPL)
        AmpdB (array): 3 values in dB, [OutputLevel_dB, CompensationValue_dB, SourceLevel_dB]

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
        RangeFreq (array): Frequency Range

    Returns:
        Frs (array): Fr vector
        WFval (array): Wraped freq. value
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
        freq (array): linaer-scale frequency [Hz] 

    Returns:
        mel (array): mel-scale frequency [mel]

    Note:
        The function was made by the GammachirPy project because there is not original code of "mel2freq" in GCFBv211pack 
    """
    mel = 2595 * np.log10(1+freq/700)
    return mel


def Mel2Freq(mel):
    """Convert mel to linear frequency

    Args:
        mel (array): mel-scale frequency [mel] 

    Returns:
        freq (array): linear-scale frequency [Hz]

    Note:
        The function was made by the GammachirPy project because there is not original code of "mel2freq" in GCFBv211pack 
    """
    freq = 700 * ((10**(mel/2595))-1)
    return freq


def Freq2ERB(cf):
    """Convert linear frequency to ERB

    Args:
        cf (array): center frequency in linaer-scale [Hz] 

    Returns:
        ERBrate (array): ERB_N rate [ERB_N] or [cam] 
        ERBwidth (array): ERB_N Bandwidth [Hz]
    """
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
        ERBrate (array): ERB_N rate [ERB_N] or [cam] 
    
    Returns:
        cf (array): center frequency in linaer-scale [Hz] 
        ERBwidth (array): ERB_N Bandwidth [Hz]
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
    #n = np.array([n]).T
    #b = np.array(b).T
    #c = np.array(c).T
    #fr = np.array(fr).T

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
        FIRCoef (array): FIR filter coefficients
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

    crctPwr, freq, _ = OutMidCrct(StrCrct, Nint, SR, 1)

    FIRCoef = [1,3,5] # dummy for test

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
        CrctLinPwr (array): Correction value in LINEAR POWER. 
            This is defined as:  CrctLiPwr =10^(-FreqChardB_toBeCmpnstd/10)
        freq (array): Corresponding Frequency at the data point
        FreqChardB_toBeCmpnstd (array): Frequency char of ELC/MAP dB 
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
            #FreqChardB_toBeCmpnstd = np.interp(freq_1d, frqTbl_1d, TblFreqChar_1d)
            ### you need to spline function in SciPy
    
    if SwPlot == 1:
        str = "*** Frequency Characteristics (" + StrCrct + "): Its inverse will be corrected. ***"
        print(str) 
        print("{}".format(str1))
        fig, ax = plt.subplots()
        plt.plot(frqTbl, TblFreqChar, 'b-',freq, FreqChardB_toBeCmpnstd, 'r--')
        ax.set_title(str)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Level (dB)')
        plt.show()

    CrctLinPwr = 10**(-FreqChardB_toBeCmpnstd/10) # in Linear Power. Checked 19 Apr 2016

    return CrctLinPwr, freq, FreqChardB_toBeCmpnstd

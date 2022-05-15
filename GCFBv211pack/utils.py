# -*- coding: utf-8 -*-
import sys
import numpy as np
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


def OutMidCrctFilt(StrCrct, SR, SwPlot, SwFilter):
    """Outer/middle ear compensation filter

    Args:
        StrCrct (string): String for Correction ELC/MAF/MAP
        SR (int): Sampling rate
        SwPlot (int): Switch of plot (0:OFF/1:ON)
        SwFilter (int): Switch of filter type
            1: FIR linear phase filter
            2: FIR mimimum phase filter (length: half of linear phase filter)

    Returns:
        FIRCoef (array): FIR filter coefficients
        StrFilt (string): Filter infomation
    """
    return FIRCoef, StrFilt
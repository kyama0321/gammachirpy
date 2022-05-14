# -*- coding: utf-8 -*-
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
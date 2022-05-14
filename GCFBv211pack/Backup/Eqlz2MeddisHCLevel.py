# -*- coding: utf-8 -*-
import numpy as np


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


def main():
    # Examples　
    fs = 48000
    f = 100
    T = 1
    t = np.array(range(T*fs))/fs
    SndIn = np.sin(2*np.pi*f*t)

    OutLeveldB_list = [30, 50, 70]
    for OutLeveldB in OutLeveldB_list:
        SndOut, AmpdB = Eqlz2MeddisHCLevel(SndIn, OutLeveldB)
        print("RMS of SndIn: {} --> {} dB SPL".format(np.sqrt(np.mean(SndIn**2)), AmpdB[2]))
        print("RMS of SndOut: {} --> {} dB SPL".format(np.sqrt(np.mean(SndOut**2)), OutLeveldB))
 
if __name__ == '__main__':
    main()

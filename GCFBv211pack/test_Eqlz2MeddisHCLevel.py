# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import utils

"""
   Matlab examples:
        rms(s(t)) == sqrt(mean(s.^2)) == 1   --> 30 dB SPL
        rms(s(t)) == sqrt(mean(s.^2)) == 10  --> 50 dB SPL
        rms(s(t)) == sqrt(mean(s.^2)) == 100 --> 70 dB SPL  
"""

# Examples
fs = 48000
f = 100
T = 1
t = np.array(range(T*fs))/fs
SndIn = np.sin(2*np.pi*f*t)

OutLeveldB_list = [30, 50, 70]
for OutLeveldB in OutLeveldB_list:
    SndOut, AmpdB = utils.Eqlz2MeddisHCLevel(SndIn, OutLeveldB)
    print("RMS of SndIn: {} --> {} dB SPL".format(np.sqrt(np.mean(SndIn**2)), AmpdB[2]))
    print("RMS of SndOut: {} --> {} dB SPL".format(np.sqrt(np.mean(SndOut**2)), OutLeveldB))


""" original code in testEqlz2MeddisHCLevel.m

SndSPLdB = 65
CalibToneRmsLeveldB = -26 # relative to rms = 1
InputRmsSPLdB = SndSPLdB - CalibToneRmsLeveldB

Snd0 = np.random.randn(1000,1)
Snd0 = Snd0/utils.rms(Snd0) # rms = 1
Snd = 10**(CalibToneRmsLeveldB/20) * Snd0

SndEq1, AmpdB1 = utils.Eqlz2MeddisHCLevel(Snd, SndSPLdB)
print("{}".format(AmpdB1))

fig, ax = plt.subplots(figsize=(8,4))
plt.plot(nn, SndEq1)
plt.show()

"""
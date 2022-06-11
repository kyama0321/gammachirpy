# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import utils
import GCFBv211 as gcfb_Main


class GCparam_default:
    fs = 48000
    NumCh = 100
    FRange = np.array([100, 6000])
    OutMidCrct = "No"
    # OutMidCrct = "ELC"
    # Ctrl = "dynamic" # used to be 'tyme-varying'
    # Ctrl = "static" # or 'fixed'


# Stiuli : a simple pulse train
fs = 48000
Tp = 10 # (ms) 100 Hz pulse train
Snd = np.array(([1]+[0]*int(Tp*fs/1000-1))*10)
Tsnd = len(Snd)/fs
print("Duration of sound = {} (ms)".format(Tsnd*1000))

SigSPLlist = np.arange(40, 100, 20)
cnt = 0

for SwDySt in range(2): # 1: only dynamic, 2: dynamic and static

    fig, ax = plt.subplots()

    for SwSPL in range(len(SigSPLlist)):
        SigSPL = SigSPLlist[SwSPL]
        Snd, _ = utils.Eqlz2MeddisHCLevel(Snd, SigSPL)

        # GCFB
        GCparam = GCparam_default() # reset all
        if SwDySt == 0: 
            GCparam.Ctrl = "dynamic"
            # GCparam.Ctrl = "static" # for checking
        else: 
            GCparam.Ctrl = "static"
        
        Tstart = time.time()
        cGCout, pGCout, GCparam, GCrest = gcfb_Main.GCFBv211(Snd, GCparam)

        Tend = time.time()
        print("Elapsed time is {} (sec) = {} times RealTime."\
            .format(np.round(Tend-Tstart, 4), np.round((Tend-Tstart)/Tsnd, 4)))

        ax = plt.subplot(len(SigSPLlist), 1, SwSPL+1)
        plt.imshow(np.maximum(cGCout, 0), aspect='auto', origin='lower', cmap='jet')
        ax.set_title("GCFB control = {}; Signal Level = {} dB SPL"\
            .format(GCparam.Ctrl, SigSPL))
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        plt.tight_layout()
        plt.pause(0.05)

plt.show()

"""
x = np.array(range(len(Snd)))/fs
fig, ax = plt.subplots(figsize=(8,4))
plt.plot(x, Snd)
ax.set_xlabel("Time [sec]")
ax.set_ylabel("Amplitude")
plt.ylim(-1.05, 1.05)
plt.show()
"""
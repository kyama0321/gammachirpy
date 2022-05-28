# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import utils
import GCFBv211 as gcfb_Main


class GCparam_default:
    fs = 48000
    NumCh = 100
    FRange = np.array([100, 6000])
    #OutMidCrct = "No"
    OutMidCrct = "ELC"
    # Ctrl = "dynamic" # used to be 'tyme-varying'
    # Ctrl = "static" # or 'fixed'


# Stiuli : a simple pulse train
fs = 48000
Tp = 10 # (ms) 100 Hz pulse train
Snd = np.array(([1]+[0]*int(Tp*fs/1000))*10)
Tsnd = len(Snd)/fs
print("Duration of sound = {} (ms)".format(Tsnd*1000))

SigSPLllst = np.arange(40, 100, 20)
cnt = 0
for SwDySt in range(1): # 0: only dynamic, 1: dynamic and static

    for SwSPL in range(len(SigSPLllst)):
        SigSPL = SigSPLllst[SwSPL]
        Snd, _ = utils.Eqlz2MeddisHCLevel(Snd, SigSPL)

        # GCFB
        GCparam = GCparam_default() # reset all
        if SwDySt == 0: 
            # GCparam.Ctrl = "dynamic"
            GCparam.Ctrl = "static"
        else: 
            GCparam.Ctrl = "Static"
        
        cGCout, pGCout, GCparam, GCrest = gcfb_Main.GCFBv211(Snd, GCparam)

        




#"""
x = np.array(range(len(Snd)))/fs
fig, ax = plt.subplots(figsize=(8,4))
plt.plot(x, Snd)
ax.set_xlabel("Time [sec]")
ax.set_ylabel("Amplitude")
plt.ylim(-1.05, 1.05)
plt.show()
#"""
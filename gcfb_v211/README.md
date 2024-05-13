# gcfb_v211

A sample-by-sample processing version of the dynamic compressive gammachirp filterbank

## Getting Started
The following instruction is based on **[gcfb_v211/demo_gcfb_v211_speech.ipynb](https://github.com/kyama0321/gammachirpy/blob/main/gcfb_v211/demo_gcfb_v211_speech.ipynb)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyama0321/gammachirpy/blob/main/gcfb_v211/demo_gcfb_v211_speech.ipynb)

1. Import packages.

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    import utils
    import gcfb_v211 as gcfb
    ```

2. Set default parameters for the gammachirp filterbank as class variables. Note that if you don't set any parameters, **gcfb.dcgc_v211()** automaticaly set default paramters determined in **gcfb.set_param()**

    ```python
    class GCparamDefault:
          fs = 48000 # sampling frequency
          num_ch = 100 # number of channels
          f_range = np.array([100, 6000]) # range of center frequencies
          out_mid_crct = 'ELC' # equal loudness contour (ELC)
          ctrl = 'dynamic' # time-varying parameter of dcGC-FB
    ```

3. Read an audio sound and normalize the signal's amplitude (-1 ~ +1). I recomend to use **utils.audioread()**. Note that the recommended sampling frequency of the input sound is 48,000 Hz.

    ```python
    # read the sample speech
    path_file = './sample/snd_gammachirpy.wav'
    snd, fs = utils.audioread(path_file)
    ```

4. Adjust the input signal level as a sound pressure level (SPL) by **utils.eqlz2meddis_hc_level()**.

    ```python
    # sound pressure level (SPL)
    dbspl = 40
    # Level equalization
    snd_eq, _ = utils.eqlz2meddis_hc_level(snd, dbspl)
    ```

5. Analyze the input signal by **gcfb.gcfb_v211()** with default parameters.

    ```python
    # GCFB
    gc_param = GCparamDefault()
    cgc_out, pgc_out, _, _ = gcfb.gcfb_v211(snd_eq, gc_param)
    ```

6. You can get the temporal output signals (num_ch $\times$ len(snd)) as :
   - **cgc_out**: outputs of the dynamic "compressive" gammachirp filterbank (dependent on the input signal level)
   - **pgc_out**: outputs of the "passive" gammachirp filterbank (not dependent on the input signal level)

7. If you change the SPL (**dbspl**), you can get and compare different outputs (**cgc_out**) from the dynamic compressive gammachirp filterbank. For example, the below figure is available at **[gcfb_v211/test_gcfb_v211_speech.py](https://github.com/kyama0321/gammachirpy/blob/main/gcfb_v211/test_gcfb_v211_speech.py)**.

<div style="text-align: center">
    <img src="../figs/gammachirpy_speech_dbspl.jpg" width="425px">
</div>
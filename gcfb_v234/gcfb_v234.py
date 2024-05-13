# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List

import utils
import gammachirp as gc


class ACFstatus:
    def __init__(self):
        self.num_ch = []
        self.num_filt = []
        self.lbz = []
        self.lap = []
        self.sig_in_prev = []
        self.sig_out_prev = []
        self.count = []

class ACFcoef:
    def __init__(self):
        self.fs = []
        self.ap = np.array([])
        self.bz = np.array([])

class cGCresp:
    def __init__(self):
        self.fr1 = []
        self.n = []
        self.b1 = []
        self.c1 = []
        self.frat = []
        self.b2 = []
        self.c2 = []
        self.n_frq_rsl = []
        self.pgc_frsp = []
        self.cgc_frsp = []
        self.cgc_nrm_frsp = []
        self.acf_frsp = []
        self.asym_func = []
        self.fp1 = []
        self.fr2 = []
        self.fp2 = []
        self.val_fp2 = []
        self.norm_fct_fp2 = []
        self.freq = []

class GCresp:
    def __init__(self):
        self.fr1 = []
        self.fr2 = []
        self.erb_space1 = []
        self.ef = []
        self.b1_val = []
        self.c1_val = []
        self.fp1 = []
        self.fp2 = []
        self.b2_val = []
        self.c1_val = []
        self.c2_val = []
        self.frat_val = []
        self.frat0_val = []
        self.frat1_val = []
        self.pc_hpaf = []
        self.frat0_pc = []

class LvlEst:
    def __init__(self):
        self.lct_erb = []
        self.decay_hl = []
        self.b2 = []
        self.c2 = []
        self.frat = []
        self.rms2spldb = []
        self.weight = []
        self.ref_db = []
        self.pwr = []
        self.exp_decay_val = []
        self.erb_space1 = []
        self.n_ch_shift = []
        self.n_ch_lvl_est = []
        self.lvl_lin_min_lim = []
        self.lvl_lin_ref = []

@dataclass
class DynHPAF:
    str_prc: str = 'frame-base' # default for gcfb_v234
    t_frame: float = 0.001 # 1ms <-- better than 5ms
    t_shift: float = 0.0005 # 0.5ms, fs = 1000 <-- more accurate than 1 ms
    len_frame: List[float] = field(default_factory=list)
    len_shift: List[float] = field(default_factory=list)
    fs: List[float] = field(default_factory=list)
    name_win: str = 'hanning'
    val_win: List[float] = field(default_factory=list)

@dataclass
class HLoss:
    f_audgram_list:  List[float] = field(default_factory=lambda: \
        np.array([125, 250, 500, 1000, 2000, 4000, 8000]))
    type: str = 'NH_NormalHearing'
    hearing_level_db: List[float] = field(default_factory=list)
    pin_loss_db_act: List[float] = field(default_factory=list)
    pin_loss_db_act_init: List[float] = field(default_factory=list)
    pin_loss_db_pas: List[float] = field(default_factory=list)
    io_func_loss_db_pas: List[float] = field(default_factory=list)
    compression_health: List[float] = field(default_factory=list)
    af_gain_cmpnst_db: List[float] = field(default_factory=list)
    hl_val_pin_cochlea_db: List[float] = field(default_factory=list)
    fb_fr1: List[float] = field(default_factory=list)
    fb_hearing_level_db: List[float] = field(default_factory=list)
    fb_pin_cochlea_db: List[float] = field(default_factory=list)
    fb_pin_loss_db_act: List[float] = field(default_factory=list)
    fb_pin_loss_db_act_init: List[float] = field(default_factory=list)
    fb_pin_loss_db_pas: List[float] = field(default_factory=list)
    fb_compression_health: List[float] = field(default_factory=list)
    compression_health_initval: List[float] = field(default_factory=list)
    fb_af_gain_cmpnst_db: List[float] = field(default_factory=list)


def gcfb_v234(snd_in, gc_param, *args):
    """Dynamic Compressive Gammachirp Filterbank (dcGC-FB)

    Args:
        snd_in (float): Input sound
        gc_param (struct): Parameters of dcGC-FB
            .fs: Sampling rate (default: 48000)
            .num_ch: Number of Channels (default: 100)
            .f_range: Frequency Range of GCFB (default: [100, 6000]) s
                     pecifying asymtopic freq. of passive GC (fr1)

    Returns:
        cgc_out: ompressive GammaChirp Filter Output
        pgc_out: Passive GammaChirp Filter Output
        gc_param (struct): Parameters of dcGC-FB
        gc_resp (struct): GC response result

    Note: 
        1)  This version is completely different from GCFB v.1.04 (obsolete).
            We introduced the "compressive gammachirp" to accomodate both the 
            psychoacoustical simultaneous masking and the compressive 
            characteristics (Irino and Patterson, 2001). The parameters were 
            determined from large dataset (See Patterson, Unoki, and Irino, 2003.)   

    References:
        Irino,T. and Unoki,M.:  IEEE ICASSP98, pp.3653-3656, May 1998.
        Irino,T. and Patterson,R.D. :  JASA, Vol.101, pp.412-419, 1997.
        Irino,T. and Patterson,R.D. :  JASA, Vol.109, pp.2008-2022, 2001.
        Patterson,R.D., Unoki,M. and Irino,T. :  JASA, Vol.114,pp.1529-1542,2003.
        Irino,T. and Patterson,R.D. : IEEE Trans.ASLP, Vol.14, Nov. 2006.
    """
    # Handling Input Parameters
    if len(args) > 0:
        help(gcfb_v234)
        sys.exit()

    size = np.shape(snd_in)
    if not len(size) == 1:
        print("Check snd_in. It should be 1 ch (Monaural) and  a single row vector.", file=sys.stderr)
        sys.exit(1)
    len_snd = len(snd_in)
    
    # Call default parametes
    gc_param, gc_resp = set_param(gc_param)
    fs = gc_param.fs
    num_ch = gc_param.num_ch

    """
    Outer-Mid Ear Compensation
    for inverse filter, use Out utits.out_mid_crctFilt('ELC', fs, 0, 1)
    """
    if gc_param.out_mid_crct == 'No':
        gc_param.field2cochlea = 'No Outer/Middle Ear correction'
        print("*** No Outer/Middle Ear correction ***")
        snd = snd_in
    else:
        # if gc_param.out_mid_crct in ["ELC", "MAF", "MAP"]:
        print("*** Outer/Middle Ear correction (minimum phase) : " \
            + f"{gc_param.out_mid_crct} ***")
        cmpnst_out_mid, param_m2c = utils.mk_filter_field2cochlea(gc_param.out_mid_crct, fs, sw_fwd_bwd=1, sw_plot=0)
        # cmpnst_out_mid, _ = utils.out_mid_crct_filt(gc_param.out_mid_crct, fs, 0, 2) # 2) minimum phase
        # 1kHz: -4 dB, 2kHz: -1 dB, 4kHz: +4 dB (ELC)
        # Now we use Minimum phase version of out_mid_crctFilt (modified 16 Apr. 2006).
        # No compensation is necessary.  16 Apr. 2006
        snd = signal.lfilter(cmpnst_out_mid, 1, snd_in)
        gc_param.field2cochlea = param_m2c

    """
    Gammachirp
    """
    print("*** Gammachirp Calculation ***")

    sw_fast_prcs = 1 # ON: fast processing for static filter
    if not sw_fast_prcs == 1:
        print("sw_fast_prcs should be 1.", file=sys.stderr)
        sys.exit(1)
    if sw_fast_prcs == 1 and gc_param.ctrl == "static":
        # 'Fast processing for linear cGC gain at gc_param.level_db_scgcfb'
        """
        for HP-AF
        """
        lvl_db = gc_param.level_db_scgcfb
        gc_resp.lvl_db = lvl_db
        frat_val = gc_param.frat0_pc + gc_param.frat1_val * (lvl_db - gc_resp.pc_hpaf)
        fr2_val = frat_val * gc_resp.fp1
        gc_resp.fr2 = fr2_val.copy()
        acf_coef_fixed = make_asym_cmp_filters_v2(fs, fr2_val, gc_resp.b2_val, gc_resp.c2_val)
    else:
        # HP-AF for dynamic-GC level estimation path. 18 Dec 2012 Checked
        gc_resp.lvl_db = [] # initialize
        fr2lvl_est = gc_param.lvl_est.frat * gc_resp.fp1
        # default gc_param.lvl_est.frat = 1.08 (GCFBv208_SetParam(gc_param))
        # ---> Linear filter for level estimation
        #acf_coef_fixed = make_asym_cmp_filters_v2(fs, fr2lvl_est, gc_param.lvl_est.b2, gc_param.lvl_est.c2)

        # 26 Jul 2020; compression health
        c2_val_compression_health = gc_param.hloss.fb_compression_health * gc_param.lvl_est.c2
        acf_coef_fixed = make_asym_cmp_filters_v2(fs, fr2lvl_est, gc_param.lvl_est.b2, c2_val_compression_health)

    """
    Start calculation
    """
    """
    Passive Gammachirp & Levfel estimation filtering
    """
    t_start = time.time()
    cgc_out = np.zeros((num_ch, len_snd)) # old
    pgc_out = np.zeros((num_ch, len_snd)) # old

    pgc_smpl = np.zeros((num_ch, len_snd)) # passive GC
    scgc_smpl = np.zeros((num_ch, len_snd)) # static GC
    p_pgc = np.zeros((num_ch, len_snd))
    cgc_out_lvl_est = np.zeros((num_ch, len_snd))

    print("--- Channel-by-channel processing ---")

    for nch in range(num_ch):
        # passive gammachirp
        pgc, _, _, _ = gc.gammachirp(gc_resp.fr1[nch], fs, gc_param.n, \
                                       gc_resp.b1_val[nch], gc_resp.c1_val[nch], 0, '', 'peak')

        # fast FFT-based filtering by the pgc
        pgc_smpl[nch, 0:len_snd] = utils.fftfilt(pgc[0,:], snd) 

        # Fixed HP-AF filtering for level setting
        # Note (13 May 2020): 4 times of second-order filtering is 
        # comparable to 1 time 8th-order filtering in processing time.
        gc_smpl1 = pgc_smpl[nch, :].copy()
        for n_filt in range(4):
            gc_smpl1 = signal.lfilter(acf_coef_fixed.bz[nch, :, n_filt], \
                                      acf_coef_fixed.ap[nch, :, n_filt], gc_smpl1)
        scgc_smpl[nch, :] = gc_smpl1.copy() # static compressive GC output : sample-by-sample

        # Fast processing for fixed cGC
        if gc_param.ctrl == 'static': # Static
            if nch == 0: # first channel
                str_gc = "Static (Fixed) Compressive-Gammachirp"
            gc_resp.fp2[nch], _ = fr1_to_fp2(gc_param.n, gc_resp.b1_val[nch], gc_resp.c1_val[nch], \
                                                 gc_resp.b2_val[nch], gc_resp.c2_val[nch], \
                                                 frat_val[nch], gc_resp.fr1[nch])
            if nch == num_ch: # final channel --> make it a vector
                gc_resp.fp2 = gc_resp.fp2

        else: # Level estimation pass for Dynamic.
            str_gc = "Passive-Gammachirp*Fixed HP-AF = Level estimation filter"

        if nch == 0 or np.mod(nch+1, 50) == 0: # 20ch --> 50ch (23 Oct 2022)
            t_now = time.time()
            print(str_gc)
            print(f"ch #{nch+1}" + f" / #{num_ch}. elapsed time = {np.round(t_now-t_start, 1)} (sec)")

    """
    Filtering of Dynamic HP-AF
    """
    if gc_param.ctrl == 'dynamic':

        if 'sample' in gc_param.dyn_hpaf.str_prc: 
            # sample-based dcGC
            if not 'NH' in gc_param.dyn_hpaf.type:
                print(f'The output of GCFBv23_SampleBase has not been checked for {gc_param.hloss.type}')

            dcgc_smpl, gc_resp = gcfb_v23_sample_base(pgc_smpl, scgc_smpl, gc_param, gc_resp)
            cgc_out = dcgc_smpl # sample output

        elif 'frame' in gc_param.dyn_hpaf.str_prc:
            # frame-based dcGC
            dcgc_frame, gc_resp = gcfb_v23_frame_base(pgc_smpl, scgc_smpl, gc_param, gc_resp)
            cgc_out = dcgc_frame # frame output
            
        else:
            raise ValueError('Specifiry "gc_param.dyn_hpaf.str_prc" properly: "sample" or "frame"')
    
    elif gc_param.ctrl == 'static':
        cgc_out = scgc_smpl.copy()
    else:
        raise ValueError('Specifiy "gc_param.ctrl" properly')

    """
    Signal path Gain Normalization at Reference Level (gain_ref_db) for static dynamic filters
    """
    _, len_out = cgc_out.shape
    if not isinstance(gc_param.gain_ref_db, str):
        frat_ref = gc_param.frat0_pc + gc_param.frat1_val * (gc_param.gain_ref_db - gc_resp.pc_hpaf)
        cgc_ref = cmprs_gc_frsp(gc_resp.fr1, fs, gc_param.n, 
                                gc_resp.b1_val, gc_resp.c1_val, frat_ref, 
                                gc_resp.b2_val, gc_resp.c2_val)
        gc_resp.gain_factor = 10**(gc_param.gain_cmpnst_db/20) * cgc_ref.norm_fct_fp2
        gc_resp.cgc_ref = cgc_ref
        # gc_resp.lvl_db = lvl_db

        dcgc_out = (gc_resp.gain_factor * np.ones((1, len_out))) * cgc_out

    elif 'NormIOfunc' in gc_param.gain_ref_db:

        gain_factor = 10**(-(gc_param.hloss.fb_af_gain_cmpnst_db)/20) # HL 0 dB == HL val dB
        dcgc_out = (gain_factor * np.ones((1, len_out))) * cgc_out
        # rms 0 dB coressponds to SPL 30 dB of Meddis IHC Level (17 Aug 2021)

    else:
        raise ValueError('Set "gc_param.gain_ref_db" properly')

    return dcgc_out, scgc_smpl, gc_param, gc_resp


def gcfb_v23_frame_base(pgc_smpl, scgc_smpl, gc_param, gc_resp):
    """ Frame-based processing of HP-AF of dcGCFB

    Args:
        pgc_smpl (array_like): passive GC sample
        scgc_smpl (array_like): static cGC sample
        gc_param (struct): Parameters of dcGC-FB
        gc_resp (struct): GC response result

    Returns:
        dcgc_smpl: sample-level output of dcGC-FB
        gc_resp: updated GC response result
    """
    exp_decay_frame = gc_param.lvl_est.exp_decay_val ** (gc_param.dyn_hpaf.len_shift)
    num_ch, len_snd = pgc_smpl.shape

    print("--- Frame base processing ---")
    t_start = time.time()

    n_frq_rsl = 1024 * 2 # for normalization
    c2_val_cmprs_hlth = gc_param.hloss.fb_compression_health * gc_param.lvl_est.c2

    scgc_resp = cmprs_gc_frsp(gc_param.fr1, gc_param.fs, gc_param.n, 
                              gc_resp.b1_val, gc_resp.c1_val, gc_param.lvl_est.frat, 
                              gc_param.lvl_est.b2, c2_val_cmprs_hlth, n_frq_rsl)

    for n_ch in range(num_ch):
        """
        signal path
        """
        # pgc only
        pgc_frame_mtrx, _ = set_frame4time_sequence(pgc_smpl[n_ch, :], 
                                                    gc_param.dyn_hpaf.len_frame, 
                                                    gc_param.dyn_hpaf.len_shift)
        len_win, len_frame = pgc_frame_mtrx.shape

        # cGC level estimation filter -- This BW is narrower than that of pGC.
        # roughly at pgc = 50
        scgc_frame_mtrx, _ = set_frame4time_sequence(scgc_smpl[n_ch, :], 
                                                     gc_param.dyn_hpaf.len_frame, 
                                                     gc_param.dyn_hpaf.len_shift)

        if n_ch == 0:
            pgc_frame = np.zeros((num_ch, len_frame))
            lvl_db_frame = np.zeros((num_ch, len_frame))
            frat_frame = np.zeros((num_ch, len_frame))
            asym_func_gain = np.zeros((num_ch, len_frame))
            dcgc_frame = np.zeros((num_ch, len_frame))
            scgc_frame = np.zeros((num_ch, len_frame))
        
        # weighted mean
        pgc_frame[n_ch, 0:len_frame] = np.sqrt(np.dot(gc_param.dyn_hpaf.val_win, pgc_frame_mtrx ** 2))
        scgc_frame[n_ch, 0:len_frame] = np.sqrt(np.dot(gc_param.dyn_hpaf.val_win, scgc_frame_mtrx ** 2))

        """
        level estimation path
        """
        lvl_lin1_frame_mtrx, _ = set_frame4time_sequence(pgc_smpl[int(gc_param.lvl_est.n_ch_lvl_est[n_ch]-1), :], 
                                                         gc_param.dyn_hpaf.len_frame, 
                                                         gc_param.dyn_hpaf.len_shift)
        lvl_lin1_frame = np.sqrt(np.dot(gc_param.dyn_hpaf.val_win, lvl_lin1_frame_mtrx ** 2))

        lvl_lin2_frame_mtrx, _ = set_frame4time_sequence(scgc_smpl[int(gc_param.lvl_est.n_ch_lvl_est[n_ch]-1), :], 
                                                         gc_param.dyn_hpaf.len_frame, 
                                                         gc_param.dyn_hpaf.len_shift)
        lvl_lin2_frame = np.sqrt(np.dot(gc_param.dyn_hpaf.val_win, lvl_lin2_frame_mtrx ** 2))

        for n_frame in range(len_frame-1): # compensation of decay constan
            lvl_lin1_frame[n_frame+1] = np.maximum(lvl_lin1_frame[n_frame+1], lvl_lin1_frame[n_frame] * exp_decay_frame)
            lvl_lin2_frame[n_frame+1] = np.maximum(lvl_lin2_frame[n_frame+1], lvl_lin2_frame[n_frame] * exp_decay_frame)

        lvl_lin_ttl_frame = gc_param.lvl_est.weight \
            * gc_param.lvl_est.lvl_lin_ref * (lvl_lin1_frame/gc_param.lvl_est.lvl_lin_ref)**gc_param.lvl_est.pwr[0] \
                + (1 - gc_param.lvl_est.weight) \
                    * gc_param.lvl_est.lvl_lin_ref * (lvl_lin2_frame/gc_param.lvl_est.lvl_lin_ref)**gc_param.lvl_est.pwr[1]

        cmpnst_half_wave_rectiry = -3 # Cmpensation of a halfwave rectification which was used in "sample-by-sample."
        lvl_db_frame[n_ch, 0:len_frame] = 20 * np.log10(np.maximum(lvl_lin_ttl_frame, gc_param.lvl_est.lvl_lin_min_lim)) \
                                                         + gc_param.lvl_est.rms2spldb + cmpnst_half_wave_rectiry
        

        af_out_db, _, gc_param = gcfb_v23_asym_func_in_out(gc_param, gc_resp,
                                                            gc_param.fr1[n_ch],
                                                            gc_param.hloss.fb_compression_health[n_ch],
                                                            lvl_db_frame[n_ch, :])
        asym_func_gain[n_ch, 0:len_frame] = 10**(af_out_db/20) # default
        
        # normaliztion (peak of scgc_frame should be 0 dB at every frequencies)
        scgc_frame1 = scgc_resp.norm_fct_fp2[n_ch] * scgc_frame[n_ch, :]
        dcgc_frame[n_ch, 0:len_frame] = asym_func_gain[n_ch, :] * scgc_frame1

        if n_ch == 0 or np.mod(n_ch+1, 50) == 0:
            t_now = time.time()
            print(f"Frame-based HP-AF: ch #{n_ch+1} / #{num_ch}. \
                  elapsed time = {np.round(t_now-t_start, 1)} (sec)")
            
    # update data
    gc_resp.lvl_db_frame = lvl_db_frame
    gc_resp.pgc_frame = pgc_frame
    gc_resp.scgc_frame = scgc_frame
    gc_resp.frat_frame = frat_frame
    gc_resp.asym_func_gain = asym_func_gain

    return dcgc_frame, gc_resp


def gcfb_v23_sample_base(pgc_smpl, scgc_smpl, gc_param, gc_resp):
    """ Sample by Sample processing of HP-AF of dcGCFB

    Args:
        pgc_smpl (array_like): passive GC sample
        scgc_smpl (array_like): static cGC sample
        gc_param (struct): Parameters of dcGC-FB
        gc_resp (struct): GC response result

    Returns:
        dcgc_smpl: sample-level output of dcGC-FB
        gc_resp: updated GC response result
    """    
    # Initial settings
    fs = gc_param.fs
    num_ch, len_snd = scgc_smpl.shapre
    num_disp = int(np.fix(len_snd/10)) # display 10 times per snd (29 Jan. 2015)
    cgc_smpl = np.zeros((num_ch, len_snd))
    gc_resp.fr2 = np.zeros((num_ch, len_snd))
    gc_resp.frat_val = np.zeros((num_ch, len_snd))
    gc_resp.fp2 = []
    lvl_db = np.zeros((num_ch, len_snd))
    lvl_lin = np.zeros((num_ch, 2))
    lvl_lin_prev = np.zeros((num_ch, 2))

    # Sample-by-sample processing
    print("--- Sample-by-sample processing ---")
    t_start = time.time()

    for nsmpl in range(len_snd):

        """
        Level estimation circuit
        """
        lvl_lin[0:num_ch, 0] = \
            np.maximum(np.max(pgc_smpl[gc_param.lvl_est.n_ch_lvl_est.astype(int)-1, nsmpl], initial=0, axis=1), \
                lvl_lin_prev[:, 0]*gc_param.lvl_est.exp_decay_val)
        lvl_lin[0:num_ch, 1] = \
            np.maximum(np.max(scgc_smpl[gc_param.lvl_est.n_ch_lvl_est.astype(int)-1, nsmpl], initial=0, axis=1), \
                lvl_lin_prev[:, 1]*gc_param.lvl_est.exp_decay_val)

        lvl_lin_prev = lvl_lin.copy()

        lvl_lin_ttl = gc_param.lvl_est.weight \
            * gc_param.lvl_est.lvl_lin_ref * (lvl_lin[:, 0] / gc_param.lvl_est.lvl_lin_ref)**gc_param.lvl_est.pwr[0] \
                + (1 - gc_param.lvl_est.weight) \
                    * gc_param.lvl_est.lvl_lin_ref * (lvl_lin[:, 1] / gc_param.lvl_est.lvl_lin_ref)**gc_param.lvl_est.pwr[1]
            
        lvl_db[:, [nsmpl]] = np.array([20 * np.log10(np.maximum(lvl_lin_ttl, gc_param.lvl_est.lvl_lin_min_lim)) \
            + gc_param.lvl_est.rms2spldb]).T

        """
        Signal path
        """
        # Filtering High-Pass Asymmetric Comressive Filter
        #frat_val = gc_param.frat[0, 0] + gc_param.frat[0, 1] * gc_resp.ef[:] + \
        #    (gc_param.frat[1, 0] + gc_param.frat[1, 1] * gc_resp.ef[:]) * lvl_db[:, [nsmpl]]
        frat_val = gc_resp.frat0_pc + gc_param.hloss.fb_compression_health * gc_resp.frat1val \
                    * (lvl_db[:, nsmpl] - gc_resp.pc_hpaf)
        fr2_val = gc_resp.fp1[:] * frat_val

        if np.mod(nsmpl, gc_param.num_update_asym_cmp) == 0: # update periodically
            acf_coef = make_asym_cmp_filters_v2(fs, fr2_val, gc_resp.b2_val, gc_resp.c2_val)

        if nsmpl == 0:
            _, acf_status = acfilterbank(acf_coef, []) # initialization

        sig_out, acf_status = acfilterbank(acf_coef, acf_status, pgc_smpl[:, nsmpl])
        cgc_smpl[:, [nsmpl]] = sig_out.copy()
        gc_resp.fr2[:, [nsmpl]] = fr2_val.copy()
        gc_resp.frat_val[:, [nsmpl]] = frat_val.copy()

        if nsmpl == 0 or np.mod(nsmpl+1, num_disp) == 0:
            t_now = time.time()
            print(f"Dynamic Compressive-Gammachirp: Time {np.round(nsmpl/fs*1000, 1)} (ms) / "\
                    + f"{np.round(len_snd/fs*1000, 1)} (ms). elapsed time = {np.round(t_now-t_start, 1)} (sec)")

    return cgc_smpl, gc_resp


def set_param(gc_param):
    """Setting Default Parameters for GCFBv234

    Args:
        gc_param (struct): Your preset gammachirp parameters
            .fs: Sampling rate (default: 48000)
            .num_ch: Number of Channels (default: 100)
            .f_range: Frequency Range of GCFB (default: [100, 6000]) 
                     specifying asymtopic freq. of passive GC (fr1)

    Returns:
        gc_param (struct): gc_param values
    """
    if not hasattr(gc_param, 'fs'):
        gc_param.fs = 48000

    if not hasattr(gc_param, 'out_mid_crct'):
        gc_param.out_mid_crct = 'ELC'
        # if no out_mid_crct is not necessary, specify gc_param.out_mid_crct = 'no'

    if not hasattr(gc_param, 'num_ch'):
        gc_param.num_ch = 100

    if not hasattr(gc_param, 'f_range'):
        gc_param.f_range = np.array([100, 6000])

    if gc_param.f_range[1]*3 > gc_param.fs:
        print('GCFB may not work properly when max(FreqRange)*3 > fs')
        input('---> Set fs properly.   OR  If you wish to continue as is, press RETURN >')
    
    # Gammachirp parameters
    if not hasattr(gc_param, 'n'):
        gc_param.n = 4 # default gammatone & gammachirp

    # Convention
    if not hasattr(gc_param, 'b1'):
        gc_param.b1 = np.array([1.81, 0]) # b1 becomes two coeffs in v210 (18 Apr. 2015). Frequency independent by 0. 

    if len(gc_param.b1) == 1:
        gc_param.b1.append(0) # frequency independent by 0

    if not hasattr(gc_param, 'c1'):
        gc_param.c1 = np.array([-2.96, 0]) # c1 becomes two coeffs. in v210 (18 Apr. 2015). Frequency independent by 0.

    if len(gc_param.c1) == 1:
        gc_param.c1.append(0) # frequency independent by 0
    
    if not hasattr(gc_param, 'frat'):
        gc_param.frat = np.array([[0.4660, 0], [0.0109, 0]])
    
    if not hasattr(gc_param, 'b2'):
        gc_param.b2 = np.array([[2.17, 0], [0, 0]]) # no level-dependency  (8 Jul 05)

    if not hasattr(gc_param, 'c2'):
        gc_param.c2 = np.array([[2.20, 0], [0, 0]]) # no level-dependency; no freq-dependency (3 Jun 05)

    if not hasattr(gc_param, 'ctrl'):
        gc_param.ctrl = 'dynamic' # default (28 Feb 2021)
    if 'fix' in gc_param.ctrl:
        gc_param.ctrl = 'static'
    if 'tim' in gc_param.ctrl:
        gc_param.ctrl = 'dynamic'

    if not 'sta' in gc_param.ctrl and not 'dyn' in gc_param.ctrl and not 'lev' in gc_param.ctrl:
        print("Specify gc_param.ctrl:  'static', 'dynamic', or 'level(-estimation). \
               (old version 'fixed'/'time-varying')", file=sys.stderr)
        sys.exit(1)

    if not hasattr(gc_param, 'gain_cmpnst_db'):
        gc_param.gain_cmpnst_db = -1 # in dB. when lvl_est.c2==2.2, 1 July 2005

    """
    Parameters for level estimation
    """
    if hasattr(gc_param, 'p_pgc_ref') or hasattr(gc_param, 'lvl_ref_db'):
        print("The parameter 'gc_param.p_pgc_ref' is obsolete.")
        print("The parameter 'gc_param.lvl_ref_db' is obsolete.")
        print("Please change it to 'gc_param.gain_ref_db'", file=sys.stderr)
        sys.exit(1)
    
    if not hasattr(gc_param, 'gain_ref_db'):
        # gc_param.gain_ref_db = 50 # reference p_pgc level for gain normalization (~v221)
        gc_param.gain_ref_db = 'NormIOfunc' # New default v230~ (25 Jul. 2020)

    if not hasattr(gc_param, 'level_db_scgcfb'):
        gc_param.level_db_scgcfb = 50 # use it as default for static-compressive GC (scGCFB)

    if not hasattr(gc_param, 'lvl_est'):
        gc_param.lvl_est = LvlEst()

    if gc_param.lvl_est.lct_erb == []:
        #gc_param.lvl_est.lct_erb = 1.0
        # Location of Level Estimation pGC relative to the signal pGC in ERB
        # see testGC_lct_erb.py for fitting result. 10 Sept 2004
        gc_param.lvl_est.lct_erb = 1.5;   # 16 July 05

    if gc_param.lvl_est.decay_hl == []:
        gc_param.lvl_est.decay_hl = 0.5; # 18 July 2005

    if gc_param.lvl_est.b2 == []:
        gc_param.lvl_est.b2 = gc_param.b2[0, 0]

    if gc_param.lvl_est.c2 == []:
        gc_param.lvl_est.c2 = gc_param.c2[0, 0]

    if gc_param.lvl_est.frat == []:
        # gc_param.lvl_est.frat = 1.1 #  when b=2.01 & c=2.20
        gc_param.lvl_est.frat = 1.08 # peak of cGC ~= 0 dB (b2=2.17 & c2=2.20)

    if gc_param.lvl_est.rms2spldb == []:
        gc_param.lvl_est.rms2spldb = 30 # 1 rms == 30 dB SPL for Meddis IHC
        gc_param.meddis_hc_level_rms0db_spldb = 30

    if gc_param.lvl_est.weight == []:
        gc_param.lvl_est.weight = 0.5

    if gc_param.lvl_est.ref_db == []:
        gc_param.lvl_est.ref_db = 50 # 50 dB SPL

    if gc_param.lvl_est.pwr == []:
        gc_param.lvl_est.pwr = np.array([1.5, 0.5]) # weight for pGC & cGC

    # new 19 Dec 2011
    if not hasattr(gc_param, 'num_update_asym_cmp'):
        # gc_param.num_update_asym_cmp = 3 # updte every 3 samples (== 3*GCFBv207)
        gc_param.num_update_asym_cmp = 1 # samply-by-sample (==GCFBv207)

    """
    new 13 May 2020
    """
    if not hasattr(gc_param, 'dyn_hpaf'):
        gc_param.dyn_hpaf = DynHPAF() # initial settings
        gc_param.dyn_hpaf.len_frame = int(np.fix(gc_param.dyn_hpaf.t_frame * gc_param.fs))
        gc_param.dyn_hpaf.len_shift = int(np.fix(gc_param.dyn_hpaf.t_shift * gc_param.fs))
        gc_param.dyn_hpaf.t_frame = gc_param.dyn_hpaf.len_frame / gc_param.fs # re-calculation
        gc_param.dyn_hpaf.t_shift = gc_param.dyn_hpaf.len_shift / gc_param.fs # re-calculation
        gc_param.dyn_hpaf.fs = 1 / gc_param.dyn_hpaf.t_shift
        # sample/frame-processing
        if 'sample' in gc_param.dyn_hpaf_str_prc:
            # sample-based
            gc_param.dyn_hpaf.str_prc = 'sample-base'
        else:
            # frame-based
            gc_param.dyn_hpaf.str_prc = 'frame-base'
            if gc_param.dyn_hpaf.name_win == 'hanning':
                # np.hanning: does not match to Matlab 'hannig()'
                # https://stackoverflow.com/questions/56485663/hanning-window-values-doesnt-match-in-python-and-matlab
                val_win = np.hanning(gc_param.dyn_hpaf.len_frame+2)[1:-1]
            elif gc_param.dyn_hpaf.name_win == 'hamming':
                # np.hamming: matces to Matlab 'hamming'
                val_win = np.hamming(gc_param.dyn_hpaf.len_frame)
            else:
                print("Select window function: hannig/hamming", file=sys.stderr)
                sys.exit(1)
            gc_param.dyn_hpaf.val_win = val_win / sum(val_win) # normalization

    """
    GCresp
    """
    gc_resp = GCresp()
    
    fr1, erb_rate1 = utils.equal_freq_scale('ERB', gc_param.num_ch, gc_param.f_range)
    gc_param.fr1 = np.array([fr1]).T
    gc_resp.fr1 = np.array([fr1]).T
    gc_resp.erb_space1 = np.mean(np.diff(erb_rate1))
    erb_rate, _ = utils.freq2erb(gc_resp.fr1)
    erb_rate_1kHz, _ = utils.freq2erb(1000)
    gc_resp.ef = erb_rate/erb_rate_1kHz - 1

    one_vec = np.ones([gc_param.num_ch, 1])
    gc_resp.b1_val = gc_param.b1[0]*one_vec + gc_param.b1[1]*gc_resp.ef
    gc_resp.c1_val = gc_param.c1[0]*one_vec + gc_param.c1[1]*gc_resp.ef

    gc_resp.fp1, _ = gc.fr2fpeak(gc_param.n, gc_resp.b1_val, gc_resp.c1_val, gc_resp.fr1)
    gc_resp.fp2 = np.zeros(np.shape(gc_resp.fp1))

    gc_resp.b2_val = gc_param.b2[0, 0]*one_vec + gc_param.b2[0, 1]*gc_resp.ef
    gc_resp.c2_val = gc_param.c2[0, 0]*one_vec + gc_param.c2[0, 1]*gc_resp.ef

    # new parameters for HP-AF (23 May 2020)
    gc_resp.frat0_val = gc_param.frat[0, 0]*one_vec + gc_param.frat[0, 1]*gc_resp.ef
    gc_resp.frat1_val = gc_param.frat[1, 0]*one_vec + gc_param.frat[1, 1]*gc_resp.ef

    gc_resp.pc_hpaf = (1 - gc_resp.frat0_val) / gc_resp.frat1_val # center level for HP-AF
    gc_resp.frat0_pc = gc_resp.frat0_val + gc_resp.frat1_val * gc_resp.pc_hpaf

    """
    Hearing Loss
    """
    gc_param = gcfb_v23_hearing_loss(gc_param, gc_resp)
        
    """
    Set Params estimation circuit
    """
    # keep lvl_est params  3 Dec 2013
    exp_decay_val = np.exp(-1/(gc_param.lvl_est.decay_hl*gc_param.fs/1000)*np.log(2)) # decay exp
    n_ch_shift = np.round(gc_param.lvl_est.lct_erb/gc_resp.erb_space1)
    n_ch_lvl_est = np.minimum(np.maximum(1, np.array([np.arange(gc_param.num_ch)+1]).T+n_ch_shift), \
                           gc_param.num_ch) # shift in num_ch [1:num_ch]
    lvl_lin_min_lim = 10**(-gc_param.lvl_est.rms2spldb/20) # minimum sould be SPL 0 dB
    lvl_lin_ref = 10**((gc_param.lvl_est.ref_db - gc_param.lvl_est.rms2spldb)/20)

    gc_param.lvl_est.exp_decay_val = exp_decay_val
    gc_param.lvl_est.erb_space1 = gc_resp.erb_space1
    gc_param.lvl_est.n_ch_shift = n_ch_shift
    gc_param.lvl_est.n_ch_lvl_est = n_ch_lvl_est
    gc_param.lvl_est.lvl_lin_min_lim = lvl_lin_min_lim
    gc_param.lvl_est.lvl_lin_ref = lvl_lin_ref

    return gc_param, gc_resp


def gcfb_v23_hearing_loss(gc_param, gc_resp):
    """ Calculate GC Hearing Loss parameters (gc_param.hloss.*)

    Args:
        gc_param (structure): parameters of dcGC-FB, necessary parameters are as follows:
            .hloss_fraudgram_list (array_like): frequency axis of audiogram
            .hloss_hearing_level (array_like): hearing level of audiogram
            .hloss_compression_health (float): compression health factor defined by gcfb_v23* (0~1)
        gc_resp (structure): parameters of dcGC-FB, necessary parameters is as follows:
            .fp1

    Returns:
        gc_param (structure): updated parameters

    Note:
        Parameters of pGC for HL are same to NH. HP-AF only changes. 
    
    """    

    gc_param.hloss = HLoss() # initalize

    len_fag = len(gc_param.hloss.f_audgram_list)

    # normal hearing
    if 'NH' in gc_param.hloss_type:
        gc_param.hloss.type = 'NH_NormalHearing'
        gc_param.hloss.hearing_level_db = np.zeros((len_fag))

        # compression health factor
        if not hasattr(gc_param, 'hloss_compression_health'):
            gc_param.hloss.compression_health = np.ones((len_fag)) # default: 100%
        else:
            gc_param.hloss.compression_health = gc_param.hloss_compression_health *  np.ones((len_fag))

        # audiogram 
        # gc_param.hloss.type == 'NH_NormalHearing' # default settings in HLoss()

    # hearing loss
    elif 'HL' in gc_param.hloss_type:
        # compression health factor
        if not hasattr(gc_param, 'hloss_compression_health'):
            gc_param.hloss.compression_health = 0.5 * np.ones((len_fag)) # default: 50%
        else:
            gc_param.hloss.compression_health = gc_param.hloss_compression_health *  np.ones((len_fag))

        # audiogram
        if 'HL0' in gc_param.hloss_type: # manual settings
            gc_param.hloss.type = 'HLval_ManualSet'
            
            # check input: length
            if not len(gc_param.hloss_hearing_level_db) == len_fag:
                raise ValueError('Set gc_param.hloss_hearing_level_db at f_audgram_list in advance.')
            else:
                # check input: isPositive
                if np.any(gc_param.hloss_hearing_level_db < 0):
                    raise ValueError('gc_param.hloss_hearing_level_db must not be negative')
                else:
                    gc_param.hloss.hearing_level_db = gc_param.hloss_hearing_level_db

        elif 'HL1' in gc_param.hloss_type: # 'HL1_Example'; preset example
            gc_param.hloss.type = 'HL1_Example'
            gc_param.hloss.hearing_level_db = np.array([10, 4, 10, 13, 48, 58, 79])

        elif 'HL2' in gc_param.hloss_type: # 'HL2_Tsuiki2002_80yr'
            gc_param.hloss.type = 'HL2_Tsuiki2002_80yr' 
            gc_param.hloss.hearing_level_db = np.array([23.5, 24.3, 26.8,  27.9,  32.9,  48.3,  68.5])

        elif 'HL3' in gc_param.hloss_type: # 'HL3_ISO7029_70yr_male'
            gc_param.hloss.type = 'HL3_ISO7029_70yr_male'  
            gc_param.hloss.hearing_level_db = np.array([8, 8, 9, 10, 19, 43, 59])

        elif 'HL4' in gc_param.hloss_type: # HL4_ISO7029_70yr_female'
            gc_param.hloss.type = 'HL4_ISO7029_70yr_female' 
            gc_param.hloss.hearing_level_db = np.array([8, 8, 9, 10, 16, 24, 41])

        elif 'HL5' in gc_param.hloss_type: # 'HL5_ISO7029_60yr_male'
            gc_param.hloss.type = 'HL5_ISO7029_60yr_male' 
            gc_param.hloss.hearing_level_db = np.array([5, 5, 6, 7, 12, 28, 39])

        elif 'HL6' in gc_param.hloss_type: # 'HL6_ISO7029_60yr_male'
            gc_param.hloss.type = 'HL6_ISO7029_60yr_male' 
            gc_param.hloss.hearing_level_db = np.array([5, 5, 6, 7, 11, 16, 26])

        elif 'HL7' in gc_param.hloss_type: # 'HL7_Example_Otosclerosis'
            gc_param.hloss.type = 'HL7_Example_Otosclerosis'
            gc_param.hloss.hearing_level_db = np.array([50, 55, 50, 50, 40, 25, 20])

        elif 'HL8' in gc_param.hloss_type: # 'HL8_Example_NoiseInduced'
            gc_param.hloss.type = 'HL8_Example_NoiseInduced' 
            gc_param.hloss.hearing_level_db = np.array([15, 10, 15, 10, 10, 40, 20])

        else:
            raise ValueError('Specify GCparam.HLoss.Type (HL0, HL1, HL2, ....) properly.')

    else:
        raise ValueError('Specify GCparam.HLoss.Type (NH, HL0, HL1, HL2, ....) properly.')
    
    if len(gc_param.hloss.compression_health) == 1:
        gc_param.hloss.compression_health = gc_param.hloss.compression_health * np.ones((len_fag))

    # set parameters of heairng loss
    gc_param.hloss.compression_health_initval = gc_param.hloss.compression_health # keep initial values
    gc_param.hloss.af_gain_cmpnst_db = np.zeros(len_fag)

    hl0_pin_cochlea_db = np.zeros(len_fag)
    hl_val_pin_cochlea_db = np.zeros(len_fag)
    pin_loss_db_act = np.zeros(len_fag)
    pin_loss_db_act_init = np.zeros(len_fag)
    pin_loss_db_pas = np.zeros(len_fag)
    io_func_loss_db_pas = np.zeros(len_fag)

    eps = sys.float_info.epsilon

    for n_fag, fr1query in enumerate(gc_param.hloss.f_audgram_list):
        hl0_pin_cochlea_db[n_fag] = utils.hl2pin_cochlea(fr1query, 0) # convert to cochlea input level
        compression_health = gc_param.hloss.compression_health[n_fag]
        _, hl0_io_func_db_ch1, _ \
            = gcfb_v23_asym_func_in_out(gc_param, gc_resp, fr1query, 1, hl0_pin_cochlea_db[n_fag])
        pin_db_act_reduction \
            = gcfb_v23_asym_func_in_out_inv_io_func(gc_param, gc_resp, fr1query, compression_health, hl0_io_func_db_ch1)

        pin_loss_db_act[n_fag] = pin_db_act_reduction - hl0_pin_cochlea_db[n_fag] # h_loss_db is positive
        pin_loss_db_act_init[n_fag] = pin_loss_db_act[n_fag] # initial value of act_loss
        pin_loss_db_pas[n_fag] = np.maximum(gc_param.hloss.hearing_level_db[n_fag] - pin_loss_db_act[n_fag], 0)

        # re-calculation of pin_loss_db_* if the values reached the under limits (8 Sep 22)
        if pin_loss_db_pas[n_fag] < eps * 10**4:
            pin_loss_db_act[n_fag] = gc_param.hloss.hearing_level_db[n_fag] - pin_loss_db_pas[n_fag]
            
            cmprs_hlth_list = np.arange(1, 0, -0.1)
            pin_loss_db_act4cmpnst = np.zeros((len(cmprs_hlth_list)))

            for n_ch, cmprs_hlth in enumerate(cmprs_hlth_list):
                pin_db_cmprs_hlth_val_inv \
                    = gcfb_v23_asym_func_in_out_inv_io_func(gc_param, gc_resp, fr1query, cmprs_hlth, hl0_io_func_db_ch1)
                pin_loss_db_act4cmpnst[n_ch] = pin_db_cmprs_hlth_val_inv - hl0_pin_cochlea_db[n_fag]
            
            func_interp1d = interp1d(pin_loss_db_act4cmpnst, cmprs_hlth_list, kind='linear', fill_value='extrapolate')
            compression_health = func_interp1d(pin_loss_db_act[n_fag])

            if np.any(np.isnan(compression_health)):
                raise ValueError('Error: compression_health recalculation')
                
            pin_db_act_reduction \
                = gcfb_v23_asym_func_in_out_inv_io_func(gc_param, gc_resp, fr1query, compression_health, hl0_io_func_db_ch1)
            pin_loss_db_act[n_fag] = pin_db_act_reduction - hl0_pin_cochlea_db[n_fag] # h_loss_db is positive
            pin_loss_db_pas[n_fag] = gc_param.hloss.hearing_level_db[n_fag] - pin_loss_db_act[n_fag] # error: +/- 0.3 dB, no problem

            if np.abs(gc_param.hloss.compression_health_initval[n_fag] - compression_health > eps):
                print(f'Compenstated GCparam.HLoss.CompressionHealth ({fr1query} Hz): \
                        {gc_param.hloss.compression_health_initval[n_fag]} \
                        --> {compression_health}')
    
        # for debug
        error_act_pas = gc_param.hloss.hearing_level_db[n_fag] \
                    - (pin_loss_db_pas[n_fag] + pin_loss_db_act[n_fag])
        if np.abs(error_act_pas) > eps * 10**2:
            print(f'{error_act_pas} {gc_param.hloss.hearing_level_db[n_fag]} \
                    {pin_loss_db_act[n_fag]} {pin_loss_db_pas[n_fag]}')
            if not gc_param.hloss.type in 'NH':
                raise ValueError('Error: HL_total = HL_ACT + HL_PAS')
        
        # update compression health foctor
        gc_param.hloss.compression_health[n_fag] = compression_health

        # overall gain control --- calculated from the max value of asym_function
        hl_val_pin_cochlea_db[n_fag] = utils.hl2pin_cochlea(fr1query, 0) + gc_param.hloss.hearing_level_db[n_fag] # convert to cochlea input level
        _, hl_val_io_func_db_ch_val, _ = gcfb_v23_asym_func_in_out(gc_param, gc_resp, fr1query, compression_health, hl_val_pin_cochlea_db[n_fag])
        gc_param.hloss.af_gain_cmpnst_db[n_fag] = hl_val_io_func_db_ch_val

    nh_gain_cmpnst_bias_db = np.array([0, 0, 0, 0, 0, 0, 0]) # no compensation (8 Oct. 2021)
    gc_param.hloss.af_gain_cmpnst_db = gc_param.hloss.af_gain_cmpnst_db + nh_gain_cmpnst_bias_db
    gc_param.hloss.hl_val_pin_cochlea_db = hl_val_pin_cochlea_db
    gc_param.hloss.pin_loss_db_act = pin_loss_db_act
    gc_param.hloss.pin_loss_db_pas = pin_loss_db_pas
    gc_param.hloss.pin_loss_db_act_init = pin_loss_db_act_init

    # interporation to gc_resp.fr1 (which is closer to fp2)
    erb_rate_fag, _ = utils.freq2erb(gc_param.hloss.f_audgram_list)
    erb_rate_fr1, _ = utils.freq2erb(gc_resp.fr1) # gc_channel
    gc_param.hloss.fb_fr1 = gc_resp.fr1
    gc_param.hloss.fb_hearing_level_db = utils.interp1(erb_rate_fag, gc_param.hloss.hearing_level_db, erb_rate_fr1, extrapolate=True)
    gc_param.hloss.fb_pin_cochlea_db = utils.interp1(erb_rate_fag, gc_param.hloss.hl_val_pin_cochlea_db, erb_rate_fr1, extrapolate=True)
    gc_param.hloss.fb_pin_loss_db_act = utils.interp1(erb_rate_fag, gc_param.hloss.pin_loss_db_act, erb_rate_fr1, extrapolate=True)
    gc_param.hloss.fb_pin_loss_db_pas = utils.interp1(erb_rate_fag, gc_param.hloss.pin_loss_db_pas, erb_rate_fr1, extrapolate=True)
    gc_param.hloss.io_func_loss_db_pas = io_func_loss_db_pas
    gc_param.hloss.fb_compression_health = \
        np.minimum(np.maximum(utils.interp1(erb_rate_fag, gc_param.hloss.compression_health, erb_rate_fr1, extrapolate=True), 0), 1)
    gc_param.hloss.fb_af_gain_cmpnst_db = utils.interp1(erb_rate_fag, gc_param.hloss.af_gain_cmpnst_db, erb_rate_fr1, extrapolate=True)

    # for debug
    # sw_plot = 1
    sw_plot = 0
    if sw_plot == 1:
        fig = plt.figure(figsize=[5, 5], tight_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(erb_rate_fr1, gc_param.hloss.fb_pin_loss_db_pas, '--', label='PAS_GainReduct')
        ax.plot(erb_rate_fr1, gc_param.hloss.fb_pin_loss_db_act, label='ACT_GainReduct')
        ax.set_xlabel('ERB_N number')
        ax.set_ylabel('Gain Reduction (dB)')
        ax.legend(loc='upper left')
        ax.set_title(f'Type: {gc_param.hloss.type}, Comp. Health: {gc_param.hloss.compression_health[0]}')
        plt.show()


    return gc_param



def make_asym_cmp_filters_v2(fs, frs, b, c):
    """Computes the coefficients for a bank of Asymmetric Compensation Filters
    This is a modified version to fix the round off problem at low freqs
    Use this with ACFilterBank.m
    See also asym_cmp_frsp_v2 for frequency response

    Args:
        fs (int): Sampling frequency
        frs (array_like): array of the center frequencies, frs
        b (array_like): array or scalar of a bandwidth coefficient, b
        c (float): array or scalar of asymmetric parameters, c

    Returns:
        acf_coef: 
            .fs (int): Sampling frequency
            .bz (array_like): MA coefficients  (num_ch*3*num_filt)
            .ap (array_like): AR coefficients  (num_ch*3*num_filt)

    Notes:
        [1] Ref for p1-p4: Unoki,M , Irino,T. , and Patterson, R.D. , "Improvement of an IIR asymmetric compensation gammachirp filter," Acost. Sci. & Tech. (ed. by the Acoustical Society of Japan ), 22 (6), pp. 426-430, Nov. 2001.
        [2] Conventional setting was removed.
            fn = frs + nfilt* p3 .*c .*b .*erbw/n;
            This frequency fn is for normalizing GC(=GT*Hacf) filter to be unity at the peak, frequnecy. But now we use Hacf as a highpass filter as well. cGC = pGC *Hacf. In this case, this normalization is useless. 
            So, it was set as the gain at frs is unity.  (4. Jun 2004 )
        [3] Removed
            acf_coef.fn(:,nff) = fn;
            n : scalar of order t^(n-1) % used only in normalization 
    """
    num_ch, len_frs = np.shape(frs)
    if len_frs > 1:
        print("frs should be a column vector frs.", file=sys.stderr)
        sys.exit(1)
    
    _, erbw = utils.freq2erb(frs)

    acf_coef = ACFcoef()
    acf_coef.fs = fs

    # New coefficients. See [1]
    num_filt = 4
    p0 = 2
    p1 = 1.7818 * (1-0.0791*b) * (1-0.1655*np.abs(c))
    p2 = 0.5689 * (1-0.1620*b) * (1-0.0857*np.abs(c))
    p3 = 0.2523 * (1-0.0244*b) * (1+0.0574*np.abs(c))
    p4 = 1.0724

    if num_filt > 4:
        print("num_filt > 4", file=sys.stderr)
        sys.exit(1) 

    acf_coef.ap = np.zeros((num_ch, 3, num_filt))
    acf_coef.bz = np.zeros((num_ch, 3, num_filt))

    for nfilt in range(num_filt):
        r  = np.exp(-p1*(p0/p4)**(nfilt) * 2*np.pi*b*erbw / fs)
        del_frs = (p0*p4)**(nfilt)*p2*c*b*erbw;  
        phi = 2*np.pi*(frs+del_frs).clip(0)/fs
        psi = 2*np.pi*(frs-del_frs).clip(0)/fs
        fn = frs # see [2]

        # second order filter
        ap = np.concatenate([np.ones(np.shape(r)), -2*r*np.cos(phi), r**2], axis=1)
        bz = np.concatenate([np.ones(np.shape(r)), -2*r*np.cos(psi), r**2], axis=1)

        vwr = np.exp(1j*2*np.pi*fn/fs)
        vwrs = np.concatenate([np.ones(np.shape(vwr)), vwr, vwr**2], axis=1)
        nrm = np.array([np.abs(np.sum(vwrs*ap, axis=1) / np.sum(vwrs*bz, axis=1))]).T
        bz = bz * (nrm*np.ones((1, 3)))

        acf_coef.ap[:,:,nfilt] = ap
        acf_coef.bz[:,:,nfilt] = bz

    return acf_coef


def fr1_to_fp2(n, b1, c1, b2, c2, frat, fr1, sr=24000, n_fft=2048, sw_plot=0):
    """Convert fr1 (for passive GC; pGC) to fp2 (for compressive GC; cGC)

    Args:
        n (int): Parameter defining the envelope of the gamma distribution (for pGC)
        b1 (float): Parameter defining the envelope of the gamma distribution (for pGC)
        c1 (float): Chirp factor (for pGC)
        b2 (float): Parameter defining the envelope of the gamma distribution (for cGC)
        c2 (float): Chirp factor  (for cGC)
        frat (float): Frequency ratio, the main level-dependent variable
        fr1 (float): Center frequency (for pGC)
        sr (int, optional): Sampling rate. Defaults to 24000.
        n_fft (int, optional): Size of FFT. Defaults to 2048.
        sw_plot (int, optional): Show plot of cgc_frsp and pGC_frsp. Defaults to 0.

    Returns:
        fp2 (float): Peak frequency (for compressive GC)
        fr2 (float): Center Frequency (for compressive GC)
    """
    _, erbw1 = utils.freq2erb(fr1)
    fp1, _ = gc.fr2fpeak(n, b1, c1, fr1)
    fr2 = frat * fp1
    _, erbw2 = utils.freq2erb(fr2)

    bw1 = b1 * erbw1
    bw2 = b2 * erbw2

    # coef1*fp2^3 + coef2*fp2^2 + coef3*fp2 + coef4 = 0 
    coef1 = -n
    coef2 = c1*bw1 + c2*bw2 + n*fr1 + 2*n*fr2
    coef3 = -2*fr2*(c1*bw1+n*fr1) - n*((bw2)**2+fr2**2) - 2*c2*bw2*fr1
    coef4 = c2*bw2*((bw1)**2+fr1**2) + (c1*bw1+n*fr1)*(bw2**2+fr2**2)
    coefs = np.array([coef1, coef2[0], coef3[0], coef4[0]])

    p = np.roots(coefs)
    fp2cand = p[np.imag(p)==0]
    if len(fp2cand) == 1:
        fp2 = fp2cand
    else:
        val, ncl = np.min(np.abs(fp2cand - fp1))
        fp2 = fp2cand[ncl] # in usual cGC range, fp2 is close to fp1

    # sw_plot = 1
    if sw_plot == 1: # Check
        fs = 48000
        n_frq_rsl = 2048
        cgc_rsp = cmprs_gc_frsp(fr1, fs, n, b1, c1, frat, b2, c2, n_frq_rsl)

        nfr2 = np.zeros((len(fp2cand), 1))
        for nn in range(len(fp2cand)):
            nfr2[nn] = np.argmin(abs(cgc_rsp.freq - fp2cand[nn]))
        
        fig, ax = plt.subplots()
        plt_freq = np.array(cgc_rsp.freq).T
        plt_cgc_frsp = np.array(cgc_rsp.cgc_frsp/np.max(cgc_rsp.cgc_frsp)).T
        plt_pgc_frsp = np.array(cgc_rsp.pgc_frsp).T

        ax.plot(plt_freq, plt_cgc_frsp, label="cgc_frsp") # compressive GC
        ax.plot(plt_freq, plt_pgc_frsp, label="pgc_frsp") # passive GC
        ax.set_xlim([0, np.max(fp2cand)*2])
        ax.set_ylim([0, 1])
        ax.legend()
        plt.show()

    return fp2, fr2

def fp2_to_fr1(n, b1, c1, b2, c2, frat, fp2):
    """Convert fp2 (for compressive GC; cGC) to fr1 (for passive GC; pGC)

    Args:
        n (int): Parameter defining the envelope of the gamma distribution (for pGC)
        b1 (float): Parameter defining the envelope of the gamma distribution (for pGC)
        c1 (float): Chirp factor (for pGC)
        b2 (float): Parameter defining the envelope of the gamma distribution (for cGC)
        c2 (float): Chirp factor  (for cGC)
        frat (float): Frequency ratio, the main level-dependent variable
        fr2 (float): Center Frequency (for compressive GC)

    Returns:
        fr1 (float): Center frequency (for pGC)
        fp1 (float): Peak frequency (for pGC)
    """    
    # Coefficients: ERBw(fr1) = alp1*fr1+alp0
    _, alp0 = utils.freq2erb(0)
    _, w1 = utils.freq2erb(1)
    alp1 = w1 - alp0

    # Coefficients: fr2=bet1*fr2+bet0
    bet1 = frat*(1+c1*b1*alp1/n)
    bet0 = frat*c1*b1*alp0/n

    # Coefficients: ERB(fr2)=zet1*Fr1+zet0
    zet1=alp1*bet1
    zet0=alp1*bet0+alp0

    # Coef1*Fr1**3 + Coef2*Fr1**2 + Coef3*Fr1 + Coef4 = 0
    coef1 = ((b2**2*zet1**2+bet1**2)*(c1*b1*alp1+n) + (c2*b2*zet1)*(b1**2*alp1**2+1))
    coef2 = ((b2**2*zet1**2+bet1**2)*(c1*b1*alp0-n*fp2) \
            + (2*b2**2*zet1*zet0-2*bet1*(fp2-bet0))*(c1*b1*alp1+n) \
            + (c2*b2*zet1)*(2*b1**2*alp1*alp0-2*fp2) + (c2*b2*zet0)*(b1**2*alp1**2+1))
    coef3 = ((2*b2**2*zet1*zet0-2*bet1*(fp2-bet0))*(c1*b1*alp0-n*fp2) \
            + (b2**2*zet0**2+(fp2-bet0)**2)*(c1*b1*alp1+n) \
            + (c2*b2*zet1)*(b1**2*alp0**2+fp2**2) \
            + (c2*b2*zet0)*(2*b1**2*alp1*alp0-2*fp2))
    coef4 = (b2**2*zet0**2+(fp2-bet0)**2)*(c1*b1*alp0-n*fp2) \
            + (c2*b2*zet0)*(b1**2*alp0**2+fp2**2)
    coefs = [coef1, coef2, coef3, coef4]

    q = np.roots(coefs)
    fr1cand = q[np.imag(q)==0]
    if len(fr1cand) == 1:
        fr1 = fr1cand
        fp1, _ = gc.fr2fpeak(n, b1, c1, fr1)
    else:
        fp1cand, _ = gc.fr2fpeak(n, b1, c1, fr1cand)
        ncl = np.argmin(np.abs(fp1cand - fp2)) 
        fp1 = fp1cand[ncl]
        fr1 = fr1cand[ncl]

    fr1 = fr1.real.astype(float)     

    return fr1, fp1


def cmprs_gc_frsp(fr1, fs=48000, n=4, b1=1.81, c1=-2.96, frat=1, b2=2.01, c2=2.20, n_frq_rsl=1024):
    """Frequency Response of Compressive GammaChirp

    Args:
        fr1 (array-like): Resonance Freqs.
        fs (int, optional): Sampling Freq. Defaults to 48000.
        n (int, optional): Order of Gamma function, t**(n-1). Defaults to 4.
        b1 (float, optional): b1 for exp(-2*pi*b1*erb(f)). Defaults to 1.81.
        c1 (float, optional): c1 for exp(j*2*pi*fr + c1*ln(t)). Defaults to -2.96.
        frat (int, optional): Frequency ratio. fr2 = frat*fp1. Defaults to 1.
        b2 (float, optional): _description_. Defaults to 2.01.
        c2 (float, optional): _description_. Defaults to 2.20.
        n_frq_rsl (int, optional): _description_. Defaults to 1024.

    Returns:
        cgc_resp: Struct of cGC response
            .pgc_frsp (array-like): Passive GC freq. resp. (num_ch*n_frq_rsl matrix)
            .cgc_frsp (array-like): Comressive GC freq. resp. (num_ch*n_frq_rsl matrix)
            .cgc_nrm_frsp (array-like): Normalized cgc_frsp (num_ch*n_frq_rsl matrix)
            .acf_frsp: Asym (array-like). Compensation Filter freq. resp.
            .asym_func (array-like): Asym Func
            .freq (array-like): Frequency (1*n_frq_rsl)
            .fp2 (array-like): Peak freq.
            .val_fp2 (array-like): Peak Value
    """
    if utils.isrow(fr1):
        fr1 = np.array([fr1]).T

    num_ch = len(fr1)

    if isinstance(n, (int, float)):
        n = n * np.ones((num_ch, 1))
    if isinstance(b1, (int, float)):
        b1 = b1 * np.ones((num_ch, 1))
    if isinstance(c1, (int, float)):
        c1 = c1 * np.ones((num_ch, 1))
    if isinstance(frat, (int, float)):
        frat = frat * np.ones((num_ch, 1))
    if isinstance(b2, (int, float)):
        b2 = b2 * np.ones((num_ch, 1))
    if isinstance(c2, (int, float)):
        c2 = c2 * np.ones((num_ch, 1))

    pgc_frsp, freq, _, _, _ = gc.gammachirp_frsp(fr1, fs, n, b1, c1, 0.0, n_frq_rsl)
    fp1, _ = gc.fr2fpeak(n, b1, c1, fr1)
    fr2 = frat * fp1
    acf_frsp, freq, asym_func = asym_cmp_frsp_v2(fr2, fs, b2, c2, n_frq_rsl)
    cgc_frsp = pgc_frsp * asym_func # cgc_frsp = pgc_frsp * acf_frsp
    
    val_fp2 = np.max(cgc_frsp, axis=1)
    nchfp2 = np.argmax(cgc_frsp, axis=1)
    if utils.isrow(val_fp2):
        val_fp2 = np.array([val_fp2]).T
    
    norm_fact_fp2 = 1/val_fp2

    # function cGCresp = CmprsGCFrsp(fr1,fs,n,b1,c1,frat,b2,c2,n_frq_rsl)
    cgc_resp = cGCresp()
    cgc_resp.fr1 = fr1
    cgc_resp.n = n
    cgc_resp.b1 = b1
    cgc_resp.c1 = c1
    cgc_resp.frat = frat
    cgc_resp.b2 = b2
    cgc_resp.c2 = c2
    cgc_resp.n_frq_rsl = n_frq_rsl
    cgc_resp.pgc_frsp = pgc_frsp
    cgc_resp.cgc_frsp = cgc_frsp
    cgc_resp.cgc_nrm_frsp = cgc_frsp * (norm_fact_fp2 * np.ones((1,n_frq_rsl)))
    cgc_resp.acf_frsp = acf_frsp
    cgc_resp.asym_func = asym_func
    cgc_resp.fp1 = fp1
    cgc_resp.fr2 = fr2
    cgc_resp.fp2 = freq[nchfp2]
    cgc_resp.val_fp2 = val_fp2
    cgc_resp.norm_fct_fp2 = norm_fact_fp2
    cgc_resp.freq = [freq]

    return cgc_resp
    

def asym_cmp_frsp_v2(frs, fs=48000, b=None, c=None, n_frq_rsl=1024, num_filt=4):
    """Amplitude spectrum of Asymmetric compensation IIR filter (ACF) for the gammachirp 
    corresponding to make_asym_cmp_filters_v2

    Args:
        frs (array_like, optional): Center freqs. Defaults to None.
        fs (int, optional): Sampling freq. Defaults to 48000.
        b (array_like, optional): Bandwidth coefficient. Defaults to None.
        c (array_like, optional): Asymmetric paramters. Defaults to None.
        n_frq_rsl (int, optional): Freq. resolution for linear freq. scale for specify renponse at frs
                                (n_frq_rsl>64). Defaults to 1024.
        num_filt (int, optional): Number of 2nd-order filters. Defaults to 4.

    Returns:
        acf_frsp: Absolute values of frequency response of ACF (num_ch * n_frq_rsl)
        freq: freq. (1 * n_frq_rsl)
        asym_func: Original asymmetric function (num_ch * n_frq_rsl)
    """
    if utils.isrow(frs):
        frs = np.array([frs]).T
    if utils.isrow(b):
        b = np.array([b]).T
    if utils.isrow(c):
        c = np.array([c]).T
    num_ch = len(frs)

    if n_frq_rsl >= 64:
        freq = np.arange(n_frq_rsl) / n_frq_rsl * fs/2
    elif n_frq_rsl == 0:
        freq = frs
        n_frq_rsl = len(freq)
    else:
        help(asym_cmp_frsp_v2)
        print("Specify n_frq_rsl 0) for frs or N>=64 for linear-freq scale", file=sys.stderr)
        sys.exit(1)

    # coef.
    sw_coef = 0 # self consistency
    # sw_coef = 1 # reference to make_asym_cmp_filters_v2

    if sw_coef == 0:
        # New Coefficients. num_filt = 4; See [1]
        p0 = 2
        p1 = 1.7818 * (1 - 0.0791*b) * (1 - 0.1655*np.abs(c))
        p2 = 0.5689 * (1 - 0.1620*b) * (1 - 0.0857*np.abs(c))
        p3 = 0.2523 * (1 - 0.0244*b) * (1 + 0.0574*np.abs(c))
        p4 = 1.0724
    else:
        acf_coef = make_asym_cmp_filters_v2(fs, frs, b, c)

    # filter coef.
    _, erbw = utils.freq2erb(frs)
    acf_frsp = np.ones((num_ch, n_frq_rsl))
    freq2 = np.concatenate([np.ones((num_ch,1))*freq, frs], axis=1)

    for nfilt in range(num_filt):

        if sw_coef == 0:
            r = np.exp(-p1 * (p0/p4)**nfilt * 2 * np.pi * b * erbw / fs)
            delfr = (p0*p4)**nfilt * p2 * c * b * erbw
            phi = 2*np.pi*np.maximum(frs + delfr, 0)/fs
            psi = 2*np.pi*np.maximum(frs - delfr, 0)/fs
            fn = frs
            ap = np.concatenate([np.ones((num_ch, 1)), -2*r*np.cos(phi), r**2], axis=1)
            bz = np.concatenate([np.ones((num_ch, 1)), -2*r*np.cos(psi), r**2], axis=1)
        else:
            ap = acf_coef.ap[:, :, nfilt]
            bz = acf_coef.bz[:, :, nfilt]

        cs1 = np.cos(2*np.pi*freq2/fs)
        cs2 = np.cos(4*np.pi*freq2/fs)
        bzz0 = np.array([bz[:, 0]**2 + bz[:, 1]**2 + bz[:, 2]**2]).T * np.ones((1, n_frq_rsl+1))
        bzz1 = np.array([2 * bz[:, 1] * (bz[:, 0] + bz[:, 2])]).T * np.ones((1, n_frq_rsl+1))
        bzz2 = np.array([2 * bz[:, 0] * bz[:, 2]]).T * np.ones((1, n_frq_rsl+1))
        hb = bzz0 + bzz1*cs1 + bzz2*cs2

        app0 = np.array([ap[:, 0]**2 + ap[:, 1]**2 + ap[:, 2]**2]).T * np.ones((1, n_frq_rsl+1))
        app1 = np.array([2 * ap[:, 1] * (ap[:, 0] + ap[:, 2])]).T * np.ones((1, n_frq_rsl+1))
        app2 = np.array([2 * ap[:, 0] * ap[:, 2]]).T * np.ones((1, n_frq_rsl+1))
        ha = app0 + app1*cs1 + app2*cs2

        h = np.sqrt(hb/ha)
        h_norm = np.array([h[:, n_frq_rsl]]).T * np.ones((1, n_frq_rsl)) # Normalizatoin by fn value

        acf_frsp = acf_frsp * h[:,0:n_frq_rsl] / h_norm

    # original Asymmetric Function without shift centering
    fd = np.ones((num_ch, 1))*freq - frs*np.ones((1, n_frq_rsl))
    be = (b * erbw) * np.ones((1, n_frq_rsl))
    cc = (c * np.ones((num_ch, 1)) * np.ones((1, n_frq_rsl))) # in case when c is scalar
    asym_func = np.exp(cc * np.arctan2(fd, be))

    return acf_frsp, freq, asym_func


def acfilterbank(acf_coef, acf_status, sig_in=[], sw_ordr=0):
    """IIR ACF time-slice filtering for time-varing filter

    Args:
        acf_coef (structure): acf_coef: coef from make_asym_cmp_filters_v2
            .ap: AR coefficents (==a ~= pole) num_ch*lap*num_filt
            .fs : sampling rate  (also switch for verbose)
                (The variables named 'a' and 'b' are not used to avoid the
                confusion to the gammachirp parameters.)
            .verbose : Not specified) quiet   1) verbose
        acf_status (structure):
            .num_ch: Number of channels (Set by initialization
            .lbz: size of MA
            .lap: size of AR
            .num_filt: Length of filters
            .sig_in_prev: Previous status of sig_in
            .sig_out_prev: Previous status of SigOut
        sig_in (array_like, optional): Input signal. Defaults to [].
        sw_ordr (int, optional): Switch filtering order. Defaults to 0.

    Returns:
        SigOut (array_like): Filtered signal (num_ch * 1)
        acf_status: Current status
    """    
    if len(sig_in) == 0 and len(acf_status) != 0:
        help(acfilterbank)
        sys.exit()

    if not hasattr(acf_status, 'num_ch'):
        acf_status = ACFstatus()

        num_ch, lbz, num_filt = np.shape(acf_coef.bz)
        num_ch, lap, _ = np.shape(acf_coef.ap)

        if lbz != 3 or lap !=3:
            print("No gaurantee for usual IIR filters except for AsymCmpFilter.\n"\
                + "Please check make_asym_cmp_filters_v2.")
    
        acf_status.num_ch = num_ch
        acf_status.num_filt = num_filt
        acf_status.lbz = lbz # size of MA
        acf_status.lap = lap # size of AR
        acf_status.sig_in_prev = np.zeros((num_ch, lbz))
        acf_status.sig_out_prev = np.zeros((num_ch, lap, num_filt))
        acf_status.count = 0
        print("ACFilterBank: Initialization of acf_status")
        sig_out = []

        return sig_out, acf_status
    
    if utils.isrow(sig_in):
        sig_in = np.array([sig_in]).T
    
    num_ch_sig, len_sig = np.shape(sig_in)
    if len_sig != 1:
        print("Input signal sould be num_ch*1 vector (1 sample time-slice)", file=sys.stderr)
        sys.exit(1)
    if num_ch_sig != acf_status.num_ch:
        print(f"num_ch_sig ({num_ch_sig}) != acf_status.num_ch ({acf_status.num_ch})")

    # time stamp
    if hasattr(acf_coef, 'verbose'):
        if acf_coef.verbose == 1: # verbose when acf_coef.verbose is specified to 1
            t_disp = 50 # ms
            t_cnt = acf_status.count/(np.fix(acf_coef.fs/1000)) # ms

            if acf_status.count == 0:
                print("ACFilterBank: Start processing")
                tic = time.time()

            elif np.mod(t_cnt, t_disp) == 0:
                toc = time.time()
                print(f"ACFilterBank: Processed {t_cnt} (ms)." \
                      + f"elapsed Time = {np.round(tic-toc, 1)} (sec)")
    
    acf_status.count = acf_status.count+1
    
    """
    Processing
    """
    acf_status.sig_in_prev = np.concatenate([acf_status.sig_in_prev[:, 1:acf_status.lbz], sig_in], axis=1)

    x = acf_status.sig_in_prev.copy()
    nfilt_list = np.arange(acf_status.num_filt)

    if sw_ordr == 1:
        nfilt_list = np.flip(nfilt_list)

    for nfilt in nfilt_list:

        forward = acf_coef.bz[:, acf_status.lbz::-1, nfilt] * x
        feedback = acf_coef.ap[:, acf_status.lap:0:-1, nfilt] * \
            acf_status.sig_out_prev[:, 1:acf_status.lap, nfilt]

        fwdSum = np.sum(forward, axis=1)
        fbkSum = np.sum(feedback, axis=1)

        y = np.array([(fwdSum - fbkSum) / acf_coef.ap[:, 0, nfilt]]).T
        acf_status.sig_out_prev[:, :, nfilt] = \
            np.concatenate([acf_status.sig_out_prev[:, 1:acf_status.lap, nfilt], y], axis=1)
        x = acf_status.sig_out_prev[:, :, nfilt].copy()

    sig_out = y

    return sig_out, acf_status


def cal_smooth_spec(fb_out, fb_param):
    """Caluculation of smoothed spectrogram from GCFB

    Args:
        fb_out (array_like): outputs of GCFB
        fb_param (struct): parameters of GCFB

    Returns:
        smooth_spec: smoothed spectrogram
        fb_param: parameters of GCFB
    """
    fs = fb_param.fs

    if not hasattr(fb_param, 'method'):
        fb_param.method = 1 # default setting
    
    # secction of method
    if fb_param.method == 1: # default setting
        fb_param.t_shift = 0.005 # 5 ms from HTK MFCC
        fb_param.n_shift = fb_param.t_shift * fs
        fb_param.t_win = 0.025 # 25 ms from HTK MFCC
        fb_param.n_win = fb_param.t_win * fs
        fb_param.type_win = 'hamming' # hamming window from HTK MFCC
        fb_param.win = np.hamming(fb_param.n_win)
    elif fb_param.method == 2:
        fb_param.t_shift = 0.005 # 5 ms from HTK MFCC
        fb_param.n_shift = fb_param.t_shift * fs
        fb_param.t_win = 0.010 # 10 ms from HTK MFCC
        fb_param.n_win = fb_param.t_win * fs
        fb_param.type_win = 'hanning' # hamming window from HTK MFCC
        # np.hanning: does not match to Matlab 'hannig()'
        # https://stackoverflow.com/questions/56485663/hanning-window-values-doesnt-match-in-python-and-matlab
        fb_param.val_win = np.hanning(fb_param.n_win+2)[1:-1]
    else:
        print("Specify FBparam.Method : 1 or 2", file=sys.stderr)
        sys.exit(1)

    print(f"fb_param.Win = {fb_param.type_win} (fb_param.n_win)")
    fb_param.win = fb_param.win / np.sum(fb_param.win) # normalized
    fb_param.type_smooth \
        = f"Temporal smoothing with a {fb_param.type_win} window"
    
    # calculation
    num_ch, len_snd = np.shape(fb_out)
    for nch in range(num_ch):
        val_frame, _ = \
            set_frame4time_sequence(fb_out[nch, :], fb_param.n_win, fb_param.n_shift)
        if nch == 0:
            len_frame = np.size(val_frame, 1)
            smooth_spec = np.zeros([num_ch, len_frame])
        val_frame_win = np.dot(fb_param.win, val_frame)
        smooth_spec[nch, :] = val_frame_win

    fb_param.tempral_positions = np.arange(len_frame-1) * fb_param.t_shift
    
    return smooth_spec, fb_param


def set_frame4time_sequence(snd, len_win, len_shift=[]):
    """Set frame for Time sequence signal used for Spectral feature extraction etc.

    Args:
        snd (array_like): Sound data
        len_win (int): Frame length in sample
        len_shift (float, optional): Frame shift in sample (== len_win/integer_value). 
            Defaults to 0.5.

    Returns:
        snd_frame (array_like): Frame matrix
        num_smpl_pnt (array_like): Number of sample point which is center of each frame
    """
    if len_shift == []:
        len_shift = len_win / 2
    
    int_div_frame = int(len_win / len_shift)

    if not np.mod(int_div_frame, 1) == 0 or not np.mod(len_win, 2) == 0:
        print(f"len_win = {len_win}, \n" \
            + f"len_shift = {len_shift}, \n" \
            + f"ratio = {int_div_frame} \n" \
            + " <-- should be integer value")
        print("len_win must be even number")
        print("len_shift must be len_win/integer value", file=sys.stderr)
        sys.exit(1)
    
    snd1 = np.array(list(np.zeros([int(len_win/2)])) + list(snd) \
                     + list(np.zeros([int(len_win/2)]))) # zero-padding
    len_snd1 = len(snd1)
    num_frame1 = np.ceil(len_snd1/len_win)
    n_lim = int(len_win * num_frame1)
    snd1 = np.array(list(snd1[0:min(n_lim, len_snd1)]) + list(np.zeros(n_lim - len_snd1)))
    len_snd1 = len(snd1)

    num_frame_all = (num_frame1-1) * int_div_frame + 1
    snd_frame = np.zeros([int(len_win), int(num_frame_all)])
    num_smpl_pnt = np.zeros(int(num_frame_all))

    for nid in range(int_div_frame):
        num_frame2 = int(num_frame1 - (nid > 0))
        n_snd = len_shift*nid + np.arange(num_frame2*len_win)
        snd2 = snd1[n_snd.astype(int)]
        mtrx = snd2.reshape(int(len_win), num_frame2, order='F').copy()
        num = np.arange(nid, num_frame_all, int_div_frame).astype(int)
        n_indx = num * len_shift # center of frame
        snd_frame[:, num] = mtrx
        num_smpl_pnt[num] = n_indx
    
    n_valid_num_smpl_pnt = np.argwhere(num_smpl_pnt <= len(snd))[:, 0]
    snd_frame = snd_frame[:, n_valid_num_smpl_pnt]
    num_smpl_pnt = num_smpl_pnt[n_valid_num_smpl_pnt]
    
    return snd_frame, num_smpl_pnt


def gcfb_v23_asym_func_in_out(gc_param, gc_resp, fr1query, compression_health, pin_db):
    """ Calculate GC Hearing Loss from GCFBv23*

    Args:
        gc_param (struct): parameters for gcfb
        gc_resp (struct): parameters for gcfb
        fr1query (array_like): Specify by Fr1  which is usually used in specifying FB freq. (not Fp1)
        compression_health (array_like): compression health factor for HP-AF
        pin_db (float): cochlea input level

    Returns:
        af_out_lin (float): c2 paramter applied with compression health factor
    """    
    gc_param.asym_func_norm_db = 100 # default

    af_out_lin = cal_asym_func(gc_param, gc_resp, fr1query, compression_health, pin_db)
    af_out_lin_norm = cal_asym_func(gc_param, gc_resp, fr1query, compression_health, gc_param.asym_func_norm_db)

    af_out_db = 20*np.log10(af_out_lin/af_out_lin_norm)
    io_func_db = af_out_db + pin_db

    return af_out_db, io_func_db, gc_param


def cal_asym_func(gc_param, gc_resp, fr1query, compression_health, pin_db):
    """ Calculate AsymFunc with GC Hearing Loss from GCFBv23*

    Args:
        gc_param (struct): parameters for gcfb
        gc_resp (struct): parameters for gcfb
        fr1query (array_like): Specify by Fr1  which is usually used in specifying FB freq. (not Fp1)
        compression_health (array_like): compression health factor for HP-AF
        pin_db (float): cochlea input level

    Returns:
        af_out_lin (float): c2 paramter applied with compression health factor
    """    
    n_ch = np.argmin(np.abs(gc_param.fr1 - fr1query)) # choosing the closest number n_ch of fr1 in the GCFB
    fp1 = gc_resp.fp1[n_ch]
    frat = gc_resp.frat0_pc[n_ch] + gc_resp.frat1_val[n_ch] * (pin_db - gc_resp.pc_hpaf[n_ch]) # changes of coefficients from center
    fr2 = frat * fp1

    # To ignore RuntimeWarning in np.log10
    if np.any(4.37*fr2/1000+1 <= 0):
        np.seterr(invalid='ignore')
        _, erbw2 = utils.freq2erb(fr2) # difinition of HP-AF
        np.seterr(invalid='raise')
    else:
        _, erbw2 = utils.freq2erb(fr2) # difinition of HP-AF

    b2e = gc_resp.b2_val[n_ch] * erbw2
    c2ch = compression_health * gc_resp.c2_val[n_ch]

    af_out_lin = np.exp(c2ch * np.arctan2(fp1 - fr2, b2e)) # apply compression_health to c2

    return af_out_lin


def gcfb_v23_asym_func_in_out_inv_io_func(gc_param, gc_resp, fr1query, compression_health, io_func_db):
    """Calculate GC Hearing Loss from GCFBv23*  -- Inverse IOfunc

    Args:
        gc_param (struct): parameters for gcfb_v23*
        gc_resp (struct): parameters for gcfb_v23*
        fr1query (_type_): Specify by Fr1  which is usually used in specifying FB freq. (not Fp1)
        compression_health (_type_): compression health factor for HP-AF
        io_func_db (_type_): input-out function

    Returns:
        pin_db: cochlea input levels
    """    
    pin_db_list = np.arange(-120, 150+0.1, 0.1) # It is necessary to use such a wide range. (sometime it exceeds 120)
    _, io_func_db_list, _ = gcfb_v23_asym_func_in_out(gc_param, gc_resp, fr1query, compression_health, pin_db_list)
    func_interp1d = interp1d(io_func_db_list, pin_db_list, kind='linear', fill_value='extrapolate')
    pin_db = func_interp1d(io_func_db)

    return pin_db


def gcfb_v23_synth_snd(gc_smpl, gc_param):
    """ Synthesis sound for GCFBv23x

    Args:
        gc_smpl (struct): sample-based output from gcfb
        gc_param (_type_): parameter for gcfb

    Returns:
        snd_syn (array-like): synthesized sound as a monaural channel
    """    
    print('*** Synthesis from GCFB 2D-sample ***')
    fs = gc_param.fs

    # inverse compensation of ELC
    if not 'NO' in gc_param.out_mid_crct:
        # inverse filter (ELC etc...)
        amp_syn = -15
        t_delay = 0.00632 # time delay for ELC filter
        n_delay = int(np.fix(t_delay*fs))

        # backward inverse filter (26 Oct. 2021)
        inv_cmpn_out_mid = utils.mk_filter_field2cochlea(gc_param.out_mid_crct, fs, -1)

        snd_mean = np.mean(gc_smpl, axis=0)
        snd_syn1 = signal.lfilter(inv_cmpn_out_mid, 1, snd_mean)

        # amplitude and time compensation
        snd_syn1_time = list(snd_syn1[n_delay:]) + list(np.zeros((1, n_delay)))
        snd_syn = amp_syn * np.array(snd_syn1_time)

    else:
        # no inverse filters
        print('No inver out_mid_crct (FF/DF/ITU+MidEar/ELC) correction')
        amp_syn = -15
        snd_syn = amp_syn * np.mean(gc_smpl, axis=0)

    return snd_syn


def gcfb_v23_env_mod_loss(cgc_frame, gc_param, em_param):
    """ Reduction of Envelope Modulation working with gcfb_v23*

    Args:
        cgc_frame (array-like): output of the frame-based dcGC-FB
        gc_param (struct): parameters for the dcGC-FB
        em_param (struct): parameters for envelope modulations (em)

    Returns:
        em_frame (array-like): output reduced by em_param
        em_param (struct): updated parameters for em
    """    
    if not 'frame' in gc_param.dyn_hpaf.str_prc:
        raise ValueError('Working only when gc_param.dyn_hpaf.str_prc == "frame-base"')
    
    """
    parameter setting
    """
    len_fag = len(gc_param.hloss.f_audgram_list)
    em_param.fs = gc_param.dyn_hpaf.fs # sampling rate for the frame-based dcGC-FB

    if not hasattr(em_param, 'reduce_db'):
        em_param.reduce_db = np.zeros(len_fag)

    if isinstance(em_param.reduce_db, int):
        em_param.reduce_db = em_param.reduce_db * np.ones(len_fag)
    elif not len(em_param.reduce_db) == len_fag:
        raise ValueError('Set em_param.reduce_db at f_audgram_list in advance.')
    
    if hasattr(em_param, 'f_cutoff'):
        em_param.f_cutoff = em_param.f_cutoff * np.ones(len_fag)
    elif not len(em_param.f_cutoff) == len_fag:
        raise ValueError('Set em_param.f_cutoff at f_audgram_list in advance.')
    
    """
    filterbank information
    """
    erb_rate_fag, _ = utils.freq2erb(gc_param.hloss.f_audgram_list)
    erb_rate_fr1, _ = utils.freq2erb(gc_param.fr1) # for GC channel
    em_param.fb_fr1 = gc_param.fr1
    em_param.fb_reduce_db = utils.interp1(erb_rate_fag, em_param.reduce_db, erb_rate_fr1, 
                                          method='linear', extrapolate=True)
    em_param.fb_f_cutoff = utils.interp1(erb_rate_fag, em_param.f_cutoff, erb_rate_fr1, 
                                         method='linear', extrapolate=True)
    """
    main: filtering
    """
    em_frame = np.zeros(cgc_frame.shape)
    em_param.order_lpf = 1 # TMTF is a first-order low-pass filter
    em_param.sample_delay = 1 # for 1st order LPF

    em_param.fc_sep_filt = 1 # DC vs high freq: separation filter
    em_param.order_sep_filt = 2
    norm_sep_filt_cutoff = em_param.fc_sep_filt / (em_param.fs/2)
    bz_sep_lp, ap_sep_lp = signal.butter(em_param.order_sep_filt, norm_sep_filt_cutoff, btype='low')
    bz_sep_hp, ap_sep_hp = signal.butter(em_param.order_sep_filt, norm_sep_filt_cutoff, btype='high')

    sw_method = 1 # RMS: separation of DC component only
    # sw_method = 2 # Lowpass-highpass separation (NOT very good)

    for nch in range(gc_param.num_ch):
        env = cgc_frame[nch, :]
        if sw_method == 1:
            env_sep_lp = np.sqrt(np.mean(env**2)) # DC component
            env_sep_hp = env - env_sep_lp
        else:
            env_sep_lp = signal.lfilter(bz_sep_lp, ap_sep_lp, env) # env separated by LPF: NO gain control
            env_sep_hp = signal.lfilter(bz_sep_hp, ap_sep_hp, env) # env separated by HPF: Gain & LPF are applied.
        
        # Lowpass of env separated by HPF
        norm_f_cutoff = em_param.fb_f_cutoff[nch] / (em_param.fs/2)
        bz, ap = signal.butter(em_param.order_lpf, norm_f_cutoff)
        env_sep_hp2 = signal.lfilter(bz, ap, env_sep_hp)
        env_sep_hp2 = 10**(-em_param.fb_reduce_db[nch]/20) * env_sep_hp2 # reduce filter gain
        
        env_rdct = env_sep_hp2 + env_sep_lp

        # compensation of filter delay
        num_cmpnst = em_param.sample_delay
        em_frame[nch, :] = np.array([list(env_rdct[num_cmpnst::])+ list(np.zeros(num_cmpnst))])
    
    return em_frame, em_param


def gcfb_v23_ana_env_mod(cgc_frame, gc_param, em_param, sw_plot=False):
    """ Analysis of Envelope Modulation using filterbank working with gcfb_v23*

    Args:
        cgc_frame (array-like): output of the frame-based dcGC-FB
        gc_param (stryct): parameter of dcGC-fB
        em_param (struct): parameter of envelope modulation processing. 

    Returns:
        gcem_frame (array-like): output of modulation filterbank (num_ch, num_mod_ch, len_frame)
    """
    if not 'frame' in gc_param.dyn_hpaf.str_prc:
        raise ValueError('Working only when gc_param.dyn_hpaf.str_prc == "frame-base"')

    if not hasattr(em_param, 'fc_mod_list'):
        em_param.fc_mod_list = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    
    if not hasattr(em_param, 'fs'):
        em_param.fs = gc_param.dyn

    num_ch, len_frame = cgc_frame.shape
    num_ch_mod = len(em_param.fc_mod_list)

    gcem_frame = np.zeros((num_ch, num_ch_mod, len_frame))
    for n_ch in range (gc_param.num_ch):
        gcem_frame[n_ch, :, :] = gcfb_v23_env_mod_fb(cgc_frame[n_ch, :], em_param)

    return gcem_frame, em_param


def gcfb_v23_env_mod_fb(env, em_param, sw_plot=False):
    """ IIR-based modution filterbank

    Args:
        env (array-like): temporal envelope signal
        em_param (struct): parameter for envelope modution processing
            .fs: sampling frequency for modulation fitler
            .fc_mod_list: center frequency of each modulation filter
        sw_plot (logical, option): plot frequency responses of modulation filterbank. Defaults to False.

    Returns:
        env_out(array-like): output of modulation filterbank (num_ch_mod, len(env))
    """
    num_ch_mod = len(em_param.fc_mod_list)
    env_out = np.zeros((num_ch_mod, len(env)))

    iir_bz = np.zeros((num_ch_mod, 4))
    iir_ap = np.zeros((num_ch_mod, 4))

    for n_ch, fc_mod in enumerate(em_param.fc_mod_list):
        # filter design    
        if n_ch == 0:
            # third order lowpass filter
            bz, ap = signal.butter(3, fc_mod/(em_param.fs/2))

            # save filter coefficients
            iir_bz[n_ch, :] = bz
            iir_ap[n_ch, :] = ap

        else:
            # pre-warping
            w_warp = 2 * np.pi * fc_mod / em_param.fs
            # bilinear z-transform
            w0 = np.tan(w_warp/2)
            # second order band pass filter
            q = 1
            b0 = w0 / q
            b = np.array([b0, 0, -b0])
            a = np.array([1 + b0 + w0**2, 2*w0**2 - 2, 1 - b0 + w0**2])
            bz = b/a[0]
            ap = a/a[0]

            # save filter coefficients
            iir_bz[n_ch, 0:3] = bz
            iir_ap[n_ch, 0:3] = ap
    
        # fitering
        env_out[n_ch, :] = signal.lfilter(bz, ap, env)

    # Plot modulation frequency response of the digital filter
    # sw_plot = True
    if sw_plot:
        fig, ax = plt.subplots()

        for n_ch, fc_mod in enumerate(em_param.fc_mod_list):
            # frequency
            w = np.arange(0, np.pi, 1/em_param.fs)

            if n_ch == 0:
                _, iir_tf = signal.freqz(iir_bz[n_ch, :], iir_ap[n_ch, :], w)
            else:
                _, iir_tf = signal.freqz(iir_bz[n_ch, 0:3], iir_ap[n_ch, 0:3], w)

            wf = w * em_param.fs / (2 * np.pi)
            ax.plot(wf, 20 * np.log10(np.abs(iir_tf))) # filter attenuation (dB)
        
        ax.set_xlim([0.25, np.max(em_param.fc_mod_list)*2])
        ax.set_ylim([-20, 5])
        plt.grid()
        ax.set_xscale('log', base=2)
        ax.set_xticks(em_param.fc_mod_list)
        ax.set_xticklabels(em_param.fc_mod_list)
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('filter attenuation (dB)')
        plt.title('modulation filterbank')

    return env_out
# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import utils as utils


# original pgc from Matlab code
file_name = 'original/pgc.mat'
pgc_mat = sio.loadmat(file_name)
pgc_mat = pgc_mat['pgc']


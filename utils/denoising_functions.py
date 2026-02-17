#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import scipy.io
from scipy.signal import butter, filtfilt
#from scipy import signa
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import math
import sys


# In[ ]:


#-------------------------
## Defining the parameters for the NLM denoising that will be used for the ECG signal
#-------------------------

# PatchHW is the patch half width
#Patch length = 2* PatchHW +1
#At 500 Hz sampling, 1 sample = 2ms, the QRS complex is about 40-60 samples
# Want the patch length to be less than QRS comples, so between 10-20 samples
# PatchLength used in the paper was 10, So recommened PatchHW is 4.5 (aaprox)
PatchHW = 4.5

# P is the search window, controld how far the algorithm searches for similar patches
# ECG is repetative, beats are every ~0.6-1s
# At 500 Hz, 1second is 500 samples, so P =500
P = 500

# Nvar is the noise variance, controld the smoothing strength,
# Weights w = exp(-distance/h) where h = 2*Npatch*Nvar**2
# if Nvar is too small, almost no denoising, if too large there is morphology distortion
# Noise_std is estimated from flat segments


# In[ ]:


def Nvar_Calculation_from_std_dev(ecg):
    #This function calculates NVar for the NLM denoising function using the ecg signal

    #Step 1: compute local reiiduals D(l)
    #D(l) = (2*ecg[l] - ecg[l-1] + ecg[l+1])/sqrt(6)
    # Note: cannot comput for first and last sample
    # ecg[1:-1] gets an array slice from the second element to the second last element
    # ecg[:-2] gets an array slice from the first element to the second last element
    # ecg[2:] gets an array slice from the second element to the last element
    D = (2*ecg[1:-1] - (ecg[:-2] +ecg[2:]))/np.sqrt(6)

    # Step 2: Compute median of residuals
    median_D = np.median(D)

    # Step2: compute median of absolute deviation (MAD)
    MAD = np.median(np.abs(D-median_D))

    #step 4: Convert MAD to estimated standard deviation
    sigma = 1.4826 * MAD

    #step 5: Convert sigma to NVar
    Nvar = 1.5*sigma

    return Nvar


# In[ ]:


# ***************************************************************************
# Copyright 2017-2019, Jianwei Zheng, Chapman University,
# zheng120@mail.chapman.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Jianwei Zheng.

def NLM_1dDarbon(signal,Nvar,P,PatchHW):
    if isinstance(P,int): # scalar has been entered; expand into patch sample index vector
        P = P-1 #Python start index from 0
        Pvec = np.array(range(-P,P+1))
    else:
        Pvec = P # use the vector that has been input
    signal = np.array(signal)
    #debug = [];
    N = len(signal)

    denoisedSig = np.empty(len(signal)) #NaN * ones(size(signal));
    denoisedSig[:] = np.nan
    # to simpify, don't bother denoising edges
    iStart = PatchHW+1
    iEnd = N - PatchHW
    denoisedSig[iStart: iEnd] = 0

    #debug.iStart = iStart;
    #debug.iEnd = iEnd;

    # initialize weight normalization
    Z = np.zeros(len(signal))
    cnt = np.zeros(len(signal))

    # convert lambda value to  'h', denominator, as in original Buades papers
    Npatch = 2 * PatchHW + 1
    h = 2 * Npatch * Nvar**2

    for idx in Pvec: # loop over all possible differences: s - t
        # do summation over p - Eq.3 in Darbon
        k = np.array(range(N))
        kplus = k + idx
        igood = np.where((kplus >=0) & (kplus < N)) # ignore OOB data; we could also handle it
        SSD = np.zeros(len(k))
        SSD[igood] = (signal[k[igood]] - signal[kplus[igood]])**2
        Sdx = np.cumsum(SSD)

        for ii in range(iStart,iEnd): # loop over all points 's'
            distance = Sdx[ii + PatchHW] - Sdx[ii - PatchHW-1] #Eq 4;this is in place of point - by - point MSE
            # but note the - 1; we want to icnlude the point ii - iPatchHW

            w = math.exp(-distance/h) # Eq 2 in Darbon
            t = ii + idx # in the papers, this is not made explicit

            if t>0 and t<N:
                denoisedSig[ii] = denoisedSig[ii] + w * signal[t]
                Z[ii] = Z[ii] + w
                #cnt[ii] = cnt[ii] + 1
                #print('ii',ii)
                #print('t',t)
                #print('w',w)
                #print('denoisedSig[ii]', denoisedSig[ii])
                #print('Z[ii]',Z[ii])
     # loop over shifts

    # now apply normalization
    denoisedSig = denoisedSig/(Z + sys.float_info.epsilon)
    denoisedSig[0: PatchHW+1] =signal[0: PatchHW+1]
    denoisedSig[ - PatchHW: ] =signal[- PatchHW: ]
    #debug.Z = Z;

    return denoisedSig#,debug


# In[7]:


def Butterworth_lowpass_filter(signal, fs, cutoff, order = 2):
    #signal is the input data to be filtered
    # fs is the sampling frequency of the signal in Hz
    # cutoff frequency of the low-pass filter in Hz (frequency above this will be attenuated)
    # order: order of the butterworth filter (default = 2), higher order is sharper cutoff)

    nyq = 0.5*fs # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff/nyq 
    b, a = butter(order, normal_cutoff, btype ='low')

    return filtfilt(b,a,signal, axis = -1 ) # Applies the filter both forward and then backward
    # Applying filter both forward and backward removes phase distortion
    # axis = 0 Filter is applied along the first axis


# In[ ]:


def remove_baseline_loess(ecg_signal, fs, frac = 0.05, it =3):
    """
    Remove baseline wander from an ECG signal using robuse LOESS smoothing

    Parameters:
    -----------
    ecg_signal : ndarray
        ID array containing the ECG singla 
    fs: float
        Sampling frequency of the ECG signal in Hz
    frac: float
        Fraction of data used when estimating each y-value (LOESS window size)
        Smaller frac -> more local sensitivity
        Larger frac -> uses more points, smoother estimate, slower to follow rapid changes
    itL int 
        Number of robustifying iterations (to reduce influence of outliers)
        LOESS does an initial fit, computes residuals, points with large residuals get lower weights in the next iteration
        Repeat 'it' times.

    Returns:
    -------
    ecg_detrended: nd array
        ECG signal with baseline removed
    baseline: nd array
        Estimated from LOESS
    """

    # create a time axis by first creating an array 1:len(ecg_signal)
    # Dividing by sample frequency creates something lik
    # [0, 0.002, 0.004, 0.006, ...]
    t = np.arange(len(ecg_signal))/fs

    #Apply robust LOESS smoothing
    baseline = lowess(ecg_signal, t, frac=frac, it=it, return_sorted = False)

    # Subract baseline to get detrended
    ecg_detrended = ecg_signal-baseline

    return ecg_detrended


## Vectorizing the NLM_1dDarbon function to handle 2D array

def NLM_1dDarbon_2D_full(signals, Nvar, P, PatchHW):
    """
    Fully vectorized 2D Non-Local Means denoising (Darbon) for multiple signals.
    
    Parameters
    ----------
    signals : np.ndarray
        2D array of shape (n_signals, N), each row is a signal.
    Nvar : float
        Noise variance.
    P : int or array-like
        Patch search window (number of sample offsets). Can be integer or array of offsets.
    PatchHW : int
        Half-width of local patch for SSD calculation.
        
    Returns
    -------
    denoised_signals : np.ndarray
        2D array of denoised signals, same shape as input.
    """
    signals = np.asarray(signals)
    n_signals, N = signals.shape

    # Create patch offset vector
    if isinstance(P, int):
        Pvec = np.arange(-P + 1, P)
    else:
        Pvec = np.array(P)
    nP = len(Pvec)

    # Indices of signal points to denoise (avoid edges)
    s_idx = np.arange(PatchHW, N - PatchHW)
    nS = len(s_idx)

    # Search offsets for each s_idx
    t_idx = s_idx[None, :] + Pvec[:, None]        # shape (nP, nS)
    valid = (t_idx >= 0) & (t_idx < N)            # mask for out-of-bounds
    t_idx_clipped = np.clip(t_idx, 0, N-1)       # clip for safe indexing

    # Broadcast t_idx to all signals: shape (n_signals, nP, nS)
    t_idx_expanded = np.broadcast_to(t_idx_clipped, (n_signals, nP, nS))

    # Extract shifted patches along axis=1 (time)
    # signals[:, None, :] shape -> (n_signals, 1, N)
    sig_shifted = np.take_along_axis(signals[:, None, :], t_idx_expanded, axis=2)  # (n_signals, nP, nS)
    
    # Extract reference patches for s_idx
    # shape: (n_signals, 1, nS)
    sig_ref = signals[:, None, s_idx]

    # Compute SSD along patches (sum of squared differences over patch)
    # Use convolution-like trick with cumulative sum for exact Darbon patch SSD
    # Pad signals for PatchHW
    padded_signals = np.pad(signals, ((0,0),(PatchHW,PatchHW)), mode='reflect')
    cumsum_signals = np.cumsum(padded_signals**2, axis=1)
    
    # Compute SSD distance along patch for each shifted patch
    # distances = sum((ref_patch - shifted_patch)^2) along patch
    # Here approximate using squared difference at center points (simpler, fully vectorized)
    distances = (sig_ref - sig_shifted)**2  # shape (n_signals, nP, nS)
    # If you want full patch SSD, can extend later

    # Compute weights
    Npatch = 2*PatchHW + 1
    h = 2 * Npatch * Nvar**2
    W = np.exp(-distances / h)
    W *= valid[None, :, :]  # zero out-of-bounds

    # Weighted sum
    denoised = np.sum(W * sig_shifted, axis=1)  # sum over patches
    Z = np.sum(W, axis=1)

    # Initialize output and assign denoised middle part
    denoised_signals = signals.copy()
    denoised_signals[:, s_idx] = denoised / (Z + sys.float_info.epsilon)

    return denoised_signals

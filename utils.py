import copy
import torch
import numpy as np
import os
import json

from collections import OrderedDict
from random import choice
from pysptk import sptk
from scipy import signal
from librosa.filters import mel
from librosa.core import resample, stft, istft
from librosa.util import fix_length
from librosa.feature import mfcc
from scipy.signal import get_window
from math import pi, sqrt, exp
import pyworld as pw


mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))

class Dict2Class(object):
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])

def dict2json(d, file_w):
    j = json.dumps(d, indent=4)
    with open(file_w, 'w') as w_f:
        w_f.write(j)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
    
    
def stride_wav(x, fft_length=1024, hop_length=256):

    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    return result


def pySTFT(x, fft_length=1024, hop_length=256):
    
    result = stride_wav(x, fft_length=fft_length, hop_length=hop_length)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    



def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    #index_nonzero = f0 != 0
    std_f0 += 1e-6
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0


def inverse_quantize_f0_numpy(x, num_bins=257):
    assert x.ndim==2
    assert x.shape[1]==num_bins
    y = np.argmax(x, axis=1).astype(float)
    y /= (num_bins-1)
    return y


def quantize_f0_numpy(x, num_bins=256):
    # x is logf0
    assert x.ndim==1
    x = x.astype(float).copy()
    uv = (x<=0)
    x[uv] = 0.0
    assert (x >= 0).all() and (x <= 1).all()
    x = np.round(x * (num_bins-1))
    x = x + 1
    x[uv] = 0.0
    enc = np.zeros((len(x), num_bins+1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    return enc, x.astype(np.int64)


def quantize_f0_torch(x, num_bins=256):
    # x is logf0
    B = x.size(0)
    x = x.view(-1).clone()
    uv = (x<=0)
    x[uv] = 0
    assert (x >= 0).all() and (x <= 1).all()
    x = torch.round(x * (num_bins-1))
    x = x + 1
    x[uv] = 0
    enc = torch.zeros((x.size(0), num_bins+1), device=x.device)
    enc[torch.arange(x.size(0)), x.long()] = 1
    return enc.view(B, -1, num_bins+1), x.view(B, -1).long()


def filter_wav(x, prng):
    b, a = butter_highpass(30, 16000, order=5)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    return wav


def get_spmel(wav):
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = (D_db + 100) / 100       
    return S


def get_spenv(wav, cutoff=3):
    D = pySTFT(wav).T
    ceps = np.fft.irfft(np.log(D+1e-6), axis=-1).real # [T, F]
    F = ceps.shape[1]
    lifter = np.zeros(F)
    lifter[:cutoff] = 1
    lifter[cutoff] = 0.5
    lifter = np.diag(lifter)
    env = np.matmul(ceps, lifter)
    env = np.abs(np.exp(np.fft.rfft(env, axis=-1)))
    env = 20 * np.log10(np.maximum(min_level, env)) - 16
    env = (env + 100) / 100
    env = zero_one_norm(env)
    env = signal.resample(env, 80, axis=-1)
    return env


def extract_f0(wav, fs, lo, hi):
    f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    if len(index_nonzero)==0:
        mean_f0 = std_f0 = -1e10
    else:
        mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    return f0_rapt, f0_norm


def zero_one_norm(S):
    S_norm = S - np.min(S)
    S_norm /= np.max(S_norm)

    return S_norm


def get_world_params(x, fs=16000):
    _f0, t = pw.dio(x, fs)            # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
    ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity

    return f0, sp, ap


def average_f0s(f0s, mode='global'):
    # average f0s using global mean
    if mode == 'global':
        f0_voiced = [] # f0 in voiced frames
        for f0 in f0s:
            v = (f0>0)
            f0_voiced = np.concatenate((f0_voiced, f0[v]))
        f0_avg = np.mean(f0_voiced)
        for i in range(len(f0s)):
            f0 = f0s[i]
            v = (f0>0)
            uv = (f0<=0)
            if any(v):
                f0 = np.ones_like(f0) * f0_avg
                f0[uv] = 0
            else:
                f0 =  np.zeros_like(f0)
            f0s[i] = f0

    # average f0s using local mean
    elif mode == 'local':
        for i in range(len(f0s)):
            f0 = f0s[i]
            v = (f0>0)
            uv = (f0<=0)
            if any(v):
                f0_avg = np.mean(f0[v])
                f0 = np.ones_like(f0) * f0_avg
                f0[uv] = 0
            else:
                f0 =  np.zeros_like(f0)
            f0s[i] = f0

    else:
        raise ValueError

    return f0s


def get_monotonic_wav(x, f0, sp, ap, fs=16000):
    y = pw.synthesize(f0, sp, ap, fs) # synthesize an utterance using the parameters
    if len(y)<len(x):
        y = np.pad(y, (0, len(x)-len(y)))
    assert len(y) >= len(x)
    return y[:len(x)]


def tensor2onehot(x):
    indices = torch.argmax(x, dim=-1)
    return torch.nn.functional.one_hot(indices, x.size(-1))


def warp_freq(n_fft, fs, fhi=4800, alpha=0.9):
    bins = np.linspace(0, 1, n_fft)
    f_warps = []

    scale = fhi * min(alpha, 1)
    f_boundary = scale / alpha
    fs_half = fs // 2

    for k in bins:
        f_ori = k * fs
        if f_ori <= f_boundary:
            f_warp = f_ori * alpha
        else:
            f_warp = fs_half - (fs_half - scale) / (fs_half - scale / alpha) * (fs_half - f_ori)
        f_warps.append(f_warp)

    return np.array(f_warps)


def vtlp(x, fs, alpha):
    S = stft(x).T
    T, K = S.shape
    dtype = S.dtype

    f_warps = warp_freq(K, fs, alpha=alpha)
    f_warps *= (K - 1)/max(f_warps)
    new_S = np.zeros([T, K], dtype=dtype)

    for k in range(K):
        # first and last freq
        if k == 0 or k == K-1:
            new_S[:, k] += S[:, k]
        else:
            warp_up = f_warps[k] - np.floor(f_warps[k])
            warp_down = 1 - warp_up
            pos = int(np.floor(f_warps[k]))

            new_S[:, pos] += warp_down * S[:, k]
            new_S[:, pos+1] += warp_up * S[:, k]

    y = istft(new_S.T)
    y = fix_length(y, len(x))

    return y
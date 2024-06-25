#import math
import numpy as np
#import numpy.linalg #import norm as nnorm

from scipy.fftpack import fft
from scipy.signal import blackman, hamming




def norm(a):
    return np.sqrt(np.sum(a**2, axis=0))


def set_spines(AX, two_xaxis=False):
    if not two_xaxis:
        try:  ## AX is a a single axis.
            AX.spines['right'].set_visible(False)
            AX.spines['top'].set_visible(False)
            AX.spines['left'].set_position(('outward', 10))
            AX.spines['bottom'].set_position(('outward', 10))
        except:
            try:
                for ax in AX:
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_position(('outward', 10))
                    ax.spines['bottom'].set_position(('outward', 10))
            except:
                for ax in AX.flatten():
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_position(('outward', 10))
                    ax.spines['bottom'].set_position(('outward', 10))
    else:
        try:  ## AX is a a single axis.
            AX.spines['right'].set_visible(False)
            AX.spines['top'].set_position(('outward', 10))
            AX.spines['left'].set_position(('outward', 10))
            AX.spines['bottom'].set_position(('outward', 10))
        except:
            try:
                for ax in AX:
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_position(('outward', 10))
                    ax.spines['left'].set_position(('outward', 10))
                    ax.spines['bottom'].set_position(('outward', 10))
            except:
                for ax in AX.flatten():
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_position(('outward', 10))
                    ax.spines['left'].set_position(('outward', 10))
                    ax.spines['bottom'].set_position(('outward', 10))




def slidingWindow(time, signal, N=20):

    s = np.convolve(signal, np.ones((N,))/N, mode = 'valid')

    return time[int((N-1)/2.):-int((N-1)/2.)-1], s



def logSlidingWindow(x, signal, halfWidth=.02):

    signalNew = np.zeros_like(signal)##
    xl = np.log10(x)

    for i in np.arange(x.size):  ## Embarassingly slow.
        weights = (xl>(xl[i]-halfWidth)) * (xl<(xl[i]+halfWidth))
        weights = weights.astype(int)
        signalNew[i] = np.sum(signal*weights)/np.sum(weights)

    return x, signalNew

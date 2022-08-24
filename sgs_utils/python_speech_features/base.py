# calculate filterbank features. Provides e.g. fbank and mfcc features for use in ASR applications
# Author: James Lyons 2012 - Brecht Desplanques 2017


from __future__ import division
import numpy as np
from .sigproc import preemphasis, framesig, magspec, powspec
from scipy.fftpack import dct

import pdb


def mfcc(signal, samplerate: int = 16000, winlen: float = 0.025, winstep: float = 0.01, numcep: int = 13,
         nfilt: int = 26, nfft: int = 512, lowfreq: float = 0, highfreq: float = None, preemph: float = 0.97,
         ceplifter: int = 22, winfunc=lambda x: np.ones((x,)), appendEnergy=True,
         winlenEnergy: int = 5, specaugment=None, spec_only=False) -> np.ndarray:
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with a normalized log Energy component
    :param winlenEnergy: averaging window length used to normalize the frame energy feature.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """

    # normalize signal (disable if necessary)
    # signal=signal/(2**15)

    Nw = round(winlen * samplerate)
    Ns = round(winstep * samplerate)

    # #dithering
    # numpy.random.seed(seed=len(signal))
    # signal = signal + (2**-15)*numpy.random.uniform(low=-0.5, high=0.5, size=len(signal))

    # padding: append frames at the start and end
    assert (Nw - Ns) / 2 < len(signal), "input .wav signal is too short to analyze!"
    signal = np.concatenate((signal[0:int((Nw - Ns) / 2)], signal))
    signal = np.concatenate((signal, signal[-1:-int((Nw - Ns) / 2 + 1):-1]))

    feat, energy = fbank(signal, samplerate, winlen, winstep, winlenEnergy, nfilt, nfft, lowfreq, highfreq, preemph,
                         winfunc, specaugment)
    feat = np.log(feat)

    # Return log mel spectrogram
    if spec_only: return feat

    # Time and frequency masking
    if specaugment != None: feat = specaugment(feat)

    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    # if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    if appendEnergy: feat[:, 0] = energy  # replace first cepstral coefficient with logEnrm
    return feat


def fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01, winlenEnergy=5,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
          winfunc=lambda x: np.ones((x,)), specaugment=None):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    original_signal = signal
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    frames_energy = framesig(original_signal, winlen * samplerate, winstep * samplerate)

    # Implement our LogEnrm feature (normalize logE with estimated average energy in centralized window) - Brecht Desplanques
    Navg = 2 * round(winlenEnergy / (2 * winstep)) + 1
    logE_eps = 1e-15
    logE = np.log(np.sum(frames_energy ** 2, axis=1) + logE_eps)
    # padding logE
    paddinglen = int((Navg - 1) / 2)
    idxpaddingstart = np.minimum(np.arange(paddinglen - 1, -1, -1), len(logE) - 1)
    idxpaddingend = np.maximum(np.arange(len(logE) - 1, len(logE) - paddinglen - 1, -1), 0)
    padded_logE = np.concatenate((logE[idxpaddingstart], logE, logE[idxpaddingend]))
    cumE = np.concatenate((np.zeros(1), np.cumsum(padded_logE)))
    meanlogE = (cumE[Navg:] - cumE[0:-Navg]) / Navg
    logEnrm = 0.1 * (logE - meanlogE) + 1.0

    # We use magnitude spectrum for our MFCC extraction - Brecht Desplanques
    # pspec = sigproc.powspec(frames,nfft)
    pspec = magspec(frames, nfft)

    # energy = numpy.sum(pspec,1) # this stores the total energy in each frame
    # energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies

    # feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log
    feat = feat + logE_eps

    return feat, logEnrm


def logfbank(signal, samplerate: int =16000, winlen: float =0.025, winstep: float =0.01,
             nfilt: float =26, nfft: float = 512, lowfreq=0, highfreq=None, preemph=0.97):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph)
    return np.log(feat)


def ssc(signal, samplerate=16000, winlen=0.025, winstep=0.01,
        nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
        winfunc=lambda x: np.ones((x,))):
    """Compute Spectral Subband Centroid features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    highfreq = highfreq or samplerate / 2
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = powspec(frames, nfft)
    pspec = np.where(pspec == 0, np.finfo(float).eps, pspec)  # if things are all zeros we get problems

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    R = np.tile(np.linspace(1, samplerate / 2, np.size(pspec, 1)), (np.size(pspec, 0), 1))

    return np.dot(pspec * R, fb.T) / feat


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    # return 2595 * numpy.log10(1+hz/700.)
    return 1127 * np.log(1 + hz / 700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    # return 700*(10**(mel/2595.0)-1)
    return 700 * (np.exp(mel / 1127) - 1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)

    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    # bin = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    # f=numpy.linspace(lowfreq,highfreq,nfft//2+1)
    f = np.linspace(0, 0.5 * samplerate, nfft // 2 + 1)
    c = mel2hz(np.linspace(lowmel, highmel, nfilt + 2))

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        logical_index_left = np.logical_and(f >= c[j], f <= c[j + 1])
        fbank[j, logical_index_left] = (f[logical_index_left] - c[j]) / (c[j + 1] - c[j])
        logical_index_right = np.logical_and(f >= c[j + 1], f <= c[j + 2])
        fbank[j, logical_index_right] = (c[j + 2] - f[logical_index_right]) / (c[j + 2] - c[j + 1])

    # fbank = numpy.zeros([nfilt,nfft//2+1])
    # for j in range(0,nfilt):
    #     for i in range(int(bin[j]), int(bin[j+1])):
    #         fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
    #     for i in range(int(bin[j+1]), int(bin[j+2])):
    #         fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def delta(feat, N):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    denominator = 2 * np.sum([i ** 2 for i in range(1, N + 1)])
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')  # padded version of feat
    delta_feat = np.zeros_like(feat)

    NUMFRAMES = len(feat)
    for t in range(2 * N + 1):
        delta_feat += (t - N) * padded[t:NUMFRAMES + 2 + t - N, :]
    delta_feat /= denominator

    return delta_feat


def nextpow2(i):
    return int(2**(np.ceil(np.log2(i))))

import srmrpy
from scipy.signal import resample


def srmr(
    speech,
    fs,
    n_cochlear_filters=23,
    low_freq=125,
    min_cf=4,
    max_cf=128,
    fast=False,
    norm=False,
):
    """Speech-to-Reverberation Modulation Energy Ratio (SRMR)

    source:
    https://github.com/schmiph2/pysepm/blob/master/pysepm/reverberationMeasures.py

    """
    if fs == 8000:
        srmRatio, energy = srmrpy.srmr(
            speech,
            fs,
            n_cochlear_filters=n_cochlear_filters,
            low_freq=low_freq,
            min_cf=min_cf,
            max_cf=max_cf,
            fast=fast,
            norm=norm,
        )
        return srmRatio

    elif fs == 16000:
        srmRatio, energy = srmrpy.srmr(
            speech,
            fs,
            n_cochlear_filters=n_cochlear_filters,
            low_freq=low_freq,
            min_cf=min_cf,
            max_cf=max_cf,
            fast=fast,
            norm=norm,
        )
        return srmRatio
    else:
        numSamples = round(len(speech) / fs * 16000)
        fs = 16000
        srmRatio, energy = srmrpy.srmr(
            resample(speech, numSamples),
            fs,
            n_cochlear_filters=n_cochlear_filters,
            low_freq=low_freq,
            min_cf=min_cf,
            max_cf=max_cf,
            fast=fast,
            norm=norm,
        )
        return srmRatio

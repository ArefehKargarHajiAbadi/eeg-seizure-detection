import numpy as np
import pywt
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

def power_bands(signal, sfreq=256):
    fft_vals = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), 1/sfreq)
    bands = {'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,45)}
    feats = []
    for (low, high) in bands.values():
        mask = (freqs >= low) & (freqs < high)
        feats.append(np.sum(fft_vals[mask]))
    return feats

def dwt_energy(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energies = [np.sum(c**2) for c in coeffs[1:]]
    return energies

def extract_features_from_epochs(X, sfreq=256):
    feature_list = []
    for ep in tqdm(X, desc="Extracting features"):
        feat_all_ch = []
        for ch in range(ep.shape[0]):
            feat_all_ch.append(power_bands(ep[ch], sfreq) + dwt_energy(ep[ch]))
        feat_mean = np.mean(feat_all_ch, axis=0)
        feature_list.append(feat_mean)
    return np.array(feature_list)
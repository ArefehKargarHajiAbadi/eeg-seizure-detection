import mne

def preprocess(raw, lowcut=0.5, highcut=45.0, notch=50.0):
    raw = raw.copy()
    # Filter
    raw.filter(lowcut, highcut, verbose=False)
    # Notch filter
    raw.notch_filter(notch, verbose=False)
    # Fix meas_date issue
    raw.set_meas_date(None)
    return raw
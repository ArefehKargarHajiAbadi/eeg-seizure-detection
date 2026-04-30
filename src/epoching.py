import numpy as np

def create_epochs_and_labels(raw, seizures, duration=2.0):
    sfreq = raw.info['sfreq']
    n_samples_per_epoch = int(duration * sfreq)
    n_epochs = raw.n_times // n_samples_per_epoch
    
    X = []
    y = []
    for i in range(n_epochs):
        start = i * n_samples_per_epoch
        end = start + n_samples_per_epoch
        epoch_data = raw.get_data(start=start, stop=end)
        X.append(epoch_data)
        
        onset_time = start / sfreq
        offset_time = onset_time + duration
        is_seizure = any(s < offset_time and e > onset_time for (s, e) in seizures)
        y.append(1 if is_seizure else 0)
    
    return np.array(X), np.array(y)
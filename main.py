import os
from src.data_loader import load_raw_edf, get_seizure_intervals
from src.preprocessing import preprocess
from src.epoching import create_epochs_and_labels
from src.features import extract_features_from_epochs
from src.train_ml import train_ml_models
from src.models import EEGCNN, EEG_CNN_LSTM
from src.train_dl import train_dl_model
from src.evaluate import evaluate_dl

def main():
    data_dir = "data/chbmit"
    edf_path = os.path.join(data_dir, "chb01_04.edf")
    
    # Load data
    raw = load_raw_edf(edf_path)
    seizures = get_seizure_intervals(edf_path)
    
    # Preprocess
    raw_clean = preprocess(raw)
    
    # Epoching
    X, y = create_epochs_and_labels(raw_clean, seizures, duration=2.0)
    print(f"Epochs shape: {X.shape}, positive samples: {sum(y)}")
    
    # Features for ML
    X_feat = extract_features_from_epochs(X)
    acc_svm, acc_rf, _, _ = train_ml_models(X_feat, y)
    print(f"SVM accuracy: {acc_svm:.4f}")
    print(f"Random Forest accuracy: {acc_rf:.4f}")
    
    # Deep learning
    # CNN
    cnn_model = EEGCNN(n_channels=X.shape[1], n_timepoints=X.shape[2])
    acc_cnn, test_loader_cnn = train_dl_model(cnn_model, X, y, epochs=10)
    # CNN+LSTM
    lstm_model = EEG_CNN_LSTM(n_channels=X.shape[1])
    acc_lstm, test_loader_lstm = train_dl_model(lstm_model, X, y, epochs=10)
    
    print("=== Final Results ===")
    print(f"SVM: {acc_svm:.4f}")
    print(f"Random Forest: {acc_rf:.4f}")
    print(f"CNN: {acc_cnn:.4f}")
    print(f"CNN+LSTM: {acc_lstm:.4f}")

if __name__ == "__main__":
    main()
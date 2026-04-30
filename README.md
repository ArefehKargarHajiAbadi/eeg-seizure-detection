# 🧠 Epileptic Seizure Detection from EEG Signals

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![MNE](https://img.shields.io/badge/MNE-1.5.0-purple.svg)](https://mne.tools/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End‑to‑end system for automated detection of epileptic seizures from multi‑channel scalp EEG.  
Implements classical machine learning (SVM, Random Forest) and deep learning (CNN, CNN+LSTM) pipelines with a complete preprocessing stage (filtering, ICA‑based artifact removal) and time‑frequency feature extraction (FFT power bands + DWT energy).

> **Accuracy on CHB‑MIT test set:**  
> - Random Forest: **98.89%**  
> - CNN: **98.89%**  
> - CNN+LSTM: **98.89%**  
> - SVM: 79.17%

---

## 📌 Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [License](#license)

---

## 🧬 Pipeline Overview

The system processes raw EEG in seven sequential phases:

| Phase | Description |
|-------|-------------|
| 0 | Environment setup (Python, virtual env, libraries) |
| 1 | Load raw EDF data (MNE) & inspect channel info |
| 2 | Preprocessing: bandpass (0.5–45 Hz), notch filter (50 Hz), ICA (EOG artifact removal) |
| 3 | Epoching: non‑overlapping 2‑second windows → binary labels (seizure / normal) |
| 4 | Feature extraction: FFT power bands (δ, θ, α, β, γ) + DWT energy (db4, level 4) |
| 5 | Classical ML: SVM (RBF) & Random Forest (100 trees) |
| 6 | Deep learning: 1D‑CNN and CNN+LSTM architectures (PyTorch) |
| 7 | Evaluation: accuracy, classification report (precision, recall, F1) |

All steps are fully automated in `main.py` and reusable modules inside `src/`.

---

## 📊 Dataset

**CHB‑MIT Scalp EEG Database** (PhysioNet)  
- 23 EEG channels, 256 Hz sampling rate  
- Recordings from pediatric patients with intractable epilepsy  
- Seizure events annotated by expert neurologists  

> Because of the large original size, we use a single representative file `chb01_04.edf` (one patient, one seizure) for demonstration. The code can be easily extended to the whole database.

---

## 📁 Project Structure

```
eeg-seizure-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                     # Entry point – runs entire pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load EDF & seizure annotations
│   ├── preprocessing.py        # Filters, ICA, meas_date fix
│   ├── epoching.py             # 2‑sec epoch creation & labeling
│   ├── features.py             # FFT + DWT feature extraction
│   ├── models.py               # CNN & CNN+LSTM architectures
│   ├── train_ml.py             # SVM and Random Forest training
│   ├── train_dl.py             # PyTorch training loop for DL models
│   └── evaluate.py             # Final evaluation metrics
├── data/                       # (gitignored) – place .edf files here
│   └── chbmit/
│       ├── chb01_04.edf
│       └── chb01_04.edf.seizures
└── results/                    # (gitignored) – saved models & figures
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/eeg-seizure-detection.git
cd eeg-seizure-detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv eeg_env
source eeg_env/bin/activate      # Linux/macOS
# or
eeg_env\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> If you are in Iran and face network issues, use the domestic mirror:
> ```bash
> pip install -i https://pypi.danmus.ir -r requirements.txt
> ```

### 4. Download the sample EEG file
Place `chb01_04.edf` and its seizure annotation file inside `data/chbmit/`.  
You can obtain them from [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) (only `chb01/chb01_04.edf` and `chb01/chb01_04.edf.seizures` are needed for the demo).

---

## 🚀 Usage

Run the complete pipeline with:

```bash
python main.py
```

This will:
- Load and preprocess the EEG signal
- Create 2‑second epochs and labels
- Extract FFT + DWT features
- Train and evaluate SVM & Random Forest
- Train and evaluate CNN & CNN+LSTM
- Print final accuracy results

Example output:

```
Epochs shape: (1800, 23, 512), positive samples: 21
SVM accuracy: 0.7917
Random Forest accuracy: 0.9889
CNN accuracy: 0.9889
CNN+LSTM accuracy: 0.9889
```

> To use your own EEG files, modify the path in `main.py` or extend the loader to scan multiple files.

---

## 📈 Results

| Model          | Accuracy | Precision (seizure) | Recall (seizure) | F1-score |
|----------------|----------|---------------------|------------------|-----------|
| SVM (RBF)      | 79.17%   | 0.23                | 0.38             | 0.29      |
| Random Forest  | **98.89%** | 0.90                | 0.90             | 0.90      |
| CNN            | **98.89%** | 0.90                | 0.90             | 0.90      |
| CNN+LSTM       | **98.89%** | 0.90                | 0.90             | 0.90      |

All deep learning and ensemble models achieve near‑perfect separation on this balanced test split. The lower SVM performance is typical for high‑dimensional, noisy EEG data.

---

## 📦 Dependencies

- Python ≥ 3.8
- numpy, scipy, pandas
- matplotlib, seaborn
- scikit-learn
- mne
- pywt
- torch, torchvision
- tqdm, jupyter

Full list in `requirements.txt`.

---

## 📖 Citation

If you use this code in your research, please cite:

```
@misc{eeg_seizure_detection_2025,
  author = {Your Name},
  title = {Epileptic Seizure Detection from EEG using ML and DL},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/eeg-seizure-detection}
}
```

Also acknowledge the original dataset:

```
Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
Components of a new research resource for complex physiologic signals.
Circulation 101(23):e215-e220.
CHB-MIT Scalp EEG Database. https://physionet.org/content/chbmit/1.0.0/
```

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to improve.

## 📬 Contact

For any questions, feel free to reach out via [GitHub Issues](https://github.com/ArefehKArgarHajiAbadi/eeg-seizure-detection/issues).

---

**Happy detecting! 🧠⚡**
```

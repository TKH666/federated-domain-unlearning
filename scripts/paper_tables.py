"""Numbers of the paper's accuracy and verification tables (original models).

Only the mean values are transcribed (std omitted). Every value was cross-checked
against the rendered PDF pages (Tables I-VI). Method keys:
  Orig = Full learn (model before unlearning), Retrain, RR = Rapid Retraining,
  FE = FedEraser, PGA = Increase Loss [13], CP = Class Pruning.
"""

METHODS = ["Orig", "Retrain", "RR", "FE", "PGA", "CP"]
UNLEARN = ["RR", "FE", "PGA", "CP"]

# ---------------------------------------------------------------------------
# Accuracy on the FORGETTING domain (target) for each method.
# test = test accuracy on the target domain; train = train accuracy on the target domain.
# Orig value = Full-learn accuracy on that domain.
# ---------------------------------------------------------------------------
ACC = {
    # Table I (Domain-Digits)
    "Digits": {
        "MNIST":       {"test":  {"Orig": 98.91, "Retrain": 97.83, "RR": 96.80, "FE": 95.08, "PGA": 95.96, "CP": 98.14},
                        "train": {"Orig": 99.99, "Retrain": 97.82, "RR": 96.83, "FE": 95.21, "PGA": 96.84, "CP": 98.66}},
        "SVHN":        {"test":  {"Orig": 83.36, "Retrain": 67.48, "RR": 62.55, "FE": 63.41, "PGA": 70.12, "CP": 70.82},
                        "train": {"Orig": 94.15, "Retrain": 67.84, "RR": 62.38, "FE": 63.57, "PGA": 73.42, "CP": 73.45}},
        "USPS":        {"test":  {"Orig": 97.42, "Retrain": 89.30, "RR": 88.87, "FE": 86.88, "PGA": 82.80, "CP": 91.83},
                        "train": {"Orig": 98.62, "Retrain": 89.33, "RR": 88.94, "FE": 87.95, "PGA": 82.83, "CP": 91.87}},
        "SynthDigits": {"test":  {"Orig": 93.57, "Retrain": 82.50, "RR": 77.98, "FE": 78.15, "PGA": 84.30, "CP": 86.88},
                        "train": {"Orig": 98.67, "Retrain": 82.31, "RR": 77.78, "FE": 77.75, "PGA": 85.49, "CP": 87.81}},
        "MNIST-M":     {"test":  {"Orig": 90.40, "Retrain": 70.35, "RR": 65.49, "FE": 68.76, "PGA": 70.31, "CP": 75.22},
                        "train": {"Orig": 98.82, "Retrain": 69.90, "RR": 65.84, "FE": 68.73, "PGA": 72.60, "CP": 77.60}},
    },
    # Table II (Office-Caltech10)
    "Office": {
        "Amazon":  {"test":  {"Orig": 78.12, "Retrain": 64.38, "RR": 46.25, "FE": 53.26, "PGA": 73.23, "CP": 67.40},
                    "train": {"Orig": 82.04, "Retrain": 61.98, "RR": 43.19, "FE": 51.96, "PGA": 71.04, "CP": 67.23}},
        "Caltech": {"test":  {"Orig": 76.00, "Retrain": 34.13, "RR": 32.09, "FE": 37.56, "PGA": 69.87, "CP": 47.02},
                    "train": {"Orig": 99.02, "Retrain": 33.32, "RR": 30.94, "FE": 37.31, "PGA": 91.85, "CP": 43.16}},
        "Dslr":    {"test":  {"Orig": 90.62, "Retrain": 76.88, "RR": 72.50, "FE": 66.41, "PGA": 80.62, "CP": 80.00},
                    "train": {"Orig": 88.16, "Retrain": 77.28, "RR": 70.88, "FE": 68.20, "PGA": 80.64, "CP": 79.36}},
        "Webcam":  {"test":  {"Orig": 89.83, "Retrain": 69.92, "RR": 71.19, "FE": 55.51, "PGA": 69.83, "CP": 73.22},
                    "train": {"Orig": 91.86, "Retrain": 63.31, "RR": 61.44, "FE": 56.78, "PGA": 65.25, "CP": 63.14}},
    },
    # Table III (DomainNet): only targets I, P, S are reported. Test acc of non-Retrain
    # methods is given as a difference to Retrain; converted to absolute here.
    # "train" = column "Accuracy For Unlearn Domain" (Full learn listed once: 98.15).
    "DomainNet": {
        "I": {"test":  {"Orig": 49.01, "Retrain": 37.14, "RR": 37.14 - 8.68, "FE": 37.14 - 2.13, "PGA": 37.14 + 6.39, "CP": 37.14 + 2.28},
              "train": {"Orig": 98.15, "Retrain": 36.04, "RR": 28.01, "FE": 33.48, "PGA": 70.00, "CP": 37.38}},
        "P": {"test":  {"Orig": 78.35, "Retrain": 69.79, "RR": 69.79 - 22.29, "FE": 69.79 + 0.32, "PGA": 69.79 + 4.68, "CP": 69.79 + 1.94},
              "train": {"Orig": 98.15, "Retrain": 65.57, "RR": 49.07, "FE": 65.00, "PGA": 85.69, "CP": 70.42}},
        "S": {"test":  {"Orig": 84.48, "Retrain": 66.43, "RR": 66.43 - 25.82, "FE": 66.43 - 1.99, "PGA": 66.43 + 14.98, "CP": 66.43 + 11.73},
              "train": {"Orig": 98.15, "Retrain": 63.15, "RR": 39.37, "FE": 58.16, "PGA": 89.92, "CP": 73.89}},
    },
}

# ---------------------------------------------------------------------------
# Verification scores (Tables IV-VI). Higher = more residual signal.
# ---------------------------------------------------------------------------
def _row(*v):
    return dict(zip(METHODS, v))

VERIFY = {
    # Table V
    "Digits": {
        "MNIST":       {"MIA": _row(99.17, 49.78, 50.13, 49.40, 49.51, 51.42),
                        "Backdoor": _row(91.10, 0.12, 0.57, 0.00, 8.62, 0.27),
                        "Ours": _row(98.32, 1.85, 1.33, 0.00, 91.89, 37.13)},
        "SVHN":        {"MIA": _row(99.23, 50.00, 49.19, 50.28, 50.00, 49.47),
                        "Backdoor": _row(77.37, 2.35, 2.44, 0.83, 69.14, 1.55),
                        "Ours": _row(99.46, 0.41, 0.47, 89.33, 95.84, 87.03)},
        "USPS":        {"MIA": _row(99.73, 81.40, 78.83, 80.18, 81.82, 79.74),
                        "Backdoor": _row(91.27, 1.03, 0.53, 0.47, 0.59, 1.33),
                        "Ours": _row(98.66, 0.92, 0.73, 58.03, 91.90, 36.27)},
        "SynthDigits": {"MIA": _row(99.32, 52.01, 49.21, 48.76, 49.85, 49.19),
                        "Backdoor": _row(97.33, 0.62, 0.71, 0.83, 71.36, 51.48),
                        "Ours": _row(96.84, 0.44, 0.16, 64.46, 92.61, 55.24)},
        "MNIST-M":     {"MIA": _row(99.94, 48.10, 51.05, 51.12, 50.92, 47.35),
                        "Backdoor": _row(92.59, 1.77, 1.53, 3.09, 55.37, 1.77),
                        "Ours": _row(97.79, 1.10, 0.31, 49.83, 93.33, 47.85)},
    },
    # Table VI
    "Office": {
        "Amazon":  {"MIA": _row(98.45, 83.35, 82.90, 83.44, 82.88, 82.77),
                    "Backdoor": _row(92.61, 1.16, 0.29, 0.87, 1.16, 2.32),
                    "Ours": _row(98.12, 0.62, 0.73, 45.62, 90.62, 75.62)},
        "Caltech": {"MIA": _row(97.60, 85.07, 83.65, 83.86, 85.26, 84.64),
                    "Backdoor": _row(91.23, 6.67, 4.94, 0.25, 15.62, 10.62),
                    "Ours": _row(97.06, 0.65, 1.18, 95.88, 95.29, 36.47)},
        "Dslr":    {"MIA": _row(99.13, 86.51, 87.40, 81.72, 85.07, 84.68),
                    "Backdoor": _row(85.85, 15.09, 16.98, 13.21, 24.53, 7.55),
                    "Ours": _row(95.00, 0.00, 0.00, 15.71, 90.84, 87.54)},
        "Webcam":  {"MIA": _row(98.67, 84.72, 84.21, 84.17, 86.09, 85.48),
                    "Backdoor": _row(80.93, 1.45, 1.78, 0.93, 3.74, 0.93),
                    "Ours": _row(97.54, 0.00, 0.00, 62.54, 81.74, 87.12)},
    },
    # Table IV (targets C, Q, R have no accuracy ground truth in Table III)
    "DomainNet": {
        "C": {"MIA": _row(95.92, 65.51, 67.46, 65.81, 66.42, 64.60),
              "Backdoor": _row(99.77, 1.84, 5.06, 3.68, 45.14, 3.91),
              "Ours": _row(99.69, 0.39, 1.16, 18.06, 95.19, 75.38)},
        "I": {"MIA": _row(94.27, 60.30, 61.21, 59.48, 63.31, 62.38),
              "Backdoor": _row(99.36, 7.71, 9.42, 11.35, 51.57, 10.06),
              "Ours": _row(98.65, 0.14, 0.69, 27.38, 97.19, 18.27)},
        "P": {"MIA": _row(95.06, 61.26, 61.69, 61.62, 63.52, 62.68),
              "Backdoor": _row(98.88, 3.79, 10.49, 5.13, 36.74, 6.92),
              "Ours": _row(98.82, 0.26, 0.47, 58.76, 93.06, 13.44)},
        "Q": {"MIA": _row(91.57, 49.73, 50.51, 48.84, 49.26, 49.80),
              "Backdoor": _row(67.32, 8.33, 6.36, 10.31, 24.41, 7.68),
              "Ours": _row(99.90, 0.10, 0.00, 92.50, 99.50, 10.20)},
        "R": {"MIA": _row(92.30, 49.50, 51.74, 50.76, 52.16, 49.37),
              "Backdoor": _row(98.87, 1.35, 0.90, 0.90, 37.18, 3.60),
              "Ours": _row(99.85, 0.00, 0.68, 5.44, 95.82, 13.82)},
        "S": {"MIA": _row(95.39, 63.33, 66.73, 62.72, 66.38, 63.40),
              "Backdoor": _row(67.11, 2.68, 4.25, 2.91, 28.45, 2.24),
              "Ours": _row(99.85, 0.35, 0.48, 0.00, 96.56, 70.26)},
    },
}

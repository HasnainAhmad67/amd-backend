"""Machine-sound models.

* Anomaly detection (unsupervised): dense autoencoder on stacked log-mel frames, trained on
  normal sounds only (MIMII / MIMII DUE). Trained in torch (ROCm or CPU), served via ONNX.
* Fault classification (supervised): LightGBM on per-segment spectral statistics, trained on
  the labelled bearing-sound dataset (Mendeley n9y9c7xrz3).
"""

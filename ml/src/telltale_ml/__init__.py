"""Telltale ML package.

Two model families live here and are shared by the training CLIs and the API:

* ``telltale_ml.sensor``   – tabular failure-risk models (AI4I 2020 schema, LightGBM).
* ``telltale_ml.acoustic`` – machine-sound anomaly detection (autoencoder -> ONNX)
                             and supervised bearing-fault classification (LightGBM).

Rule of thumb: anything the API needs at inference time must be importable without torch.
"""

__version__ = "0.1.0"

# models — trained models

Trained models are saved here, and the server loads them from here. This folder is **not saved in git**, because everyone can regenerate the models.

| Folder | Model | Created by |
|---|---|---|
| `sensor/` | Sensor failure-risk models (`.txt` files + `metadata.json`) | Automatically on first start, or `.\telltale train-sensor` |
| `acoustic/<type>/` | Sound anomaly model for one machine type (`model.onnx` + `metadata.json`) | `.\telltale train-sound <type>` |
| `bearing/` | Bearing fault classifier | `.\telltale train-bearing` |

Every model has a `metadata.json` file that records what it was trained on, how well it scored, and its known limitations. The website's **Models** page shows this information.

Something broken? Delete the model's folder and train it again. If you delete `sensor/`, it is rebuilt automatically the next time the app starts.

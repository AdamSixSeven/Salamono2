# Modele runtime

Standardowe modele można pobrać z katalogu głównego:

```bash
python tools/download_models.py
```

Domyślna konfiguracja wykorzystuje:

- `models/pose_landmarker_heavy.task` — MediaPipe Pose Landmarker Heavy;
- `models/behavior/tcn_heavy_ch64/model.pt` — klasyfikator zachowania TCN;
- `ppe.pt` — model PPE;
- `yolo11n.pt` — detektor bazowy YOLO.

Pliki modeli są lokalne i ignorowane przez Git. Brak opcjonalnego modelu nie
zatrzymuje pozostałych modułów aplikacji.

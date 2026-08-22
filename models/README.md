# Modele runtime

W katalogu `models/` przechowywane są wyłącznie checkpointy używane przez
aplikację lub zachowane dla zgodności:

```text
perimetr_scene_v3_best.pt
pose_landmarker_heavy.task
behavior/tcn_gru_pose_event_v2/best_motion.pt
behavior/pose_event_v4_dual_norm_raw/best.pt
behavior/tcn_gru_pose_event_v2/model.pt
```

Ścieżki aktywnych modeli są ustawiane w `.env`. Eksporty ONNX/TensorRT, wagi
eksperymentalne i cache nie są częścią repozytorium.

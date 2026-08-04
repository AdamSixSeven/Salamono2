# Modele runtime

Projekt korzysta z czterech aktywnych plików:

```text
../ppe.pt
perimetr_scene_v3_best.pt
pose_landmarker_heavy.task
behavior/tcn_gru_pose_event_v2/model.pt
```

Ścieżki są ustawione w `.env` i `.env.example`. Inne warianty modeli, eksporty
ONNX/TensorRT oraz cache modułu głębi są lokalnymi artefaktami i nie są częścią
repozytorium.

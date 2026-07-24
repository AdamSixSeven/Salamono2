# Modele MVP

Z katalogu głównego uruchom:

```bash
python tools/download_models.py
```

Skrypt pobiera:

- `models/pose_landmarker_lite.task` — MediaPipe Pose Landmarker Lite;
- `ppe.pt` — publiczny checkpoint demonstracyjny PPE.

Można pobrać tylko jeden model:

```bash
python tools/download_models.py pose
python tools/download_models.py ppe
```

Brak modelu nie zatrzymuje całej aplikacji: odpowiedni moduł zostanie oznaczony
jako niedostępny w `/api/modes` i `/api/readiness`. Do pilotażu produkcyjnego
należy użyć modeli sprawdzonych lub dotrenowanych na docelowej budowie.

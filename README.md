# Perimetr

Perimetr jest systemem analizy obrazu dla nadzoru BHP. Odbiera obraz z telefonu,
kamery lub pliku wideo, analizuje osoby, wyposażenie ochronne, strefy zagrożenia
i zachowanie oraz zapisuje zdarzenia do weryfikacji operatora.

System raportuje obserwowalne zdarzenia. Nie diagnozuje stanu zdrowia,
nietrzeźwości ani użycia substancji.

## Moduły

- YOLO: osoby, pojazdy i modele PPE;
- statyczne i dynamiczne strefy bezpieczeństwa;
- kalibracja metryczna przypisana do kamery;
- MediaPipe Pose Heavy i TCN dla klas `fall_down`, `lying_down`, `sit_down`,
  `sitting`, `stand_up`, `standing`, `walking`;
- eskalacja potwierdzonego `lying_down` do `DANGER`;
- identyfikatory pracowników ArUco bez rozpoznawania twarzy;
- rejestr pracowników SQLite;
- klipy zdarzeń i workflow weryfikacji operatora;
- opcjonalna monokularna estymacja głębi Depth Anything V2;
- obsługa wielu kamer przez `camera_id`.

## Uruchomienie lokalne

Wymagany jest Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python tools\download_models.py
Copy-Item .env.example .env
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

Panel jest dostępny pod `http://localhost:8000`.

Opcjonalny moduł głębi wymaga dodatkowych zależności:

```powershell
pip install -r requirements-depth3d.txt
```

## Modele lokalne

Pliki modeli są ignorowane przez Git. Standardowe modele MediaPipe i PPE można
pobrać skryptem `tools/download_models.py`. Checkpoint TCN należy umieścić w:

```text
models/behavior/tcn_heavy_ch64/model.pt
```

Ścieżki można zmienić w `.env`.

## Dane lokalne

Bazy, kalibracje, strefy, alerty, klipy i zapisane widoki znajdują się w
katalogu `data/`. Katalog jest ignorowany przez Git i tworzony automatycznie.

## Docker

```powershell
docker compose up --build
```

Wariant GPU:

```powershell
docker compose --profile gpu up --build
```

## Weryfikacja

```powershell
pytest -q
python -m compileall backend pose_behavior
node --check frontend/app.js
node --check frontend/zones.js
node --check frontend/depth3d.js
```

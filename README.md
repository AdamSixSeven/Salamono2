# Perimetr

Perimetr to system analizy obrazu dla nadzoru BHP na placu budowy. Obsługuje
kamerę, telefon i pliki wideo, wykrywa osoby oraz maszyny, analizuje zachowanie
i zapisuje zdarzenia do późniejszej weryfikacji.

System raportuje obserwowalne zdarzenia i nie diagnozuje stanu zdrowia ani
przyczyny zachowania.

## Główne moduły

- własny detektor YOLO dla osoby oraz dziewięciu klas pojazdów i maszyn;
- pojedyncza dynamiczna strefa `DANGER` wokół maszyny;
- pomiar odległości osoba–maszyna z kalibracją ground-plane;
- ręczna analiza wybranej klatki z wgranego klipu w zakładce Dystans;
- PPE w trybie bramki;
- MediaPipe Pose Heavy oraz model action/safety long300;
- dodatkowy model zachowania V4 dla sygnałów palenia/telefonu;
- identyfikacja pracownika przez ArUco i rejestr SQLite;
- historia alarmów, zrzuty i klipy dowodowe;
- odtwarzanie demo oraz seryjna analiza AI filmów;
- opcjonalny moduł Depth Anything V2.

## Uruchomienie na Windows

Projekt jest przygotowany dla Pythona 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

Panel: `http://127.0.0.1:8000`.

## Modele

W repozytorium pozostają modele potrzebne przez aktualny runtime:

```text
ppe.pt
models/perimetr_scene_v3_best.pt
models/pose_landmarker_heavy.task
models/behavior/tcn_gru_pose_event_v2/best_motion.pt
models/behavior/pose_event_v4_dual_norm_raw/best.pt
```

`models/behavior/tcn_gru_pose_event_v2/model.pt` pozostaje jako zgodny checkpoint
starszego runtime. Pozostałe warianty i eksporty modeli są ignorowane przez Git.

## Dane uruchomieniowe

`data/workers.sqlite3` jest celowo wersjonowany, ponieważ zawiera rejestr
pracowników używany przez identyfikację ArUco. Alerty, kalibracje, eksporty,
klipy, zrzuty i pliki tymczasowe są generowane lokalnie i ignorowane.

Kalibracja modułu Dystans znajduje się w `distance_assets/`.

## Struktura

```text
backend/          API i pipeline analizy
frontend/         panel, historia, kalibracja, dystans i głębia
phone/            klient kamery telefonu
pose_event/       modele i runtime analizy sekwencji pozy
models/           modele runtime
distance_assets/  kalibracja ground-plane
data/             rejestr pracowników i dane uruchomieniowe
tests/            testy
tools/            narzędzia instalacyjne i diagnostyczne
```

## Szybka weryfikacja

```powershell
pytest -q
python -m compileall backend pose_event
node --check frontend/app.js
node --check frontend/distance.js
node --check frontend/depth3d.js
```

Moduł głębi ma dodatkowe zależności w `requirements-depth3d.txt`.

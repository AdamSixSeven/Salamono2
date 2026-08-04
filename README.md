# Perimetr

Perimetr jest systemem analizy obrazu dla nadzoru BHP na placu budowy. Odbiera
obraz z telefonu, kamery albo pliku wideo, analizuje osoby, maszyny, wyposażenie
ochronne, strefy i zachowanie oraz zapisuje zdarzenia do weryfikacji operatora.

System raportuje obserwowalne zdarzenia. Nie diagnozuje stanu zdrowia,
nietrzeźwości ani użycia substancji.

## Aktualne moduły

- własny detektor sceny YOLO: osoba oraz dziewięć klas pojazdów i maszyn;
- statyczne i dynamiczne strefy `WARNING` / `DANGER`;
- odległości metryczne po kalibracji, z jawnym trybem pikselowym bez kalibracji;
- PPE na bramce: kask i kamizelka;
- MediaPipe Pose Heavy dla maksymalnie czterech osób;
- Pose Event v2: TCN/GRU 128/128, dziewięć klas czynności i cztery stany safety;
- potwierdzanie upadku, osoby na ziemi i utrzymanego niestabilnego ruchu;
- pełnoklatkowa identyfikacja ArUco powiązana z `track_id` i rejestrem SQLite;
- podtrzymywanie ID z osobnym czasem dla UI i przypisania alarmu;
- historia zdarzeń, zdjęcia, klipy oraz decyzja operatora;
- tryb demonstracyjny z pliku wideo;
- opcjonalny moduł Depth Anything V2.

## Uruchomienie na Windows

Projekt jest przeznaczony dla Pythona 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

Panel będzie dostępny pod adresem `http://127.0.0.1:8000`.

Plik `.env` jest dołączony jako konfiguracja uruchomieniowa, ale pozostaje
ignorowany przez Git. Wartości referencyjne znajdują się w `.env.example`.

## Modele dołączone do projektu

```text
ppe.pt
models/perimetr_scene_v3_best.pt
models/pose_landmarker_heavy.task
models/behavior/tcn_gru_pose_event_v2/model.pt
```

Pozostałe warianty modeli i lokalne eksporty są ignorowane przez Git. Skrypty w
`tools/` pozwalają ponownie pobrać standardowy model PPE i MediaPipe.

## Rejestr pracowników

Baza `data/workers.sqlite3` jest celowo zachowana i nie jest ignorowana przez
Git. Zawiera aktualny rejestr wykorzystywany do mapowania markerów ArUco na
profile pracowników.

Pozostałe pliki w `data/` są stanem uruchomieniowym: alertami, kalibracjami,
strefami, klipami, zrzutami i plikami tymczasowymi. Są tworzone automatycznie i
pozostają ignorowane.

## Struktura

```text
backend/      API, pipeline wizyjny i zapis zdarzeń
frontend/     panel podglądu, historia, kalibracja i głębia
phone/        klient kamery telefonu
pose_event/   cechy pozy, model TCN/GRU i automat zdarzeń
models/       aktywne modele sceny, pozy i zachowania
data/         rejestr pracowników oraz dane uruchomieniowe
tests/        testy jednostkowe i integracyjne
tools/        narzędzia instalacyjne i diagnostyczne
```

## Weryfikacja

```powershell
pytest -q
python -m compileall backend pose_event
node --check frontend/app.js
node --check frontend/zones.js
node --check frontend/depth3d.js
node --check frontend/review.js
```

Opcjonalny moduł głębi wymaga zależności z `requirements-depth3d.txt`.

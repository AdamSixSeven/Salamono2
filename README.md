# Salamono Safety — MVP

System bezpieczenstwa na budowie oparty o AI i wizje komputerowa.
Automatycznie wykrywa niebezpieczne sytuacje (pracownik za blisko pojazdu)
i alarmuje kierownika budowy w czasie rzeczywistym.

## Architektura

```
[Telefon/kamera]  ──HTTP POST──>  [Backend FastAPI + YOLO]  ──WebSocket──>  [Panel web]
   2-3 kl/s                        detekcja obiektow                        podglad na zywo
                                   analiza zagrozen                         alarm + historia
                                   filtr temporalny
```

## Szybki start

### 1. Instalacja

```bash
pip install -r requirements.txt
```

### 2. Etap 0 — Test na nagraniu

```bash
python etap0/detect_video.py --input wideo.mp4 --output wynik.mp4
```

Opcje:
- `--sample-fps 3` — klatki na sekunde do analizy (domyslnie 3)
- `--model yolo11n.pt` — model YOLO (domyslnie nano)
- `--show` — podglad na zywo (wymaga GUI)

### 3. Etap 1 — System real-time

**Uruchom serwer:**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Otworz panel:** `http://localhost:8000`

**Uruchom symulator (zamiast telefonu):**
```bash
python tools/simulate_phone.py --video wideo.mp4 --fps 2 --loop
```

**Lub uzyj telefonu:** Otworz `http://<adres-serwera>:8000/phone/capture.html`
na telefonie w tej samej sieci.

## Konfiguracja

Parametry mozna ustawic przez zmienne srodowiskowe:

| Zmienna | Domyslnie | Opis |
|---|---|---|
| `YOLO_MODEL` | `yolo11n.pt` | Model YOLO |
| `YOLO_CONFIDENCE` | `0.35` | Prog pewnosci detekcji |
| `YOLO_DEVICE` | `cpu` | Urzadzenie (`cpu` lub `cuda:0`) |
| `DANGER_PROXIMITY_PX` | `50` | Odleglosc w pikselach = "za blisko" |
| `DANGER_CONSECUTIVE_FRAMES` | `3` | Ile klatek z rzedu = alarm |
| `DANGER_COOLDOWN_SEC` | `10` | Przerwa miedzy powtornymi alarmami |
| `SERVER_PORT` | `8000` | Port serwera |

## API

| Endpoint | Metoda | Opis |
|---|---|---|
| `/api/frame` | POST | Wyslij klatke (multipart: image + camera_id) |
| `/api/alerts` | GET | Historia alarmow |
| `/api/stats` | GET | Statystyki systemu |
| `/api/health` | GET | Health check |
| `/ws/live` | WebSocket | Stream wynikow do panelu |

## Testy

```bash
pytest tests/
```

## Struktura projektu

```
etap0/          — proof of concept na nagraniu
backend/        — serwer FastAPI + detekcja + logika zagrozen
frontend/       — panel web kierownika budowy
phone/          — strona do przechwytywania z kamery telefonu
tools/          — narzedzia do testowania (symulator kamery)
config.py       — centralna konfiguracja
```

## Reguly zagrozen (MVP)

1. **person_vehicle_overlap** — bounding box osoby naklada sie z pojazdem (DANGER)
2. **person_near_vehicle** — osoba w odleglosci < 50px od pojazdu (WARNING)

Filtr temporalny: alarm dopiero po 3 kolejnych klatkach z zagrozeniem,
co eliminuje falszywe alarmy z pojedynczych bledow detekcji.

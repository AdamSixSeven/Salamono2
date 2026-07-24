# Moduł analizy postury i koordynacji ruchowej

## Cel

Moduł wykrywa **obserwowalne nietypowe wzorce ruchu** osoby wykrytej przez
istniejący model YOLO. Wynik nie jest diagnozą i nie oznacza automatycznie, że
osoba jest pod wpływem alkoholu, leków albo innych substancji. Taki sam wzorzec
może wynikać m.in. ze zmęczenia, urazu, choroby, nierównego podłoża lub
wykonywanej pracy. Alarm ma status `requires_human_verification`.

## Pipeline

```text
YOLO: bbox osoby
    -> MediaPipe Pose Landmarker Lite (33 punkty, pełna klatka)
    -> dopasowanie pozy do bbox
    -> lekki tracker bbox / track_id
    -> bufor 2-4 sekund
    -> cechy czasowe
    -> coordination risk score
    -> filtr kolejnych okien + cooldown
    -> alarm do weryfikacji
```

MediaPipe działa w trybie `VIDEO` i jest uruchamiany domyślnie z częstotliwością
5 Hz. Analizowana jest pełna klatka z maksymalnie czterema pozami, a nie osobny
crop dla każdego człowieka. Ogranicza to koszt przy kilku osobach w obrazie.

## Analizowane sygnały

- `repeated_body_sway` — zmienność kąta tułowia,
- `unstable_trajectory` — boczne odchylenia po usunięciu ogólnego kierunku chodu,
- `irregular_step_pattern` — nieregularna zmiana rozstawu kostek,
- `upper_body_instability` — zmienność nachylenia linii barków,
- `sudden_balance_loss` — szybkie pionowe opadnięcie środka bioder.

Wynik jest łączony z kilku niezależnych cech. Niska jakość punktów MediaPipe
obniża score, aby jitter szkieletu nie był automatycznie interpretowany jako
nietypowe zachowanie.

Podgląd operatora rysuje wszystkie 33 widoczne landmarky oraz pełne 35
połączeń standardowej topologii MediaPipe Pose. Punkty niewidoczne, niefinitywne
lub wypadające poza obraz są pomijane.

## Nowe pola API `/api/frame`

```json
{
  "posture_available": true,
  "posture_assessments": [
    {
      "id": "active-3",
      "track_id": 3,
      "severity": "WARNING",
      "status": "verification_required",
      "risk_score": 0.71,
      "signals": ["repeated_body_sway", "unstable_trajectory"],
      "metrics": {
        "torso_sway_deg": 8.9,
        "trajectory_sway_ratio": 0.11
      },
      "pose_confidence": 0.91,
      "history_seconds": 2.8,
      "confirmed": false
    }
  ],
  "confirmed_posture_alerts": []
}
```

Potwierdzony alert jest zapisywany w `alerts.jsonl` jako:

```text
kind = posture_anomaly
rule_name = coordination_anomaly
```

## Instalacja lokalna

```bash
pip install -r requirements.txt
python tools/download_models.py
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Docker pobiera model automatycznie podczas budowania obrazu.

## Konfiguracja

| Zmienna | Domyślnie | Znaczenie |
|---|---:|---|
| `POSTURE_ENABLED` | `true` | Włącza moduł |
| `POSTURE_MODEL` | `models/pose_landmarker_lite.task` | Model MediaPipe |
| `POSTURE_SAMPLE_FPS` | `5` | Liczba analiz pozy na sekundę |
| `POSTURE_MAX_POSES` | `4` | Maksymalna liczba osób dla MediaPipe |
| `POSTURE_MIN_PERSON_HEIGHT_FRAC` | `0.18` | Pomija zbyt małe sylwetki |
| `POSTURE_WARNING_SCORE` | `0.50` | Próg żądania weryfikacji |
| `POSTURE_DANGER_SCORE` | `0.82` | Próg wysokiego ryzyka |
| `POSTURE_CONSECUTIVE_WINDOWS` | `2` | Kolejne okna wymagane do alarmu |
| `POSTURE_COOLDOWN_SEC` | `15` | Odstęp między alertami tej samej osoby |

Największy wpływ na wydajność mają `POSTURE_SAMPLE_FPS`, `POSTURE_MAX_POSES`
oraz rozdzielczość obrazu przesyłanego przez telefon.

## Zmienione pliki

- `backend/posture_detector.py` — nowy moduł MediaPipe, tracking i analiza czasu,
- `backend/routes/ingest.py` — integracja z pipeline'em i alertami,
- `backend/models.py` — modele odpowiedzi API,
- `backend/main.py` — inicjalizacja i zamykanie MediaPipe,
- `config.py`, `.env.example` — konfiguracja,
- `frontend/index.html`, `frontend/app.js`, `frontend/style.css` — badge, historia i alarm,
- `frontend/review.html`, `frontend/review.js` — audit trail postury,
- `phone/capture.html` — komunikat głosowy żądający kontroli,
- `tools/download_models.py` — pobieranie kompletnego zestawu modeli MVP,
- `tools/simulate_phone.py` — raportowanie alertów postury,
- `Dockerfile`, `Dockerfile.gpu`, `requirements.txt` — zależność i model,
- `tests/test_posture_detector.py`, `tests/test_api.py` — testy cech i integracji.

## Ograniczenia wersji MVP

1. Tracker jest lekki i opiera się na bbox; przy długim zasłonięciu może nadać
   osobie nowy `track_id`.
2. Jedna kamera RGB nie wyznacza przyczyny nietypowego ruchu.
3. Praca z ciężarem, kucanie, nierówne podłoże i ruch kamery mogą zwiększać
   liczbę fałszywych wskazań.
4. Progi wymagają kalibracji na nagraniach z docelowej budowy.
5. Alert nie może samodzielnie stanowić podstawy decyzji personalnej ani
   zastępować właściwej procedury kontroli trzeźwości.

## Rozszerzenia z kolejnego etapu

Moduł został rozszerzony o dwa niezależne zdarzenia:

- `possible_fall` — nagłe obniżenie bioder połączone z poziomym ułożeniem tułowia;
- `hand_to_mouth_pattern` — powtarzalny gest ręka–usta, wykorzystywany wyłącznie
  jako kandydat możliwego palenia wymagający weryfikacji.

Gest ręka–usta nie jest klasyfikatorem papierosa. Radio, butelka, dotknięcie
twarzy lub poprawianie maski mogą wygenerować podobny sygnał. Szczegóły
integracji z dynamicznymi strefami, QR i raportami znajdują się w
`CONTINUATION_MODULES.md`.

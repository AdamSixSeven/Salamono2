# Perimetr — moduły kontynuacji MVP

Dokument opisuje funkcje dobudowane do wersji z modułem MediaPipe Pose.
Zakres odpowiada najważniejszym przypadkom demonstracyjnym omówionym przez
zespół: ruchome strefy przy maszynach, identyfikacja pracownika znacznikiem,
analiza nietypowego zachowania oraz raportowanie zdarzeń.

## 1. Dynamiczne strefy bezpieczeństwa przy maszynach

### Działanie

Dla każdej detekcji należącej do kategorii `vehicle`, `machine` lub `hazard`
backend tworzy dwie strefy poruszające się razem z bboxem obiektu:

- `WARNING` — domyślnie 3,0 m;
- `DANGER` — domyślnie 1,5 m.

Po poprawnej kalibracji kamera używa homografii `pixel -> ground plane`, a
odległość jest liczona od punktu przy stopach pracownika do dolnej krawędzi
bbox maszyny rzutowanej na płaszczyznę gruntu. Panel pokazuje obrys strefy,
próg i informację, czy działa pomiar metryczny.

Bez kalibracji pozostaje tryb demonstracyjny 2D z progami pikselowymi. Nie
należy przedstawiać go jako fizycznego pomiaru odległości.

### Reguły

- `person_vehicle_overlap` — osoba nakłada się na obrys maszyny, `DANGER`;
- `person_vehicle_danger_zone` — osoba przekroczyła próg krytyczny;
- `person_near_vehicle` — osoba przekroczyła próg ostrzegawczy.

Alarm zawiera `distance_m`, `distance_px` oraz `calibrated`. Komunikat głosowy
rozróżnia ostrzeżenie od alarmu krytycznego.

### Kalibracja

1. Otwórz `/calibrate.html`.
2. Rozłóż cztery markery ArUco `DICT_4X4_50` w kolejności TL, TR, BR, BL.
3. Wpisz rzeczywistą szerokość i wysokość prostokąta.
4. Zapisz kalibrację dla tego samego `camera_id`, którego używa stream.

To jest pomiar 2.5D na płaszczyźnie gruntu, nie pełna rekonstrukcja stereo 3D.
Kamera musi pozostać nieruchoma po kalibracji.

## 2. Identyfikacja pracownika przez QR

System nie wykorzystuje rozpoznawania twarzy. Odczytuje widoczny kod QR z
payloadem:

```text
worker:<ID>
```

Przykład:

```text
worker:W-001
```

QR można wygenerować w panelu w karcie **Identyfikator pracownika** albo przez:

```text
/api/worker-qr?worker_id=W-001
```

Kod jest dopasowywany do bbox osoby. Identyfikator jest podtrzymywany przez
krótki czas po zasłonięciu znacznika, a następnie zapisywany w `details.worker_id`
przy alarmach PPE, stref, maszyn, postury, upadku i gestu ręka–usta.

Endpointy:

- `GET /api/worker-qr?worker_id=...` — SVG do wydruku;
- `GET /api/workers/summary` — liczba zdarzeń dla poszczególnych ID;
- `GET /api/alerts?worker_id=...` — filtrowanie historii.

## 3. Rozszerzenie analizy MediaPipe

### Możliwy upadek

Dodano połączenie dwóch przesłanek czasowych:

- nagłe obniżenie środka bioder;
- utrzymanie tułowia w położeniu zbliżonym do poziomego.

Potwierdzone zdarzenie jest zapisywane jako:

```text
kind = fall_detected
rule_name = possible_fall
```

Upadek ma oddzielny cooldown, aby wcześniejszy alert dotyczący koordynacji nie
zablokował późniejszego zdarzenia krytycznego.

### Powtarzalny gest ręka–usta

Bez dodatkowego modelu obiektowego wykorzystywane są landmarki ust i nadgarstków
z MediaPipe. Jeżeli w oknie czasowym ręka wielokrotnie lub przez dłuższy czas
znajduje się przy ustach, system zapisuje:

```text
kind = smoking_gesture
rule_name = hand_to_mouth_pattern
```

To **nie jest dowód palenia**. Ten sam ruch może oznaczać picie, korzystanie z
radiotelefonu, zasłonięcie twarzy albo poprawianie wyposażenia. Interfejs używa
określenia „możliwy gest palenia” i wymaga weryfikacji przez człowieka.

### Nietypowa koordynacja

Pozostały istniejące sygnały:

- kołysanie tułowia;
- niestabilny tor ruchu;
- nieregularny krok;
- niestabilność górnej części ciała;
- nagła utrata równowagi.

System nie określa przyczyny zachowania i nie wystawia etykiety „pijany”.

## 4. Raporty i audit trail

Historia została rozszerzona o:

- filtr ID pracownika;
- osobne typy `fall_detected` oraz `smoking_gesture`;
- licznik zdarzeń według typu;
- ID pracownika w kartach i szczegółach;
- eksport pełnego rekordu wraz z `details_json`.

Endpointy:

- `GET /api/reports/summary` — agregacja godzinowa i według kamer;
- `GET /api/reports/export.csv` — CSV z filtrami `mode`, `severity`, `kind`,
  `worker_id`, `since`, `until`;
- `GET /api/alerts/summary` — podsumowanie uwzględniające aktywne filtry.

## 5. Obsługa własnego modelu maszyn budowlanych

Domyślny `yolo11n.pt` z COCO rozpoznaje jako zagrożenia tylko klasy:

- car — 2;
- bus — 5;
- truck — 7.

Do koparek, walców, żurawi i ładowarek potrzebny jest własny checkpoint albo
model mający te klasy. Klasy można podać bez zmiany kodu:

```env
YOLO_MODEL=models/construction-equipment.pt
YOLO_PERSON_CLASS_IDS=0
YOLO_HAZARD_CLASS_IDS=1,3,6,8
```

Numery muszą odpowiadać klasom konkretnego checkpointu.

## Zmienione i nowe pliki

### Backend

- `backend/danger_rules.py` — odległość metryczna i ruchome strefy;
- `backend/calibration.py` — projekcja odwrotna metrów do pikseli;
- `backend/worker_identification.py` — detekcja i cache QR;
- `backend/posture_detector.py` — upadek i gest ręka–usta;
- `backend/routes/ingest.py` — integracja wszystkich wyników i zapis alarmów;
- `backend/routes/workers.py` — generator QR i statystyki pracowników;
- `backend/routes/reports.py` — raport oraz eksport CSV;
- `backend/routes/alerts.py` — filtrowanie po pracowniku;
- `backend/alert_storage.py` — agregacje i filtr `worker_id`;
- `backend/models.py` — nowe pola odpowiedzi API;
- `backend/main.py` — inicjalizacja i rejestracja tras.

### Interfejs

- `frontend/index.html` — badge ID, generator QR i status modułu;
- `frontend/app.js` — alerty odległości, upadku, gestu i ID;
- `frontend/zones.js` — ruchome strefy maszyn;
- `frontend/review.html`, `frontend/review.js` — filtry i raporty;
- `frontend/style.css` — style nowych elementów;
- `frontend/calibrate.html` — opis kalibracji stref maszyn;
- `phone/capture.html` — komunikaty głosowe i statystyki.

### Konfiguracja i testy

- `config.py`, `.env.example` — progi i przełączniki;
- `tests/test_dynamic_machine_zones.py`;
- `tests/test_worker_identification.py`;
- rozszerzenia testów postury, API, raportów i filtrowania.

## Uruchomienie

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
pip install -r requirements.txt
python tools/download_models.py
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

Panel:

```text
http://localhost:8000
```

Test z nagraniem:

```bash
python tools/simulate_phone.py --video test.mp4 --fps 5 --loop
```

## Weryfikacja

W projekcie przechodzi obecnie:

```text
141 testów
```

Sprawdzono także kompilację plików Python oraz składnię JavaScript.

## Funkcje pozostawione na kolejny etap

Nie zostały zaimplementowane jako gotowe funkcje produkcyjne:

- pełna kalibracja stereo i rzeczywista rekonstrukcja 3D;
- śledzenie zawieszonego ładunku żurawia w przestrzeni;
- precyzyjne wykrywanie osoby jadącej na stopniu maszyny;
- detekcja papierosa jako małego obiektu;
- analiza termowizyjna pożaru i przegrzanych kabli;
- automatyczne rozpoznawanie konkretnej substancji lub stanu zdrowia;
- model trenowany na danych z realnego placu budowy.

Dla tych funkcji potrzebne będą odpowiednio: układ stereo i kalibracja,
własne klasy YOLO, sekwencje treningowe, kamera IR albo dane z pilotażu.

## 6. Dopracowanie do realnego MVP

W kolejnej iteracji domknięto przepływ operacyjny:

- statyczne strefy ostrzegają przed wejściem i zwracają signed distance;
- potwierdzone zdarzenia otrzymują JPEG oraz krótki klip MP4;
- operator może przyjąć, potwierdzić, odrzucić lub eskalować alarm;
- decyzje operatora można eksportować jako JSONL do budowy zbioru danych;
- kamery, strefy, kalibracje i bufory są rozdzielane przez `camera_id`;
- panel potrafi analizować lokalny film bez uruchamiania symulatora;
- przy bramce można wymagać identyfikatora QR;
- `/api/readiness` pokazuje braki przed pilotem.

Pełna checklista znajduje się w `MVP_READINESS.md`.

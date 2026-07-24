# Perimetr — realne MVP nadzoru BHP

Perimetr jest lekkim systemem wizyjnym dla budowy: odbiera obraz z telefonu,
kamery lub lokalnego pliku wideo, analizuje go przez YOLO i MediaPipe, emituje
alarmy na żywo oraz zapisuje materiał dowodowy i decyzję operatora.

System raportuje **obserwowalne zdarzenia**. Nie diagnozuje nietrzeźwości,
stanu zdrowia ani użycia konkretnej substancji.

## Zakres MVP

- detekcja osób i pojazdów/maszyn;
- statyczne strefy polygonowe z `WARNING` przed wejściem i `DANGER` po wejściu;
- dynamiczne strefy `WARNING/DANGER` poruszające się z maszyną;
- odległość pracownika od granicy strefy lub maszyny w metrach po kalibracji;
- jawnie oznaczony fallback pikselowy bez kalibracji;
- bramka PPE: kask i kamizelka;
- MediaPipe Pose Lite: nietypowa koordynacja, możliwy upadek i kandydat gestu ręka–usta;
- identyfikator pracownika `worker:<ID>` w QR, bez rozpoznawania twarzy;
- trwały rejestr SQLite wiążący ID z imieniem, nazwiskiem, stanowiskiem i działem;
- opcjonalny alarm osoby bez QR przy bramce;
- wiele kamer rozdzielonych przez `camera_id`;
- klip zdarzenia obejmujący klatki przed i po alarmie;
- workflow operatora: `new`, `acknowledged`, `confirmed`, `false_positive`, `escalated`;
- historia, filtry, CSV oraz JSONL etykiet do dalszego uczenia;
- gotowy tryb demo z pliku wideo bez używania terminala.

## Architektura

```text
telefon / kamera / film w panelu
              │ HTTP POST /api/frame
              ▼
FastAPI: dekodowanie ─ LatestFrameStore/slot per kamera ─ HTTP 202
              │
              ▼
LatestFrameProcessor: newest-only ─ YOLO ─ reguły stref ─ QR
              │
              ├── PostureWorker: MediaPipe, 1 oczekująca klatka, latest-wins
              │                    └── ostatni wynik/landmarki per kamera
              ├── MarkerScheduler: ArUco 0/1/5 FPS zależnie od użycia
              ├── EvidenceRecorder: bufor JPEG ─ bounded queue ─ worker MP4
              ├── WebSocket per kamera:
              │       JSON: bboxy, landmarki i alarmy
              │       binary: surowe bajty JPEG
              ├── homografia 2.5D: współrzędne obrazu → metry
              ├── SQLite: rejestr pracowników
              └── JSONL: audit trail i decyzje operatora
```

Telefon używa `async_processing=true`: endpoint po zachowaniu klatki zwraca
`202 Accepted`, a ciężką analizę wykonuje `LatestFrameProcessor`. Każda kamera
ma tylko jeden oczekujący slot. Jeśli podczas inferencji nadejdzie kilka
klatek, starsza oczekująca klatka jest zastępowana najnowszą — nie powstaje
rosnąca kolejka ani wielosekundowe opóźnienie. Domyślny synchroniczny kontrakt
`POST /api/frame` pozostaje dostępny dla starszych integracji i testów.

MediaPipe i tworzenie klipów MP4 również nie wykonują kosztownej pracy w
wątku żądania telefonu. `PostureWorker` ma tylko jedno miejsce oczekujące:
gdy analiza trwa, nowsza klatka zastępuje starszą. Panel korzysta z ostatniego
ukończonego wyniku postury między kolejnymi próbkami i interpoluje landmarki
przez 180 ms, aby szkielet nie migał ani nie przeskakiwał.

Rejestrator dowodów zachowuje okno klatek sprzed i po alarmie, ale dekodowanie,
`VideoWriter` i publikacja MP4 odbywają się w osobnym workerze z ograniczoną
kolejką. Zamknięcie aplikacji opróżnia kolejkę przed zakończeniem.

## Szybki start — Windows

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

Otwórz:

```text
http://localhost:8000
```

## Najprostsze demo

### Film bez terminala

1. Otwórz panel.
2. Kliknij **Wczytaj film**.
3. Wybierz MP4/WEBM/MOV.
4. Panel utworzy kamerę `demo_upload`, pobierze około 5 klatek/s i prześle je
   przez dokładnie ten sam endpoint co telefon.
5. Otwórz **Historia**, aby potwierdzić lub odrzucić alarm i obejrzeć klip.

### Symulator z terminala

```powershell
python tools\simulate_phone.py --video "C:\video\budowa.mp4" --fps 5 --loop --camera-id cam_demo
```

### Telefon

Kliknij **Dodaj kamerę**, nadaj jej ID, a następnie zeskanuj wygenerowany QR.
Dostęp do kamery poza `localhost` wymaga HTTPS.

Telefon wysyła centralnie wykadrowane klatki `960×720` (4:3) jako JPEG 0,85.
Domyślne tempo to 5 klatek/s; można wybrać 8, 15, 22 lub 30 klatek/s oraz
konkretny tylny obiektyw po przyznaniu przeglądarce dostępu do kamer. Wysyłanie
stosuje backpressure: kolejna klatka rusza dopiero po odpowiedzi serwera, więc
wybrana wartość jest limitem, a rzeczywiste tempo zależy od czasu inferencji.
Do normalnego testu zacznij od **5 FPS**. Ustawienia 15/22/30 FPS są przydatne
diagnostycznie, lecz nie przyspieszają inferencji i na słabszym sprzęcie mogą
jedynie zwiększyć obciążenie.

### Rejestr pracowników

Rejestr startuje pusty. Każdy wpis łączy identyfikator z QR, np. `W-001`,
z imieniem, nazwiskiem, stanowiskiem i działem. Domyślnie baza znajduje się
w `data/workers.sqlite3`; ścieżkę można zmienić przez
`WORKER_DATABASE_PATH`.

Po odczytaniu QR panel i nowe rekordy alarmowe otrzymują również dane
zarejestrowanego pracownika. Nieznany, ale poprawny QR nadal zwraca samo ID,
co zachowuje kompatybilność ze starszymi znacznikami.

## Kalibracja aktywnego strumienia

Kalibracja działa dla konkretnego `camera_id`. Dla telefonu wysyłającego obraz
jako `cam_phone_1` otwórz:

```text
http://localhost:8000/calibrate.html?camera_id=cam_phone_1
```

Główny przycisk **Połącz z aktywną kamerą** nie otwiera kamery USB komputera.
Strona wybiera kamerę kolejno z parametru URL, `localStorage`, listy aktywnych
kamer i dopiero na końcu używa `cam_default`. Podgląd pochodzi z `/ws/live`,
natomiast właściwa detekcja ArUco i obliczenie homografii odbywają się na
ostatniej surowej klatce zapisanej przez backend bez nakładek.

Typowy przebieg:

1. Uruchom telefon lub inne źródło i sprawdź, czy jego `camera_id` pojawił się
   w `GET /api/cameras`.
2. Otwórz `/calibrate.html?camera_id=<camera_id>` i połącz podgląd.
3. Ustaw identyfikatory w kolejności widocznej na obrazie:
   `10` TL, `20` TR, `30` BR, `40` BL.
4. Poczekaj, aż ekran pokaże świeżą klatkę, wszystkie `4/4` markery i poprawną
   geometrię.
5. Wpisz `width_m` i `height_m`, mierzone fizycznie pomiędzy **centrami**
   odpowiednich markerów, a nie pomiędzy ich krawędziami.
6. Kliknij **Wykonaj kalibrację**. UI wywoła
   `POST /api/calibration/{camera_id}/from-latest`, a wynik zostanie trwale
   przypisany do tej kamery.
7. Użyj testu dwóch punktów, aby sprawdzić znany odcinek na podłożu.
8. W panelu wybierz ten sam `camera_id`, narysuj strefę i ustaw odległość
   ostrzegawczą, np. `1.5 m`.

Klatka użyta przez `/from-latest` musi mieć najwyżej 3 sekundy. Endpoint
`/preview` raportuje jej wiek, rozdzielczość, wykryte i brakujące ID, jakość
geometrii oraz zgodność zapisanej kalibracji z aktualnym obrazem.

### Fizyczne ustawienie markerów

- Użyj czterech różnych markerów ArUco `DICT_4X4_50`. Strona kalibracji pozwala
  pobrać markery 10, 20, 30 i 40 jako PNG z białym marginesem i podpisem.
- Wszystkie cztery markery muszą być jednocześnie widoczne, leżeć płasko na tej
  samej płaszczyźnie gruntu i tworzyć prostokąt lub znany czworokąt.
- TL/TR/BR/BL oznacza położenie widoczne na obrazie kamery, nie kierunki świata.
- `width_m` to odległość centrum TL–TR (odpowiednio BR–BL), a `height_m`
  centrum TL–BL (odpowiednio TR–BR).
- Markery powinny być matowe, niezasłonięte, z pełnym białym marginesem.
  W pomieszczeniu zwykle sprawdzają się rozmiary 15–25 cm; z większej odległości
  należy użyć większego wydruku.
- Po kalibracji nie przesuwaj kamery ani statywu, nie zmieniaj obiektywu,
  orientacji telefonu, proporcji, cropu ani ustawienia markerów. Po każdej
  takiej zmianie wykonaj kalibrację ponownie.

Zaawansowana sekcja **Użyj kamery tego komputera** pozwala ręcznie wykonać
kalibrację z wybranego urządzenia USB. Nie uruchamia się automatycznie i nie
jest używana podczas kalibracji strumienia telefonu. Ma osobne pole
`camera_id` (domyślnie `cam_local_usb`), więc lokalna klatka nie może
nadpisać kalibracji wybranego telefonu.

### Model homografii i zgodność obrazu

Nowa kalibracja zapisuje wymiary klatki źródłowej, jej proporcje, środki
markerów, metadane jakości i homografię:

```text
znormalizowane współrzędne obrazu [u, v] → metry na płaszczyźnie gruntu
```

Dzięki normalizacji można zmienić rozdzielczość, np. z `960×720` na `1280×960`,
o ile proporcje i crop pozostają takie same. Różnica proporcji większa niż 1%
unieważnia użycie kalibracji dla bieżącej klatki. System zgłasza wtedy
`calibration_active=true`, `calibration_valid=false`, podaje
`calibration_warning` i wraca do jawnie opisanego pomiaru pikselowego.

Starsze rekordy bez wymiarów źródłowych pozostają możliwe do odczytu, lecz nie
są uznawane za bezpieczne do pomiaru metrycznego; wymagają ponownej kalibracji.

### Przykłady API kalibracji

Kalibracja ze świeżej klatki przyjmuje tablicę ID (obsługiwany jest również
zgodny wstecznie zapis tekstowy `"10,20,30,40"`):

```http
POST /api/calibration/cam_phone_1/from-latest
Content-Type: application/json

{
  "marker_ids": [10, 20, 30, 40],
  "width_m": 3.0,
  "height_m": 2.0
}
```

Praktyczny test kalibracji:

```http
POST /api/calibration/cam_phone_1/measure
Content-Type: application/json

{
  "point_a": [120, 620],
  "point_b": [760, 620],
  "frame_width": 960,
  "frame_height": 720
}
```

Odpowiedź zawiera współrzędne obu punktów na płaszczyźnie i `distance_m`.
Endpoint odrzuca pomiar, jeśli podane proporcje klatki nie pasują do
kalibracji.

### Diagnostyka

- `GET /api/cameras` — sprawdź, czy źródło faktycznie wysyła klatki i pod jakim
  `camera_id`.
- `GET /api/calibration/{camera_id}/preview` — sprawdź świeżość klatki, markery
  oraz zgodność formatu.
- `GET /api/calibration/{camera_id}` zwraca `404`, dopóki dla tego ID nie
  zapisano kalibracji. Samo `404` nie oznacza awarii kamery.
- Jeśli widzisz `cam_default`, mimo że oczekujesz telefonu, połącz najpierw
  telefon i otwórz stronę z jawnym `?camera_id=cam_phone_1`.

## Warstwy podglądu

Obraz LIVE jest przesyłany bez wrysowanych adnotacji. Panel nakłada osobno
BBOX-y, strefy, markery, odległości oraz pełny szkielet MediaPipe, dlatego każdą
warstwę można rzeczywiście włączyć i wyłączyć. Przycisk `POS` steruje wyłącznie
szkieletem MediaPipe — nie wyłącza analizy postury ani alarmów. Materiał
dowodowy alarmów nadal zawiera adnotacje.

Połączenie panelu i kalibracji używa subskrypcji konkretnej kamery:

```text
/ws/live?camera_id=cam_phone_1&binary=true
```

Dla każdej klatki WebSocket wysyła najpierw mały JSON z metadanymi i
nakładkami, a następnie binarną wiadomość z JPEG. Panel dekoduje ją jako `Blob`
i `createImageBitmap`, bez przesyłania obrazu jako Base64. Klient, który
potrzebuje samych wyników, może dodać `include_frame=false`.

## Ustawienia wydajności

Wartości startowe są celowo dobrane pod płynny strumień `960×720` przy
telefonie ustawionym na 5 FPS:

| Zmienna | Domyślnie | Znaczenie |
|---|---:|---|
| `YOLO_IMG_SIZE` | `512` | rozmiar wejścia YOLO; `640` pomaga przy małych, odległych osobach kosztem opóźnienia |
| `POSTURE_SAMPLE_FPS` | `3` | częstotliwość MediaPipe niezależna od FPS kamery |
| `POSTURE_MAX_POSES` | `1` | maksymalna liczba sylwetek analizowanych przez MediaPipe |
| `POSTURE_MIN_PERSON_HEIGHT_FRAC` | `0.25` | pomija bbox-y niższe niż 25% wysokości obrazu; to nie jest próg pewności landmarków |
| `WORKER_ID_SAMPLE_FPS` | `1.5` | częstotliwość dekodowania QR |
| `EVIDENCE_SAMPLE_FPS` | `3` | częstotliwość próbek w buforze klipu zdarzenia |

Wszystkie wartości można nadpisać w prywatnym `.env`. Repozytorium dostarcza
jedynie `.env.example` i nie modyfikuje istniejącego `.env`.

ArUco jest uruchamiane adaptacyjnie:

- bez aktywnej strefy markerowej i bez otwartej kalibracji — detektor nie
  wykonuje pracy;
- dla aktywnej strefy markerowej — maksymalnie 1 raz/s, a ostatni wynik jest
  przechowywany w cache;
- gdy strona kalibracji odpytuje `/preview` — 5 razy/s przez odnawianą,
  krótką dzierżawę; po zamknięciu strony tryb wygasa automatycznie.

Jeżeli wiele osób ma być jednocześnie analizowanych z bliska, zwiększ
`POSTURE_MAX_POSES`. Jeżeli ważniejsze są sylwetki odległe, zmniejsz
`POSTURE_MIN_PERSON_HEIGHT_FRAC` i rozważ `YOLO_IMG_SIZE=640`. Obie zmiany
zwiększają koszt CPU/GPU.

## Modele

```bash
python tools/download_models.py
```

Pobierane są:

- `models/pose_landmarker_lite.task`;
- `ppe.pt` — model demonstracyjny PPE.

`yolo11n.pt` pobiera Ultralytics przy pierwszym uruchomieniu. Domyślne COCO
rozpoznaje samochód, autobus i ciężarówkę. Koparka, walec, ładowarka i żuraw
wymagają własnego modelu oraz poprawnych `YOLO_HAZARD_CLASS_IDS`.

### NVIDIA: FP16 i opcjonalny TensorRT

Na CUDA ustaw np.:

```dotenv
YOLO_DEVICE=0
YOLO_IMG_SIZE=512
```

Detektor automatycznie włącza `half=True` na CUDA/TensorRT; pozostawia pełną
precyzję na CPU i Apple MPS. Największe przyspieszenie może dać silnik
TensorRT FP16 wyeksportowany przez Ultralytics:

```bash
yolo export model=yolo11n.pt format=engine device=0 imgsz=512 half=True
```

Następnie konfiguracja może bez zmiany kodu wskazywać plik `.engine`:

```dotenv
YOLO_MODEL=yolo11n.engine
YOLO_DEVICE=0
YOLO_IMG_SIZE=512
```

Silnik TensorRT należy budować i testować dla docelowej wersji CUDA/TensorRT
oraz docelowej klasy GPU. Zachowaj plik `.pt` jako przenośny fallback.
MediaPipe pozostaje obciążeniem CPU, dlatego jego osobny worker nadal ma
znaczenie również przy szybkim YOLO.

## Kontrola gotowości

```text
GET /api/readiness
```

Najważniejsze pola:

- `ready_for_demo` — działa podstawowy przepływ detekcja → alarm → dowód → review;
- `ready_for_metric_demo` — co najmniej jedna kamera jest jednocześnie online,
  ma aktywną strefę oraz poprawną kalibrację zgodną z bieżącymi proporcjami;
- `cameras` / `camera_readiness` — stan per `camera_id`, w tym
  `zones_configured`, `calibration_exists`, `calibration_valid`,
  `calibration_warning` i `metric_distance_ready`;
- `metric_ready_cameras` — lista kamer gotowych do pomiarów metrycznych;
- `warnings` — brakujące modele, strefy lub kalibracje.

Kalibracja lub strefa innej kamery nie spełnia warunku gotowości. Przykładowo
strefa `cam_phone_1` nigdy nie korzysta z kalibracji `cam_default`.

Przykładowy element `cameras`:

```json
{
  "camera_id": "cam_phone_1",
  "online": true,
  "zones_configured": true,
  "calibration_exists": true,
  "calibration_valid": true,
  "metric_distance_ready": true
}
```

## API

| Endpoint | Metoda | Zastosowanie |
|---|---|---|
| `/api/frame` | POST | klatka + `camera_id` + tryb; `async_processing=true` zwraca szybkie `202`, domyślnie zachowuje zgodny tryb synchroniczny |
| `/api/cameras` | GET | aktywne strumienie |
| `/api/readiness` | GET | checklist MVP/pilota |
| `/api/modes` | GET | dostępność modułów |
| `/api/zones` | GET/POST | strefy konkretnej kamery |
| `/api/alerts` | GET | historia z filtrami |
| `/api/alerts/{id}/review` | PATCH | decyzja operatora i notatka |
| `/api/reports/export.csv` | GET | pełny eksport audit trail |
| `/api/reports/training.jsonl` | GET | potwierdzone/odrzucone etykiety |
| `/api/workers` | GET/POST | lista i utworzenie pracownika |
| `/api/workers/{worker_id}` | GET/PUT/DELETE | odczyt, zmiana lub usunięcie pracownika |
| `/api/worker-qr` | GET | SVG identyfikatora pracownika |
| `/api/calibration/{camera_id}` | GET/POST/DELETE | stan, zapis lub usunięcie kalibracji |
| `/api/calibration/{camera_id}/from-latest` | POST | preferowana kalibracja z ostatniej surowej klatki |
| `/api/calibration/{camera_id}/preview` | GET | świeżość klatki, markery, jakość i zgodność kalibracji |
| `/api/calibration/{camera_id}/measure` | POST | test odległości pomiędzy dwoma punktami obrazu |
| `/api/calibration/{camera_id}/latest-frame` | GET | zgodny wstecznie odczyt ostatniej surowej klatki |
| `/api/calibration/markers/{marker_id}.png` | GET | marker ArUco `DICT_4X4_50` do druku |
| `/ws/live` | WebSocket | wyniki na żywo |

## Docker

```bash
cp .env.example .env
docker compose up --build
```

Dane są utrzymywane w wolumenie `/app/data`, w tym baza pracowników, alarmy,
strefy, kalibracje, miniatury i klipy.

## Testy

```bash
pytest -q
```

Testy obejmują również magazyn klatek, kalibrację `from-latest`, walidację
geometrii, zmianę rozdzielczości i proporcji, osobne kamery, endpoint pomiaru,
readiness per kamera oraz kontrakt strony kalibracji. Aktualny wynik:
**199 passed**.

## Granice MVP

Nie zaimplementowano jako wiarygodnych funkcji produkcyjnych:

- diagnozy „pijany” lub rozpoznania substancji;
- dokładnego rozpoznawania maszyn bez własnego modelu danych budowlanych;
- pełnej rekonstrukcji stereo 3D;
- detekcji papierosa jako małego obiektu;
- analizy IR/termowizyjnej;
- automatycznej decyzji kadrowej lub karnej.

Kalibracja jest modelem **2.5D jednej płaszczyzny**, a nie pomiarem 3D.
Odległości są wiarygodne tylko dla punktów rzutowanych na skalibrowaną
płaszczyznę gruntu. Wynik może być błędny dla rusztowania, drabiny, innej
kondygnacji, wykopu, pochyłego terenu i obiektów położonych wysoko nad ziemią.
Punkt osoby jest przybliżany dolnym środkiem bboxu, a odległość od maszyny —
odległością od tego punktu do dolnej krawędzi bboxu maszyny na płaszczyźnie.

Szczegółowy opis zmian i scenariusz pilota znajdują się w
[`MVP_READINESS.md`](MVP_READINESS.md).

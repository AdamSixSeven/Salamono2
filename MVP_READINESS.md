# Perimetr — zakres „realnego MVP” i gotowość pilota

## Cel

Ta wersja zamienia techniczne demo w zamknięty przepływ operacyjny:

```text
konfiguracja kamery i strefy
→ detekcja i ostrzeżenie
→ zapis dowodu
→ reakcja operatora
→ historia i eksport
→ etykieta do dalszego uczenia
```

Zakres został dobrany do przypadków omawianych w transkrypcie: PPE, statyczne
i ruchome strefy, odległość od maszyny, nietypowy ruch/upadek, identyfikator
pracownika, komunikat na miejscu oraz materiał do późniejszej analizy.

## Nowe lub dopracowane funkcje

### 1. Odległość od statycznej strefy

Każda strefa polygonowa ma teraz margines ostrzegawczy:

- na zewnątrz i w marginesie: `zone_approach`, `WARNING`;
- na granicy lub wewnątrz: `zone_breach`, poziom strefy;
- poza marginesem: brak alarmu.

Odległość jest signed:

```text
+1.20 m  — przed strefą
 0.00 m  — granica
-0.45 m  — wewnątrz strefy
```

Po kalibracji wynik jest metryczny. Bez niej UI jawnie pokazuje piksele.

### 2. Dynamiczna strefa maszyny

Dla wykrytego obiektu zagrożenia tworzony jest bufor `WARNING/DANGER`, który
porusza się razem z bboxem. Odległość jest liczona od przybliżonego punktu
stóp osoby do dolnej krawędzi maszyny na płaszczyźnie gruntu.

### 3. Materiał zdarzenia

`backend/evidence.py` utrzymuje osobny ring buffer dla każdej kamery. Po
potwierdzeniu reguły zapisuje MP4 z klatkami przed alarmem i, jeśli strumień
trwa dalej, uzupełnia go o część po alarmie. Miniatura i klip są powiązane z
jednym rekordem zdarzenia.

### 4. Obsługa przez operatora

W `/review.html` zdarzenie może zostać oznaczone jako:

- `acknowledged` — operator odebrał alarm;
- `confirmed` — zdarzenie prawdziwe;
- `false_positive` — fałszywy alarm;
- `escalated` — przekazane dalej.

Można dodać operatora i notatkę. Przycisk **Potwierdź alarm** w panelu LIVE
ustawia `acknowledged`, zamiast tylko chować baner.

### 5. Dane do dalszego uczenia

```text
GET /api/reports/training.jsonl?review_status=confirmed
GET /api/reports/training.jsonl?review_status=false_positive
```

Eksport zawiera sygnały, bbox, odległości, kamerę i decyzję operatora. Nie jest
to jeszcze automatyczny pipeline treningowy, ale usuwa najważniejszą barierę:
brak uporządkowanych etykiet z pilota.

### 6. Wiele kamer

`camera_id` rozdziela:

- stream i WebSocket;
- strefy;
- kalibrację;
- bufor dowodowy;
- historię i raporty.

Panel ma selektor kamer. **Dodaj kamerę** tworzy ID i od razu pokazuje QR do
połączenia telefonu.

### 7. Film jako źródło demo

Przycisk **Wczytaj film** analizuje lokalny plik bez osobnego skryptu. Film
jest próbkowany do około 5 FPS i trafia do normalnego `/api/frame`, dzięki
czemu wszystkie alarmy, klipy i raporty działają identycznie jak dla telefonu.

### 8. Identyfikacja bez twarzy

QR `worker:<ID>` jest dopasowywany do bbox osoby i krótko cache'owany przy
zasłonięciu. Przy bramce można wymagać widocznego ID. Na placu opcja jest
domyślnie wyłączona, ponieważ plecy, odzież i zasłonięcia powodowałyby wiele
fałszywych alarmów.

### 9. Analiza postury

MediaPipe pracuje czasowo, a nie na pojedynczej klatce. Raportuje:

- kołysanie i niestabilny tor;
- nieregularność kroku;
- możliwy upadek/osunięcie;
- kandydat gestu ręka–usta.

Wynik zawsze ma interpretację `requires_human_verification`.

### 10. Kalibracja właściwego źródła obrazu

Kalibracja nie uruchamia już domyślnie kamery komputera. Dla aktywnego
`camera_id` backend zachowuje ostatnią surową klatkę przed naniesieniem bboxów
i innych warstw, a `/calibrate.html`:

- pobiera listę kamer z `GET /api/cameras`;
- wybiera ID z URL, `localStorage` albo pierwszej aktywnej kamery;
- pokazuje obraz tej kamery z `/ws/live` bez wymuszania proporcji 16:9;
- odpytuje `/api/calibration/{camera_id}/preview` o świeżość, wykryte markery
  i jakość geometrii;
- zapisuje kalibrację przez
  `POST /api/calibration/{camera_id}/from-latest`;
- pozwala sprawdzić znany dystans przez
  `POST /api/calibration/{camera_id}/measure`.

Ostatnia klatka jest przechowywana osobno dla każdego `camera_id`, w ograniczonym
i bezpiecznym wątkowo magazynie. Do wykonania kalibracji musi być świeższa niż
3 sekundy. Opcjonalna kamera USB znajduje się w sekcji zaawansowanej i wymaga
jawnego uruchomienia przez użytkownika. Korzysta z osobnego `camera_id`, dzięki
czemu lokalny upload nie nadpisuje kalibracji telefonu.

Homografia mapuje znormalizowane współrzędne obrazu `[u, v]` na metry. Zmiana
samej rozdzielczości przy zachowaniu proporcji jest obsługiwana, natomiast
względna różnica proporcji większa niż 1% wyłącza metry i zgłasza ostrzeżenie.
Każdy rekord zawiera rozdzielczość źródłową, proporcje, środki markerów i raport
jakości. Stare rekordy bez tych danych wymagają ponownej kalibracji.

Przed zapisem sprawdzane są m.in. cztery różne ID, komplet rogów markerów,
wypukłość i kolejność TL/TR/BR/BL, pole i kształt czworokąta, rozmiar markerów
oraz odwracalność i uwarunkowanie macierzy.

## Pliki dodane

- `backend/evidence.py`;
- `tools/download_models.py`;
- `MVP_READINESS.md`;
- nowe testy workflow, dowodów, stref, QR i multi-camera.

## Najważniejsze pliki zmienione

- `backend/routes/ingest.py` — pełna orkiestracja pipeline'u;
- `backend/zone_rules.py` — signed distance i podejście do strefy;
- `backend/worker_identification.py` — QR i brak identyfikatora;
- `backend/alert_storage.py`, `backend/routes/alerts.py` — review;
- `backend/routes/reports.py` — CSV i etykiety JSONL;
- `backend/main.py` — readiness, kamera, evidence;
- `backend/latest_frame_store.py` — ostatnia surowa klatka per `camera_id`;
- `backend/calibration.py`, `backend/routes/calibration.py` — znormalizowana
  homografia, jakość, podgląd, kalibracja z aktywnego strumienia i test pomiaru;
- `frontend/calibrate.html` — kalibracja aktywnej kamery krok po kroku;
- `frontend/index.html`, `frontend/app.js` — kamera, upload filmu, ACK;
- `frontend/review.html`, `frontend/review.js` — obsługa incydentu;
- `config.py`, `.env.example`, Docker — konfiguracja i trwałość danych.

## Scenariusz pilota demonstracyjnego

1. Uruchom system i sprawdź `/api/readiness`.
2. Uruchom telefon jako np. `cam_phone_1` i sprawdź jego obecność w
   `/api/cameras`.
3. Otwórz `/calibrate.html?camera_id=cam_phone_1`.
4. Rozłóż na jednej płaszczyźnie markery 10 TL, 20 TR, 30 BR i 40 BL. Zmierz
   szerokość i wysokość **od centrum do centrum**, poczekaj na status `4/4`
   i wykonaj kalibrację.
5. Testem dwóch punktów sprawdź odcinek o znanej długości.
6. Narysuj dla `cam_phone_1` strefę „Wykop A” z ostrzeżeniem 1,5 m.
7. Wygeneruj QR `W-001` i umieść go na kamizelce.
8. Nagraj osobno:
   - podejście i wejście do strefy;
   - podejście do pojazdu;
   - brak PPE na bramce;
   - kontrolowane zachwianie i bezpiecznie upozorowane osunięcie.
9. Wczytaj film w panelu albo wyślij go symulatorem.
10. Potwierdź odbiór alarmu w LIVE.
11. W Historii obejrzyj JPEG/MP4, ustaw `confirmed` lub `false_positive`
    i dodaj notatkę.
12. Wyeksportuj CSV oraz JSONL.

Nie należy przeprowadzać testu przez spożywanie alkoholu ani faktyczne
narażanie pracownika. Zachowania powinny być bezpiecznie inscenizowane.

## Kryteria odbioru MVP

- alarm strefy pojawia się po wymaganej liczbie klatek, nie po pojedynczym błędzie;
- `WARNING` występuje przed przekroczeniem granicy;
- kalibracja metryczna jest przypisana do właściwej kamery;
- `ready_for_metric_demo` jest prawdziwe tylko wtedy, gdy ta sama aktywna
  kamera ma strefę i kalibrację zgodną z aktualnymi proporcjami obrazu;
- zmiana orientacji, obiektywu lub cropu wyłącza niezgodną kalibrację i
  pozostawia jawny fallback pikselowy;
- zdarzenie ma miniaturę, klip i unikalne ID;
- potwierdzenie w LIVE zmienia status rekordu;
- historia filtruje po kamerze, typie, poziomie, pracowniku i review;
- restart kontenera nie usuwa danych z wolumenu;
- brak MediaPipe/PPE nie zatrzymuje całego backendu i jest pokazany w readiness;
- system nie używa rozpoznawania twarzy i nie zwraca diagnozy nietrzeźwości.

## Co nadal wymaga pilota i danych

1. Progi postury muszą zostać dostrojone na nagraniach z docelowej kamery.
2. Model maszyn budowlanych wymaga własnych klas i materiału z budów.
3. Homografia jest modelem 2.5D i działa wyłącznie dla punktów znajdujących się
   na jednej skalibrowanej płaszczyźnie gruntu. Nie jest rekonstrukcją 3D.
   Wynik może być błędny dla rusztowania, drabiny, innej kondygnacji, wykopu,
   pochyłego terenu lub obiektu wysoko nad ziemią.
4. QR może być niewidoczny; site-wide enforcement pozostaje opt-in.
5. Gest ręka–usta nie odróżnia papierosa od radia, picia lub dotknięcia twarzy.
6. Skuteczność należy raportować jako false alerts/person-hour, event recall i
   opóźnienie alarmu, a nie samą accuracy.

## Weryfikacja techniczna

```powershell
pytest -q
python -m compileall backend
node --check frontend/app.js
node --check frontend/zones.js
node --check frontend/review.js
```

Testy obejmują kalibrację ze świeżej klatki, odrzucenie klatki starej,
walidację geometrii, projekcję przy innej rozdzielczości i tych samych
proporcjach, odrzucenie niezgodnych proporcji, osobne kamery, znany dystans,
readiness per kamera oraz zachowanie frontendu. Ostatni pełny przebieg:
`199 passed`.

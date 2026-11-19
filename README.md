# Laboratorium 2A — Klasyczna detekcja znaków drogowych i sygnalizatorów (BDD100K)

Projekt realizuje klasyczny (bez sieci neuronowych) system detekcji obiektów klas `traffic_sign` oraz `traffic_light` wraz z rozpoznawaniem koloru światła (`red`, `yellow`, `green`, `unknown`) na zbiorze **BDD100K**.

> **Organizacja plików:** katalog `ai/` pełni rolę brudnopisu eksperymentalnego (jest ignorowany w `.gitignore`). Zaakceptowane notebooki docelowo trafiają bezpośrednio do głównego katalogu `lab4/`. Każdy z trzech notebooków jest **samowystarczalny** (niezbędne funkcje pomocnicze są wbudowane bezpośrednio w komórkach notebooka, bez importowania luźnych plików `.py`).

---

## Architektura i działanie kodu

### 1. Warstwa bazowa, dane i ewaluacja (`classical.py`)
Moduł odpowiada za wczytywanie danych BDD100K, detektor bazowy (HSV + kontury) oraz spójną ewaluację:
- **Podział i manifest (`make_manifest`, `annotations`, `inventory`)**:
  - Zachowuje oryginalny podział BDD100K (`train` / `val` / `test`), nie mieszając obrazów między splitami.
  - Zapisuje `manifest.json` (domyślnie `1050` train / `150` val / `300` test, `seed=42`), dzięki czemu wszystkie metody pracują na identycznym podzbiorze.
  - Filtruje obiekty do klas docelowych (`traffic_sign`, `traffic_light`) i przypisuje kolor światła (`red`, `yellow`, `green`, `unknown`).
- **Detektor bazowy HSV + kontury (`masks`, `contours`, `detect`)**:
  - Konwertuje obraz z BGR do HSV i buduje maski kolorów (`red`, `yellow`, `green`, `blue`) z progowaniem nasycenia (`saturation`) i jasności (`value`), a także maskę jasnych tablic (`white`).
  - Dla **znaków drogowych (`traffic_sign`)**: wyznacza kontury po morfologicznym domknięciu (`MORPH_CLOSE`), filtruje po powierzchni, proporcjach (`0.35 <= bw/bh <= 3.5`), wypełnieniu (`fill >= 0.4`), wypukłości (`solidity >= 0.75`) oraz gęstości krawędzi Canny'ego.
  - Dla **sygnalizatorów (`traffic_light`)**: szuka ciemnych, zwartych obudów (`gray < 65`) o pionowych lub poziomych proporcjach, a dodatkowo generuje hipotezy wokół małych, silnie nasyconych punktów świetlnych (rozszerzenie w dół/górę w poszukiwaniu obudowy).
  - Łączy kandydatów i wykonuje **NMS** (`nms`, `iou`).
- **Klasyfikator koloru światła (`classify_color`)**:
  - Dla wycinka sygnalizatora liczy największe pola konturów w maskach `red`, `yellow`, `green`.
  - Zwraca dominujący kolor tylko wtedy, gdy przekracza minimalny próg pola oraz ma co najmniej `1.4×` przewagi nad drugim kolorem; w przeciwnym razie zwraca `unknown`.
- **Ewaluacja (`match`, `evaluate`)**:
  - Dopasowuje predykcje do Ground Truth zachłannie po `score` przy progu **IoU ≥ 0.5** (niezależnie dla `traffic_sign` i `traffic_light`).
  - Liczy **Precision**, **Recall**, **F1** oraz macierz pomyłek koloru wyłącznie dla prawidłowo zlokalizowanych sygnalizatorów.
  - Zbiera listę błędów (`FP` / `FN`) do analizy jakościowej.

### 2. Eksploracja danych (`data_exploration.py`)
Moduł wspomagający analizę zbioru treningowego (Zadanie 1):
- `summarize_train`: zbiera statystyki liczby obiektów, kolorów świateł, obiektów małych/zasłoniętych/uciętych, warunków pogodowych i pory dnia oraz wykrywa brakujące pliki lub puste adnotacje.
- `select_examples`, `draw_all`, `gallery`, `crop_gallery`: buduje galerie obrazów (w tym trudniejsze warunki, np. noc/deszcz) oraz wycinki obiektów docelowych.
- Dodatkowo `generate_label_heatmaps.py` generuje mapy gęstości położeń środków ramek na całym oficjalnym `train`.

### 3. Propozycje ramek, ekstrakcja cech i Random Forest (`feature_detector.py`)
Rozszerzenie detekcji o klasyfikator uczenia maszynowego na ręcznie zaprojektowanych cechach:
- **Generator propozycji (`proposals`)**:
  - Łączy do `60` hipotez z detektora HSV/obudów (z obniżonymi progami) oraz do `40` prostokątów z konturów krawędzi Canny'ego, deduplikując ramki przy `IoU >= 0.9` (limit `100` propozycji na obraz).
- **Wektor cech (`features`, łącznie 387 cech na wycinek `32×32`)**:
  - `geometry` (6): znormalizowana szerokość, wysokość, proporcje, pole, środek `(x, y)`.
  - `color` (34): histogramy kanałów H (12), S (8), V (8) oraz frakcje pikseli `red`, `yellow`, `green`, `blue`, `white`, `dark`.
  - `texture` (12): histogram Uniform LBP (10 binów), odchylenie standardowe jasności, wariancja Laplasjanu.
  - `edges` (11): 9-binowy histogram orientacji gradientów Sobela ważony magnitudą, gęstość Canny'ego, średnia magnituda.
  - `hog` (324): deskryptor HOG na siatce `3×3` bloków po `2×2` komórki (`9` orientacji).
- **Budowa zbioru i trening (`feature_table`, `train_forest`)**:
  - Dla `train`: łączy ramki GT, ich losowy jitter (`±10%`), pozytywne propozycje oraz losowe/kandydackie wycinki tła (`IoU <= 0.2`); odrzuca przypadki niejednoznaczne (`0.2 < IoU < 0.5`). Wyniki są cache'owane w `.npz`/`.json`.
  - Dla `val`: używa **wyłącznie rzeczywistych propozycji** z `proposals()` (bez przecieku GT).
  - Trenuje `RandomForestClassifier` (`n_estimators=100`, `max_depth=18`, `class_weight='balanced_subsample'`).
- **Analiza i detekcja (`compare_groups`, `permutation_groups`, `rf_detect`)**:
  - Porównuje 12 wariantów zestawów cech (ablacja grup) oraz ważność permutacyjną na `val`.
  - `rf_detect` klasyfikuje propozycje wybranym modelem i progiem prawdopodobieństwa, nakłada NMS, a dla wykrytych `traffic_light` uruchamia `classify_color`.

---

## Notebooki i kolejność prac

1. **`01_dane_i_wyjasnienie.ipynb`** — utworzenie/wczytanie manifestu (`1050/150/300`), statystyki `train`, galerie z trudnymi warunkami i wycinki obiektów.
2. **`02_detekcja_i_ocena.ipynb`** — detektor bazowy HSV + kontury + geometria: dobór parametrów na `val`, ocena na `test`, macierz pomyłek kolorów oraz analiza przykładów `FP` i `FN`.
3. **`03_cechy_i_las_losowy.ipynb`** — generator propozycji, ekstrakcja 387 cech, porównanie 12 wariantów i ważności permutacyjnej na `val`, dobór progu oraz końcowa ocena na `test`.

---

## Uruchomienie środowiska

W katalogu `lab4/`:

```powershell
# Jeśli .venv nie istnieje:
uv venv --python 3.13
uv pip install -r ai/requirements.txt

# Uruchomienie JupyterLab:
.\.venv\Scripts\python.exe -m jupyterlab
```

Skrypty pomocnicze do weryfikacji technicznej (smoke-testy):
```powershell
.\.venv\Scripts\python.exe ai/verify_solution.py
.\.venv\Scripts\python.exe ai/verify_features.py
```

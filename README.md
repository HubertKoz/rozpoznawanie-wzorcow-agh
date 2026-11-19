# Laboratorium 2A: znaki i sygnalizatory

1. `1_data.ipynb` — podział i analiza adnotacji.
2. `2_detector.ipynb` — propozycje ramek, cechy, Random Forest i ocena.

Funkcje znajdują się w `utils_common.py`, `utils_data.py` i `utils_detector.py`. Uruchamiaj notebooki z katalogu `lab4`, w środowisku z OpenCV, NumPy, pandas, matplotlib, Pillow, scikit-learn i Jupyter. Dane BDD100K powinny znajdować się w katalogach `bdd100k_images_100k` i `bdd100k_labels`.

Domyślnie używamy 1500 obrazów z oficjalnych podziałów: 1050 train, 150 val i 300 test. Manifest jest w `outputs/official_1500_seed42/manifest.json`. Próg dobieramy na val, a na teście obliczamy precision, recall i F1 przy IoU ≥ 0,5 osobno dla znaków i świateł. Kolor oceniamy dla poprawnie zlokalizowanych świateł. Predykcje zapisywane są do `simple_rf_predictions.json` w katalogu eksperymentu.

Cechy opisują geometrię, rozkład HSV, teksturę LBP i krawędzie. Propozycje wykorzystują kolor, kontury i ich geometrię, w tym wypełnienie oraz wypukłość, zgodnie z tematyką wykładu 5. Random Forest klasyfikuje wycinki jako znak, światło lub tło. Wynik zawiera klasę, ramkę, score i kolor; unknown oznacza kolor nieokreślony.

Wcześniejsze wyniki w `outputs` dotyczą poprzedniego wariantu cech i nie są wynikami uproszczonego modelu. Przed sformułowaniem wniosków uruchom notebooki i przeanalizuj wyniki oraz błędy.

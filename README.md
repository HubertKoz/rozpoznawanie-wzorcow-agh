# lab5: srodowisko

Wlasne srodowisko lab5 (bez lab4 i glownego .venv). Z folderu `lab5/`:

    uv sync                      # tworzy lab5/.venv wg pyproject.toml (torch CUDA 12.8)
    .venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available())"

Notebooki: otworz `1_trening.ipynb` / `2_ewaluacja.ipynb` i wybierz interpreter `lab5/.venv`.
Z terminala: `.venv\Scripts\python.exe -m ipykernel install --user --name lab5`.

Dane (zdjecia, etykiety) sa czytane z `../lab4/` (tylko sciezki). Hardlinki w `yolo_data/` wymagaja
tego samego dysku NTFS. `yolov8n.pt` lezy w `lab5/` (po usunieciu ultralytics pobierze go sam przy `YOLO("yolov8n.pt")`).

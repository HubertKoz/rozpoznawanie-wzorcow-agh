# funkcje YOLO: konwersja danych, trening, predykcje (COCO, hybryda, fine-tuned), czas
import os
import time

import numpy as np

import utils_common as w

NAMES = ["traffic_sign", "traffic_light_red", "traffic_light_yellow", "traffic_light_green", "traffic_light_unknown"]
COCO = {9: "traffic_light", 11: "traffic_sign"}  # w COCO jest tylko swiatlo i znak STOP
DATA = w.LAB5 / "yolo_data"
RUN = w.LAB5 / "runs" / "yolov8n_5kl"


#### KONWERSJA I TRENING ####
def class_id(obj):
    return 0 if obj["class"] == "traffic_sign" else 1 + w.COLORS.index(obj["color"])


def convert():
    # etykiety w formacie YOLO powstaja w lab5, zdjecia to hardlinki do lab4 (zero kopii na dysku)
    manifest = w.make_manifest()
    for split, records in manifest["splits"].items():
        (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in records:
            src = w.DATA_ROOT / r["image"]
            dst = DATA / "images" / split / (r["id"] + ".jpg")
            if not dst.exists():
                os.link(src, dst)
            image = w.load_image(src)
            h, wd = image.shape[:2]
            lines = []
            for o in w.annotations(w.DATA_ROOT / r["label"], image.shape):
                x1, y1, x2, y2 = o["box"]
                lines.append(f"{class_id(o)} {(x1 + x2) / 2 / wd:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / wd:.6f} {(y2 - y1) / h:.6f}")
            (DATA / "labels" / split / (r["id"] + ".txt")).write_text("\n".join(lines))
    yaml = f"path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
    yaml += "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    (DATA / "data.yaml").write_text(yaml, encoding="utf-8")


def train(epochs=60, imgsz=1280, batch=8, name="yolov8n_5kl", fraction=1.0, project=None):
    from ultralytics import YOLO
    model = YOLO(str(w.LAB5 / "yolov8n.pt"))
    # best.pt wybierany po walidacji; test nie bierze udzialu w treningu
    return model.train(data=str(DATA / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch,
                       patience=15, seed=42, workers=4, fraction=fraction, device=0,
                       project=str(project or w.LAB5 / "runs"), name=name, exist_ok=True)


#### COCO I HYBRYDA ####
def raw_predict(model, records, imgsz, conf=.05, device=0):
    # surowe detekcje przy niskim progu, prog dobieramy potem bez ponownego liczenia
    raw = {}
    for r in records:
        image = w.load_image(w.DATA_ROOT / r["image"])
        res = model.predict(image, imgsz=imgsz, conf=conf, classes=list(COCO), device=device, verbose=False)[0]
        raw[r["id"]] = [{"cls": int(c), "box": b.tolist(), "score": float(s)}
                        for c, b, s in zip(res.boxes.cls, res.boxes.xyxy, res.boxes.conf)]
    return raw


def to_items(raw, conf, color_fn=None):
    out = {}
    for rid, dets in raw.items():
        out[rid] = []
        for d in dets:
            if d["score"] < conf:
                continue
            item = {"class": COCO[d["cls"]], "box": d["box"], "score": d["score"], "color": None}
            if item["class"] == "traffic_light":
                item["color"] = color_fn(rid, d["box"]) if color_fn else "unknown"
            out[rid].append(item)
    return out


def color_fn_for(records, sat, value):
    cache = {}

    def fn(rid, box):
        if rid not in cache:
            r = next(r for r in records if r["id"] == rid)
            cache[rid] = w.load_image(w.DATA_ROOT / r["image"])
        h, wd = cache[rid].shape[:2]
        x1, y1, x2, y2 = np.clip(box, 0, [wd, h, wd, h]).astype(int)
        return w.classify_color(cache[rid][y1:y2, x1:x2], sat, value)
    return fn


def hybrid_detect(model, image, p, device=0):
    # YOLO COCO lokalizuje, kolor swiatla z HSV w ROI
    res = model.predict(image, imgsz=p["imgsz"], conf=p["conf"], classes=list(COCO), device=device, verbose=False)[0]
    out = []
    for c, b, s in zip(res.boxes.cls, res.boxes.xyxy, res.boxes.conf):
        item = {"class": COCO[int(c)], "box": b.tolist(), "score": float(s), "color": None}
        if item["class"] == "traffic_light":
            x1, y1, x2, y2 = np.clip(b.cpu().numpy(), 0, [image.shape[1], image.shape[0]] * 2).astype(int)
            item["color"] = w.classify_color(image[y1:y2, x1:x2], p["sat"], p["value"])
        out.append(item)
    return out


#### FINE-TUNED ####
def ft_predict(model, image, imgsz=1280, conf=.01, device=0):
    # agnostic_nms, zeby jedno swiatlo nie dalo kilku kolorow
    res = model.predict(image, imgsz=imgsz, conf=conf, agnostic_nms=True, device=device, verbose=False)[0]
    return [{"cls": int(c), "box": b.tolist(), "score": float(s)}
            for c, b, s in zip(res.boxes.cls, res.boxes.xyxy, res.boxes.conf)]


def ft_raw(model, records, imgsz=1280, conf=.01):
    return {r["id"]: ft_predict(model, w.load_image(w.DATA_ROOT / r["image"]), imgsz, conf) for r in records}


def ft_items(raw, conf):
    out = {}
    for rid, dets in raw.items():
        out[rid] = []
        for d in dets:
            if d["score"] < conf:
                continue
            name = NAMES[d["cls"]]
            light = name != "traffic_sign"
            out[rid].append({"class": "traffic_light" if light else name, "box": d["box"], "score": d["score"],
                             "color": name.replace("traffic_light_", "") if light else None})
    return out


#### CZAS ####
def time_fn(fn, images, n=100):
    fn(images[0])  # rozgrzewka
    t0 = time.perf_counter()
    for im in images[:n]:
        fn(im)
    return n / (time.perf_counter() - t0)


#### BLEDY ####
def show_errors(res, items, images, gt, name, kind, cls, n=6):
    import matplotlib.pyplot as plt
    ids = list(dict.fromkeys(e["id"] for e in res["errors"] if e["type"] == kind and e["class"] == cls))[:n]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, rid in zip(axes.ravel(), ids):
        w.draw(images[rid], items[rid], ax, f"{kind} {cls} {rid}")
        for g in gt[rid]:
            if g["class"] == cls:
                x1, y1, x2, y2 = g["box"]
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1, linestyle="--"))
    for ax in axes.ravel()[len(ids):]:
        ax.axis("off")
    fig.suptitle(f"{name}: {kind} ({cls}); zielone/niebieskie = predykcja, czerwone przerywane = GT")
    fig.tight_layout()
    fig.savefig(w.LAB5 / "wykresy" / f"{name}_{kind}_{cls}.png", dpi=100)
    return fig

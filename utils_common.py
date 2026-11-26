# wspolne funkcje lab5: dane BDD100K (z lab4/, tylko sciezka), manifest, ocena, kolor HSV
from pathlib import Path
from collections import Counter
import json
import time

import cv2
import numpy as np
import pandas as pd

#### SCIEZKI ####
LAB5 = Path(__file__).resolve().parent
DATA_ROOT = LAB5.parent / "lab4"  # zdjecia i etykiety zostaja w lab4, nie kopiujemy ich
OUT = LAB5 / "outputs"

CLASSES = ("traffic_sign", "traffic_light")
COLORS = ("red", "yellow", "green", "unknown")


#### WCZYTYWANIE ####
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_image(path):
    # imdecode, bo cv2.imread nie lubi polskich znakow w sciezce
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def annotations(path, shape):
    data = read_json(path)
    frames = data.get("frames", [])
    objects = frames[0].get("objects", []) if frames else data.get("labels", [])
    h, w = shape[:2]
    result = []
    for obj in objects:
        category = obj.get("category", "").replace(" ", "_")
        box = obj.get("box2d")
        if category not in CLASSES or not box:
            continue
        x1, y1, x2, y2 = np.clip([box["x1"], box["y1"], box["x2"], box["y2"]], 0, [w, h, w, h]).astype(float)
        if x2 <= x1 or y2 <= y1:
            continue
        color = obj.get("attributes", {}).get("trafficLightColor", "unknown")
        if color not in COLORS:
            color = "unknown"
        result.append({"class": category, "box": [x1, y1, x2, y2],
                       "color": color if category == "traffic_light" else None})
    return result


#### PODZIAL ####
def inventory(split):
    images = {}
    for folder in sorted((DATA_ROOT / "bdd100k_images_100k").glob(f"**/{split}")):
        for path in folder.glob("*.jpg"):
            images[path.stem] = path
    pairs = []
    for label in sorted((DATA_ROOT / "bdd100k_labels").glob(f"**/{split}/*.json")):
        image = images.get(label.stem)
        if image is not None and image.stat().st_size > 0 and label.stat().st_size > 0:
            pairs.append({"id": label.stem, "image": image.relative_to(DATA_ROOT).as_posix(),
                          "label": label.relative_to(DATA_ROOT).as_posix()})
    return pairs


def make_manifest(n=1500, seed=42):
    # ten sam kod i seed co w lab4, wiec dostajemy ten sam podzial (sprawdzane w cw5)
    path = OUT / "manifest.json"
    if path.exists():
        return read_json(path)
    sizes = {"train": int(0.7 * n), "val": int(0.1 * n)}
    sizes["test"] = n - sum(sizes.values())
    rng = np.random.default_rng(seed)
    splits = {}
    for split, size in sizes.items():
        pairs = inventory(split)
        splits[split] = [pairs[i] for i in rng.permutation(len(pairs))[:size]]
    manifest = {"seed": seed, "n": n, "splits": splits}
    save_json(path, manifest)
    return manifest


#### OCENA ####
def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    inter = float(np.prod(wh))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter)
    return inter / union if union > 0 else 0.


def match(pred, gt, category, threshold=.5):
    # zachlannie po score, kazdy GT najwyzej raz, kolor nie wplywa na dopasowanie
    order = sorted([i for i, p in enumerate(pred) if p["class"] == category], key=lambda i: -pred[i]["score"])
    remaining = {i for i, g in enumerate(gt) if g["class"] == category}
    matches, fp = [], []
    for i in order:
        j = max(remaining, key=lambda j: (iou(pred[i]["box"], gt[j]["box"]), -j), default=None)
        if j is not None and iou(pred[i]["box"], gt[j]["box"]) >= threshold:
            matches.append((i, j))
            remaining.remove(j)
        else:
            fp.append(i)
    return matches, fp, sorted(remaining)


def load_gt(records):
    gt = {}
    for r in records:
        image = load_image(DATA_ROOT / r["image"])
        gt[r["id"]] = annotations(DATA_ROOT / r["label"], image.shape)
    return gt


def evaluate(records, predictions, gt):
    counts = {c: Counter(TP=0, FP=0, FN=0) for c in CLASSES}
    pairs, errors = [], []
    for r in records:
        pred, truth = predictions[r["id"]], gt[r["id"]]
        for category in CLASSES:
            matched, fp, fn = match(pred, truth, category)
            counts[category].update(TP=len(matched), FP=len(fp), FN=len(fn))
            if category == "traffic_light":
                pairs += [(truth[j]["color"], pred[i]["color"], r["id"], pred[i]["box"]) for i, j in matched]
            errors += [{"id": r["id"], "type": "FP", "class": category, "box": pred[i]["box"]} for i in fp]
            errors += [{"id": r["id"], "type": "FN", "class": category, "box": truth[j]["box"]} for j in fn]
    rows = []
    for category, c in counts.items():
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        rows.append({"class": category, "TP": tp, "FP": fp, "FN": fn,
                     "precision": tp / (tp + fp) if tp + fp else 0.,
                     "recall": tp / (tp + fn) if tp + fn else 0.,
                     "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.})
    confusion = pd.DataFrame(0, index=COLORS, columns=COLORS)
    for actual, predicted, _, _ in pairs:
        confusion.loc[actual, predicted] += 1
    acc = sum(a == p for a, p, _, _ in pairs) / len(pairs) if pairs else None
    return {"metrics": pd.DataFrame(rows), "confusion": confusion, "errors": errors,
            "color_pairs": pairs, "color_acc": acc, "matched_lights": len(pairs)}


#### KOLOR HSV ####
# dokladnie ta sama metoda koloru co w lab4 (classify_color), zeby porownanie bylo uczciwe
def color_masks(image, saturation, value):
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    valid = (s >= saturation) & (v >= value)
    # w OpenCV H ma zakres 0..179, czerwony przechodzi przez koniec skali
    return {"red": valid & ((h <= 12) | (h >= 165)),
            "yellow": valid & (h >= 15) & (h <= 38),
            "green": valid & (h >= 40) & (h <= 95)}


def classify_color(crop, saturation=100, value=65):
    if crop.size == 0:
        return "unknown"
    scores = {}
    for color, mask in color_masks(crop, saturation, value).items():
        found = cv2.findContours(np.uint8(mask) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        scores[color] = max([cv2.contourArea(c) for c in found], default=0.)
    ranking = sorted(scores, key=scores.get, reverse=True)
    best, second = ranking[:2]
    if scores[best] < max(2., crop.shape[0] * crop.shape[1] * .006) or scores[best] < 1.4 * scores[second]:
        return "unknown"
    return best


#### WIZUALIZACJA ####
def draw(image, items, ax, title=""):
    from matplotlib.patches import Rectangle
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for it in items:
        x1, y1, x2, y2 = it["box"]
        edge = "lime" if it["class"] == "traffic_sign" else "cyan"
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=edge, linewidth=1.3))
        label = it["class"].replace("traffic_", "") + (f"/{it['color']}" if it.get("color") else "")
        ax.text(x1, y1, label, color="black", fontsize=6, bbox={"facecolor": edge, "alpha": .7, "pad": 1})
    ax.set_title(title, fontsize=8)
    ax.axis("off")

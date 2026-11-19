"""Wspólne funkcje pomocnicze używane we wszystkich trzech notebookach (BDD100K)."""

from pathlib import Path
from collections import Counter
import json

import cv2
import numpy as np

CLASSES = ("traffic_sign", "traffic_light")
COLORS = ("red", "yellow", "green", "unknown")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_image(path):
    # imdecode obsługuje polskie znaki w ścieżce na Windows
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Nie można odczytać obrazu: {path}")
    return image


def inventory(root, split="train"):
    root = Path(root)
    if split not in ("train", "val", "test"):
        raise ValueError(split)
    images = {}
    for folder in sorted((root / "bdd100k_images_100k").glob(f"**/{split}")):
        for path in folder.glob("*.jpg"):
            images[path.stem] = path
    labels = sorted((root / "bdd100k_labels").glob(f"**/{split}/*.json"))
    pairs = []
    for label in labels:
        image = images.get(label.stem)
        if image is not None and image.stat().st_size > 0 and label.stat().st_size > 0:
            pairs.append({
                "id": label.stem,
                "image": image.relative_to(root).as_posix(),
                "label": label.relative_to(root).as_posix(),
            })
    return pairs, {
        "split": split,
        "images": len(images),
        "labels": len(labels),
        "paired": len(pairs),
        "unpaired_images": len(images.keys() - {p.stem for p in labels}),
        "unpaired_labels": len(labels) - len(pairs),
    }


def annotations(path, shape=None):
    data = read_json(path)
    frames = data.get("frames", [])
    objects = frames[0].get("objects", []) if frames else data.get("labels", [])
    result, problems = [], Counter()
    for obj in objects:
        category = obj.get("category", "").replace(" ", "_")
        if category not in CLASSES:
            continue
        box = obj.get("box2d")
        if not box:
            problems["missing_box"] += 1
            continue
        try:
            coords = np.array([box[k] for k in ("x1", "y1", "x2", "y2")], float)
        except (KeyError, ValueError, TypeError):
            problems["invalid_box"] += 1
            continue
        if not np.isfinite(coords).all() or coords[2] <= coords[0] or coords[3] <= coords[1]:
            problems["invalid_box"] += 1
            continue
        if shape is not None:
            h, w = shape[:2]
            clipped = np.clip(coords, [0, 0, 0, 0], [w, h, w, h])
            if not np.array_equal(clipped, coords):
                problems["clipped_box"] += 1
            coords = clipped
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                problems["empty_after_clip"] += 1
                continue
        attrs = obj.get("attributes", {})
        color = attrs.get("trafficLightColor", "unknown")
        if color not in COLORS:
            color = "unknown"
        result.append({
            "class": category,
            "box": coords.tolist(),
            "color": color if category == "traffic_light" else None,
            "occluded": bool(attrs.get("occluded", False)),
            "truncated": bool(attrs.get("truncated", False)),
        })
    return result, data.get("attributes", {}), dict(problems)


def make_manifest(root, output, n=1500, seed=42):
    output = Path(output)
    if output.exists():
        manifest = read_json(output)
        if manifest["seed"] != seed or manifest["n"] != n:
            raise ValueError("Istniejący podział ma inne n/seed. Użyj nowego katalogu eksperymentu.")
        return manifest
    if n < 1000:
        raise ValueError("Polecenie wymaga co najmniej 1000 obrazów.")
    sizes = {"train": int(0.7 * n), "val": int(0.1 * n)}
    sizes["test"] = n - sum(sizes.values())
    splits, counts = {}, {}
    rng = np.random.default_rng(seed)
    for split, size in sizes.items():
        pairs, counts[split] = inventory(root, split)
        if len(pairs) < size:
            raise RuntimeError(f"{split}: za mało par: {len(pairs)} / {size}.")
        order = rng.permutation(len(pairs))[:size]
        splits[split] = [pairs[i] for i in order]
        for item in splits[split]:
            image = load_image(Path(root) / item["image"])
            annotations(Path(root) / item["label"], image.shape)
    all_ids = [r["id"] for records in splits.values() for r in records]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Powtórzone identyfikatory między oficjalnymi splitami.")
    manifest = {
        "seed": seed,
        "n": n,
        "source": "official_train_val_test",
        "inventory": counts,
        "splits": splits,
    }
    save_json(output, manifest)
    return manifest


def draw(image, boxes, title="", ax=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        color = "lime" if item["class"] == "traffic_sign" else "cyan"
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.3))
        label = item["class"] + (f"/{item['color']}" if item.get("color") else "")
        ax.text(x1, y1, label, color="black", fontsize=6, bbox={"facecolor": color, "alpha": 0.7, "pad": 1})
    ax.set(title=title)
    ax.axis("off")
    return ax

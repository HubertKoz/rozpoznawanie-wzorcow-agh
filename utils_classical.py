# klasyczny detektor z lab4 (HSV + kontury + geometria), skopiowany bez zmian z zamrozonymi parametrami
from dataclasses import dataclass

import cv2
import numpy as np

import utils_common as w

COLORS = w.COLORS
iou = w.iou


def nms(items, threshold=.35):
    kept = []
    for item in sorted(items, key=lambda x: x['score'], reverse=True):
        if all(item['class'] != old['class'] or iou(item['box'], old['box']) <= threshold for old in kept):
            kept.append(item)
    return kept


@dataclass(frozen=True)
class Params:
    saturation: int = 100
    value: int = 65
    sign_min_area: int = 70
    light_min_area: int = 25
    score_threshold: float = .38
    max_y: float = .85


def masks(image, p):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = (s >= p.saturation) & (v >= p.value)
    # OpenCV: H ma zakres 0..179. Czerwony przechodzi przez koniec skali.
    return hsv, {'red': valid & ((h <= 12) | (h >= 165)),
                 'yellow': valid & (h >= 15) & (h <= 38),
                 'green': valid & (h >= 40) & (h <= 95),
                 'blue': valid & (h >= 96) & (h <= 130)}


def contours(mask, close=3):
    mask = np.uint8(mask) * 255
    if close:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def classify_color(crop, p):
    if crop.size == 0:
        return 'unknown'
    _, mm = masks(crop, p)
    scores = {}
    for color in COLORS[:3]:
        areas = [cv2.contourArea(c) for c in contours(mm[color], close=0)]
        scores[color] = max(areas, default=0.)
    ranking = sorted(scores, key=scores.get, reverse=True)
    best, second = ranking[:2]
    if scores[best] < max(2., crop.shape[0]*crop.shape[1]*.006) or scores[best] < 1.4*scores[second]:
        return 'unknown'
    return best


def detect(image, p=Params()):
    """HSV + kontury + geometria. Wszystkie ramki w oryginalnej skali obrazu."""
    h, w = image.shape[:2]
    hsv, mm = masks(image, p)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = []
    # Znaki kolorowe oraz jasne tablice (np. ograniczenie prędkości w USA).
    white = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 165)
    sign_masks = [mm['red'] | mm['yellow'] | mm['blue'] | mm['green'], white]
    for mask in sign_masks:
        for c in contours(mask):
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw*bh
            if not (p.sign_min_area <= area <= .06*w*h and .35 <= bw/bh <= 3.5 and y < p.max_y*h):
                continue
            fill = cv2.contourArea(c)/area
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = cv2.contourArea(c) / hull_area if hull_area else 0.
            if fill < .4 or solidity < .75:
                continue
            edge = cv2.Canny(gray[y:y+bh, x:x+bw], 70, 150).mean()/255
            if edge < .025:
                continue
            score = float(min(1., .45*fill + .35*min(edge*8, 1) + .2))
            candidates.append({'class': 'traffic_sign', 'box': [x,y,x+bw,y+bh], 'score': score, 'color': None})
    # Obudowy sygnalizatorów: ciemne, zwarte pionowe lub poziome prostokąty.
    dark = gray < 65
    light_boxes = []
    for c in contours(dark):
        x, y, bw, bh = cv2.boundingRect(c)
        if not (p.light_min_area <= bw*bh <= .025*w*h and y < p.max_y*h):
            continue
        ratio = bw/bh
        if not (.18 <= ratio <= .8 or 1.5 <= ratio <= 5.):
            continue
        fill = cv2.contourArea(c)/(bw*bh)
        if fill >= .4:
            light_boxes.append(([x,y,x+bw,y+bh], fill, False))
    # Mała świecąca lampa może nie połączyć się z obudową w masce ciemności.
    # Rozszerzenie to hipoteza geometrii, nie ramka referencyjna.
    # Świecące punkty muszą być jaśniejsze i bardziej nasycone niż zwykły znak.
    lamp_valid = (hsv[:,:,1] >= max(p.saturation,110)) & (hsv[:,:,2] >= 155)
    for color in COLORS[:3]:
        for c in contours(mm[color] & lamp_valid, close=0):
            x,y,bw,bh = cv2.boundingRect(c)
            fill = cv2.contourArea(c)/(bw*bh)
            if 5 <= bw*bh <= 500 and .55 <= bw/bh <= 1.8 and fill >= .45 and y < p.max_y*h:
                for shift in (0., 1., 2.):
                    light_boxes.append(([x-.4*bw,y-shift*bh,x+1.4*bw,y+(3-shift)*bh], .5, True))
    for box, fill, expanded in light_boxes:
        x1,y1,x2,y2 = np.clip(box, [0,0,0,0], [w,h,w,h]).astype(int)
        crop = image[y1:y2,x1:x2]
        if expanded and (crop.size == 0 or np.mean(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) < 85) < .35):
            continue
        color = classify_color(crop, p)
        score = float(.25 + .25*fill + (.3 if color != 'unknown' else 0) - (.05 if expanded else 0))
        candidates.append({'class': 'traffic_light', 'box': [int(x1),int(y1),int(x2),int(y2)],
                           'score': score, 'color': color})
    return nms([item for item in candidates if item['score'] >= p.score_threshold])

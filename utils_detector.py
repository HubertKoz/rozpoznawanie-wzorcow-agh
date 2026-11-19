"""Propozycje, cechy i ocena klasycznego detektora."""
import inspect
from utils_common import CLASSES, COLORS, read_json, save_json, load_image, annotations
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import json
import time

import cv2
import numpy as np
import pandas as pd


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    inter = float(np.prod(wh))
    union = float(np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - inter)
    return inter / union if union > 0 else 0.


def nms(items, threshold=.35):
    kept = []
    for item in sorted(items, key=lambda x: x['score'], reverse=True):
        if all(item['class'] != old['class'] or iou(item['box'], old['box']) <= threshold for old in kept):
            kept.append(item)
    return kept


@dataclass(frozen=True)
class Params:
    saturation: int = 85
    value: int = 65
    sign_min_area: int = 45
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


def match(pred, gt, category, threshold=.5):
    """Greedy po score; jeden GT najwyżej raz. Kolor nie wpływa na dopasowanie."""
    pi = sorted([i for i,p in enumerate(pred) if p['class']==category], key=lambda i: -pred[i]['score'])
    remaining = {i for i,g in enumerate(gt) if g['class']==category}
    matches, fp = [], []
    for i in pi:
        j = max(remaining, key=lambda j: (iou(pred[i]['box'], gt[j]['box']), -j), default=None)
        if j is not None and iou(pred[i]['box'], gt[j]['box']) >= threshold:
            matches.append((i,j))
            remaining.remove(j)
        else:
            fp.append(i)
    return matches, fp, sorted(remaining)


def evaluate(root, records, p, predictions=None):
    rows, errors, color_pairs, saved = [], [], [], {}
    counts = {c: Counter(TP=0, FP=0, FN=0) for c in CLASSES}
    started = time.perf_counter()
    for record in records:
        image = load_image(Path(root)/record['image'])
        gt, _, _ = annotations(Path(root)/record['label'], image.shape)
        pred = predictions[record['id']] if predictions is not None else detect(image, p)
        saved[record['id']] = pred
        for category in CLASSES:
            matched, fp, fn = match(pred, gt, category)
            counts[category].update(TP=len(matched), FP=len(fp), FN=len(fn))
            for pi, gi in matched:
                if category == 'traffic_light':
                    color_pairs.append((gt[gi]['color'], pred[pi]['color']))
            for index in fp:
                errors.append({'id':record['id'], 'type':'FP', 'class':category, 'box':pred[index]['box']})
            for index in fn:
                errors.append({'id':record['id'], 'type':'FN', 'class':category, 'box':gt[index]['box']})
    for category, c in counts.items():
        tp, fp, fn = c['TP'], c['FP'], c['FN']
        rows.append({'class':category, **c, 'precision':tp/(tp+fp) if tp+fp else 0.,
                     'recall':tp/(tp+fn) if tp+fn else 0.,
                     'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.})
    matrix = pd.DataFrame(0, index=COLORS, columns=COLORS)
    for actual, predicted in color_pairs:
        matrix.loc[actual, predicted] += 1
    return {'metrics':pd.DataFrame(rows), 'confusion':matrix, 'errors':errors, 'predictions':saved,
            'seconds':time.perf_counter()-started, 'matched_lights':len(color_pairs),
            'color_accuracy':sum(a==b for a,b in color_pairs)/len(color_pairs) if color_pairs else None}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


LABELS = ('background', 'traffic_sign', 'traffic_light')
FEATURE_VERSION = 'simple_v2'


def proposals(image, limit=100):
    """HSV/ciemne obudowy oraz krawędzie; nie otrzymuje etykiet ani GT."""
    h,w = image.shape[:2]
    base = detect(image, Params(saturation=65, sign_min_area=25,
                                light_min_area=16, score_threshold=0., max_y=1.))
    # Maksymalnie 60 hipotez z masek i 40 z konturów krawędzi.
    boxes = [p['box'] for p in sorted(base, key=lambda p:-p['score'])[:60]]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,70,160)
    extras = []
    for c in contours(edges > 0):
        x,y,bw,bh = cv2.boundingRect(c)
        if 24 <= bw*bh <= .06*w*h and .15 <= bw/bh <= 5. and min(bw,bh)>=3:
            extras.append((cv2.arcLength(c,True)/(bw+bh), [x,y,x+bw,y+bh]))
    boxes += [b for _,b in sorted(extras,key=lambda p:-p[0])[:40]]
    unique = []
    for box in boxes:
        if all(iou(box,old)<.9 for old in unique):
            unique.append(box)
        if len(unique)>=limit:
            break
    return unique


def features(image, box):
    """Stała kolejność nazw. Zwraca jedynie wielkości mierzone na obrazie."""
    h,w = image.shape[:2]
    x1,y1,x2,y2 = np.clip(box,[0,0,0,0],[w,h,w,h]).astype(int)
    if x2<=x1 or y2<=y1:
        raise ValueError(f'Pusta ramka {box}')
    crop = image[y1:y2,x1:x2]
    small = cv2.resize(crop,(32,32),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    hh,ss,vv = cv2.split(hsv)
    f = {'geometry.width':(x2-x1)/w, 'geometry.height':(y2-y1)/h,
         'geometry.aspect':(x2-x1)/(y2-y1), 'geometry.area':(x2-x1)*(y2-y1)/(w*h),
         'geometry.center_x':(x1+x2)/(2*w), 'geometry.center_y':(y1+y2)/(2*h)}
    for channel, name, bins, upper in [(hh,'h',12,180),(ss,'s',8,256),(vv,'v',8,256)]:
        hist = np.histogram(channel,bins=bins,range=(0,upper))[0]/channel.size
        f.update({f'color.{name}_hist_{i:02}':float(v) for i,v in enumerate(hist)})
    saturated = (ss>75)&(vv>70)
    for name, mask in [('red',((hh<13)|(hh>164))&saturated),
                       ('yellow',(hh>=15)&(hh<=38)&saturated),
                       ('green',(hh>=40)&(hh<=95)&saturated),
                       ('blue',(hh>=96)&(hh<=130)&saturated),
                       ('white',(ss<65)&(vv>165)),('dark',vv<65)]:
        f[f'color.fraction_{name}'] = float(mask.mean())
    # Uniform LBP, 8 sąsiadów: liczba zmian 0/1 <=2 -> liczba jedynek, inaczej bin 9.
    center = gray[1:-1,1:-1]
    neighbors = [gray[:-2,:-2],gray[:-2,1:-1],gray[:-2,2:],gray[1:-1,2:],
                 gray[2:,2:],gray[2:,1:-1],gray[2:,:-2],gray[1:-1,:-2]]
    bits = np.stack([n>=center for n in neighbors])
    transitions = (bits != np.roll(bits,1,axis=0)).sum(axis=0)
    codes = np.where(transitions<=2,bits.sum(axis=0),9)
    hist = np.bincount(codes.ravel(),minlength=10)/codes.size
    f.update({f'texture.lbp_{i:02}':float(v) for i,v in enumerate(hist)})
    f['texture.gray_std'] = float(gray.std()/255)
    f['texture.laplacian_var'] = float(cv2.Laplacian(gray,cv2.CV_32F).var()/255**2)
    gx = cv2.Sobel(gray,cv2.CV_32F,1,0)
    gy = cv2.Sobel(gray,cv2.CV_32F,0,1)
    magnitude, angle = cv2.cartToPolar(gx,gy,angleInDegrees=True)
    hist = np.histogram(angle%180,bins=9,range=(0,180),weights=magnitude)[0]
    hist = hist/(hist.sum()+1e-8)
    f.update({f'edges.orientation_{i:02}':float(v) for i,v in enumerate(hist)})
    f['edges.density'] = float((cv2.Canny(gray,70,150)>0).mean())
    f['edges.magnitude'] = float(magnitude.mean()/1020)
    return f


def assign_label(box, truth):
    if not truth:
        return 'background'
    overlaps = [iou(box,g['box']) for g in truth]
    best = int(np.argmax(overlaps))
    if overlaps[best]>=.5:
        return truth[best]['class']
    return 'background' if max(overlaps)<=.2 else None


def feature_table(root, records, training, seed=42, limit=100, cache=None):
    """Train: GT+jitter+kandydaci+tło. Val: wyłącznie rzeczywiste propozycje."""
    root = Path(root)
    signature = {'version':FEATURE_VERSION, 'records':fingerprint(records),
                 'training':training, 'seed':seed, 'limit':limit,
                 'code':hashlib.sha256((FEATURE_VERSION + ''.join(inspect.getsource(fn) for fn in (proposals, features, assign_label, feature_table))).encode()).hexdigest()}
    if cache:
        cache = Path(cache)
        if cache.with_suffix('.json').exists() and cache.with_suffix('.npz').exists():
            if read_json(cache.with_suffix('.json')) == signature:
                data = np.load(cache.with_suffix('.npz'),allow_pickle=False)
                return pd.DataFrame(data['X'],columns=data['columns']), data['y'], data['groups']
    rng = np.random.default_rng(seed)
    rows, targets, groups = [],[],[]
    for number, record in enumerate(records):
        image = load_image(root/record['image'])
        truth,_,_ = annotations(root/record['label'],image.shape)
        candidates = proposals(image,limit)
        if training:
            # GT używamy tylko do konstrukcji uczących wycinków, nigdy w detekcji.
            positives = []
            for obj in truth:
                positives.append(obj['box'])
                b = np.array(obj['box'],float)
                bw,bh = b[2:]-b[:2]
                jitter = b+rng.uniform(-.1,.1,4)*np.array([bw,bh,bw,bh])
                jitter = np.clip(jitter,[0,0,0,0],[image.shape[1],image.shape[0]]*2)
                if iou(jitter,b)>=.5:
                    positives.append(jitter.tolist())
            # Losowe tło obejmuje także obrazy bez znaków i świateł.
            h,w = image.shape[:2]
            for _ in range(20):
                bw = int(rng.integers(6,max(7,w//8)))
                bh = int(rng.integers(6,max(7,h//6)))
                x = int(rng.integers(0,w-bw+1)); y = int(rng.integers(0,h-bh+1))
                candidates.append([x,y,x+bw,y+bh])
            negatives = [b for b in candidates if assign_label(b,truth)=='background']
            if len(negatives)>24:
                negatives = [negatives[i] for i in rng.choice(len(negatives),24,replace=False)]
            detected_positive = [b for b in candidates if assign_label(b,truth) in LABELS[1:]]
            candidates = positives+detected_positive+negatives
        for box in candidates:
            label = assign_label(box,truth)
            if label is None:
                continue  # niejednoznaczny wycinek, 0.2 < IoU < 0.5
            rows.append(features(image,box)); targets.append(label); groups.append(record['id'])
        if (number+1)%100==0:
            print(f'Cechy: {number+1}/{len(records)} obrazów; {len(rows)} wycinków',flush=True)
    if not rows:
        raise ValueError('Brak wycinków do analizy.')
    X = pd.DataFrame(rows,dtype=np.float32)
    y = np.asarray(targets,dtype=str); groups = np.asarray(groups,dtype=str)
    if cache:
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache.with_suffix('.npz'),X=X.to_numpy(),y=y,groups=groups,
                            columns=np.asarray(X.columns,dtype=str))
        save_json(cache.with_suffix('.json'),signature)
    return X,y,groups


def train_forest(X,y,seed=42):
    missing = set(LABELS)-set(y)
    if missing:
        raise ValueError(f'Brak klas w danych uczących: {missing}. Zwiększ train.')
    model = RandomForestClassifier(n_estimators=100,max_depth=18,min_samples_leaf=2,
                                   max_features='sqrt',class_weight='balanced_subsample',
                                   random_state=seed,n_jobs=-1)
    model.fit(X,y)
    return model


def macro_f1(y,pred):
    return f1_score(y,pred,labels=list(LABELS),average='macro',zero_division=0)


def compare_groups(X,y,V,vy,seed=42):
    """Ablacja: te same train/val i hiperparametry, inna lista cech."""
    groups = list(dict.fromkeys(c.split('.')[0] for c in X.columns))
    variants = {g:[c for c in X if c.startswith(g+'.')] for g in groups}
    variants['color+geometry'] = [c for c in X if c.split('.')[0] in ('color','geometry')]
    variants['all'] = list(X.columns)
    variants.update({'without_'+g:[c for c in X if not c.startswith(g+'.')] for g in groups})
    rows,models = [],{}
    for name,columns in variants.items():
        start = time.perf_counter()
        model = train_forest(X[columns],y,seed)
        pred = model.predict(V[columns])
        rows.append({'variant':name,'features':len(columns),'val_crop_macro_F1':macro_f1(vy,pred),
                     'seconds':time.perf_counter()-start})
        models[name] = (model,columns)
        print(name,round(rows[-1]['val_crop_macro_F1'],4),flush=True)
    return pd.DataFrame(rows).sort_values('val_crop_macro_F1',ascending=False),models


def permutation_groups(model,X,y,repeats=3,seed=42):
    """Mieszamy wszystkie kolumny grupy jednym porządkiem, zachowując jej wnętrze."""
    rng = np.random.default_rng(seed)
    baseline = macro_f1(y,model.predict(X))
    rows = []
    for group in dict.fromkeys(c.split('.')[0] for c in X.columns):
        columns = [c for c in X if c.startswith(group+'.')]
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled.loc[:,columns] = X[columns].to_numpy()[rng.permutation(len(X))]
            drops.append(baseline-macro_f1(y,model.predict(shuffled)))
        rows.append({'group':group,'mean_F1_drop':float(np.mean(drops)),
                     'std_F1_drop':float(np.std(drops)),'features':len(columns)})
    return pd.DataFrame(rows).sort_values('mean_F1_drop',ascending=False)


def rf_detect(image,model,columns,threshold=.5,limit=100):
    boxes = proposals(image,limit)
    if not boxes:
        return []
    X = pd.DataFrame([features(image,b) for b in boxes],dtype=np.float32)[columns]
    probabilities = model.predict_proba(X)
    items = []
    for box,probs in zip(boxes,probabilities):
        index = int(np.argmax(probs))
        category = model.classes_[index]
        score = float(probs[index])
        if category=='background' or score<threshold:
            continue
        x1,y1,x2,y2 = map(int,box)
        color = classify_color(image[y1:y2,x1:x2],Params()) if category=='traffic_light' else None
        items.append({'class':str(category),'box':box,'score':score,'color':color})
    return nms(items)


def predict_records(root,records,model,columns,threshold=.5,limit=100):
    predictions = {}
    for i,record in enumerate(records):
        predictions[record['id']] = rf_detect(load_image(Path(root)/record['image']),model,columns,threshold,limit)
        if (i+1)%50==0:
            print(f'Detekcja RF: {i+1}/{len(records)}',flush=True)
    return predictions


def proposal_recall(root,records,limit=100):
    """Sufit recall dla tych propozycji: bez poprawnej ramki RF nie ma szans."""
    counts = {c:[0,0] for c in LABELS[1:]}
    for record in records:
        image = load_image(Path(root)/record['image'])
        truth,_,_ = annotations(Path(root)/record['label'],image.shape)
        boxes = proposals(image,limit)
        for obj in truth:
            counts[obj['class']][1] += 1
            counts[obj['class']][0] += int(any(iou(b,obj['box'])>=.5 for b in boxes))
    return pd.DataFrame([{'class':c,'covered_GT':a,'all_GT':b,'proposal_recall':a/b if b else 0.}
                         for c,(a,b) in counts.items()])


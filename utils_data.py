"""Funkcje do analizy danych BDD100K."""
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def raw_objects(data):
    frames = data.get('frames',[])
    return frames[0].get('objects',[]) if frames else data.get('labels',[])


def summarize_train(root,records):
    scenes,objects,categories,issues = [],[],Counter(),Counter()
    for record in records:
        data = read_json(Path(root)/record['label'])
        raw = raw_objects(data)
        # Nagłówek JPEG wystarcza do odczytu wymiarów; bez dekodowania całego zdjęcia.
        with Image.open(Path(root)/record['image']) as image:
            w,h = image.size
        truth,attrs,problems = annotations(Path(root)/record['label'],(h,w))
        issues.update(problems)
        categories.update(o.get('category','<missing>') for o in raw)
        row = {'id':record['id'],**attrs,'width':w,'height':h,
               'all_annotations':len(raw), 'all_bbox':sum('box2d' in o for o in raw),
               'poly2d':sum('poly2d' in o for o in raw),'targets':len(truth),
               'signs':sum(o['class']=='traffic_sign' for o in truth),
               'lights':sum(o['class']=='traffic_light' for o in truth),
               'occluded':any(o['occluded'] for o in truth),
               'truncated':any(o['truncated'] for o in truth),
               'tiny':any(min(o['box'][2]-o['box'][0],o['box'][3]-o['box'][1])<10 for o in truth)}
        for color in ('red','yellow','green','unknown'):
            row[color] = sum(o['color']==color for o in truth)
        scenes.append(row)
        for o in truth:
            x1,y1,x2,y2 = o['box']
            objects.append({'id':record['id'],'class':o['class'],'color':o['color'],
                            'width':x2-x1,'height':y2-y1,'occluded':o['occluded'],
                            'truncated':o['truncated']})
    return pd.DataFrame(scenes),pd.DataFrame(objects),categories,dict(issues)


def select_examples(scene):
    """Jawnie raportujemy brak danego przypadku zamiast podmieniać go innym."""
    conditions = {
        'Znaki, bez sygnalizatorów':(scene.signs>0)&(scene.lights==0),
        'Sygnalizatory, bez znaków':(scene.lights>0)&(scene.signs==0),
        'Znaki i sygnalizatory':(scene.signs>0)&(scene.lights>0),
        'Brak klas docelowych, inne adnotacje':(scene.targets==0)&(scene.all_annotations>0),
        'Całkowicie pusta lista obiektów':scene.all_annotations==0,
        'Czerwone światło':scene.red>0,'Żółte światło':scene.yellow>0,
        'Zielone światło':scene.green>0,'Kolor unknown':scene.unknown>0,
        'Obiekt zasłonięty':scene.occluded,'Obiekt ucięty':scene.truncated,
        'Mały obiekt (<10 px)':scene.tiny,
        'Wiele znaków (>=5)':scene.signs>=5,'Adnotacje polilinii':scene.poly2d>0}
    for column,value,title in [('timeofday','night','Noc'),('timeofday','dawn/dusk','Świt / zmierzch'),
                                ('weather','rainy','Deszcz'),('weather','snowy','Śnieg'),
                                ('weather','foggy','Mgła')]:
        if column in scene:
            conditions[title] = scene[column]==value
    rows = []
    used = set()
    for title,mask in conditions.items():
        available = scene.loc[mask,'id'].tolist()
        name = next((n for n in available if n not in used),available[0] if available else None)
        if name: used.add(name)
        rows.append({'example':title,'count_in_train':len(available),'id':name})
    return pd.DataFrame(rows)


def draw_all(image,data,ax):
    """Pozostałe bbox oraz polilinie, żeby rozróżnić brak celu od braku adnotacji."""
    ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    raw = raw_objects(data)
    names = sorted(set(o.get('category','?') for o in raw))
    palette = plt.get_cmap('tab20')
    colors = {name:palette(i%20) for i,name in enumerate(names)}
    for obj in raw:
        name = obj.get('category','?'); color = colors[name]
        if 'box2d' in obj:
            b = obj['box2d']; x,y = b['x1'],b['y1']
            ax.add_patch(Rectangle((x,y),b['x2']-x,b['y2']-y,fill=False,color=color,linewidth=1))
            ax.text(x,y,name,fontsize=5,color='black',bbox={'facecolor':color,'alpha':.7,'pad':.5})
        poly = obj.get('poly2d',[])
        if poly:
            # Dostarczona wersja przechowuje punkty [x,y,typ]. To podgląd polilinii,
            # nie renderowanie krzywych Beziera ani maski segmentacyjnej.
            if isinstance(poly[0],list):
                xy = np.asarray([[p[0],p[1]] for p in poly],float)
                ax.plot(xy[:,0],xy[:,1],color=color,linewidth=1,linestyle='--',label=name)
    ax.axis('off')


def gallery(root,records,examples,out):
    lookup = {r['id']:r for r in records}
    selected = examples.loc[examples.id.notna()].to_dict('records')
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(selected),6):
        fig,axes = plt.subplots(3,2,figsize=(17,15))
        for ax,item in zip(axes.ravel(),selected[start:start+6]):
            r = lookup[item['id']]
            image = load_image(Path(root)/r['image'])
            truth,attrs,_ = annotations(Path(root)/r['label'],image.shape)
            draw(image,truth,ax=ax)
            ax.set_title(item['example']+'\n'+r['id']+' | '+str(attrs),fontsize=8)
        for ax in axes.ravel()[len(selected[start:start+6]):]: ax.axis('off')
        fig.tight_layout(); fig.savefig(out/f'gallery_{start//6+1}.png',dpi=110)
        plt.show()


def crop_gallery(root,records,out):
    selected = {}
    for r in records:
        truth,_,_ = annotations(Path(root)/r['label'])
        for obj in truth:
            key = obj['class'] if obj['class']=='traffic_sign' else 'traffic_light_'+obj['color']
            b = obj['box']; area = (b[2]-b[0])*(b[3]-b[1])
            selected.setdefault(key,[]).append((area,r,obj))
    fig,axes = plt.subplots(2,5,figsize=(16,8))
    keys = ['traffic_sign','traffic_light_red','traffic_light_yellow','traffic_light_green','traffic_light_unknown']
    for row,mode in enumerate(['największy','najmniejszy']):
        for ax,key in zip(axes[row],keys):
            if key in selected:
                candidates = sorted(selected[key],key=lambda item:item[0])
                _,r,obj = candidates[-1] if row==0 else candidates[0]
                image = load_image(Path(root)/r['image'])
                x1,y1,x2,y2 = np.clip(obj['box'],[0,0,0,0],[image.shape[1],image.shape[0]]*2).astype(int)
                x2=max(x1+1,x2); y2=max(y1+1,y2)
                ax.imshow(cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))
                ax.set_title(key.replace('traffic_','')+f'\n{mode}: {x2-x1} × {y2-y1} px',fontsize=8)
            else: ax.set_title(key+'\nBRAK w train',fontsize=8)
            ax.axis('off')
    fig.tight_layout(); fig.savefig(Path(out)/'target_crops.png',dpi=130); plt.show()

from utils_common import read_json, annotations, load_image, draw


def position_size_table(root, records):
    """Położenie i rozmiar ramek w proporcji do rozmiaru obrazu."""
    rows = []
    for record in records:
        with Image.open(Path(root) / record['image']) as image:
            w, h = image.size
        objects, _, _ = annotations(Path(root) / record['label'], (h, w))
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            rows.append({'class': obj['class'], 'x': (x1+x2)/(2*w),
                         'y': (y1+y2)/(2*h), 'width': (x2-x1)/w,
                         'height': (y2-y1)/h})
    return pd.DataFrame(rows, columns=['class', 'x', 'y', 'width', 'height'])


def plot_position_size(objects, bins=(16, 9), min_count=5):
    """Liczba środków oraz średni rozmiar w komórkach mapy."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for row, category in enumerate(('traffic_sign', 'traffic_light')):
        data = objects[objects['class'] == category]
        count, _, _ = np.histogram2d(data['y'], data['x'], bins=(bins[1], bins[0]), range=((0, 1), (0, 1)))
        for col, (name, values) in enumerate((('Liczba obiektów', None), ('Średnia szerokość / szer. obrazu', data['width']), ('Średnia wysokość / wys. obrazu', data['height']))):
            if values is None:
                shown = count
            else:
                total, _, _ = np.histogram2d(data['y'], data['x'], bins=(bins[1], bins[0]), range=((0, 1), (0, 1)), weights=values)
                shown = np.divide(total, count, out=np.full_like(total, np.nan), where=count >= min_count)
            plot = axes[row, col].imshow(shown, origin='upper', extent=(0, 1, 1, 0), aspect='auto', cmap='viridis')
            axes[row, col].set(title=f'{category}: {name}', xlabel='x / szerokość obrazu', ylabel='y / wysokość obrazu')
            fig.colorbar(plot, ax=axes[row, col])
    return fig

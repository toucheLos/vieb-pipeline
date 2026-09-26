import sys, os, json, glob, numpy as np, cv2
sys.path.insert(0,'/home/tul26194/vieb-pipeline'); sys.path.insert(1,'/home/tul26194/recur'); sys.path.insert(0,'/home/tul26194/vieb-pipeline/scripts')
import point_prompts as pp, stabilise3 as s3
from recur.render import video as vid
from vieb.io import spine
from vieb.pixel import sam
from vieb.tok import ego as tego
rows=[]
for f in glob.glob('work/point_prompts/*.json'):
    d=json.load(open(f))
    for r in d['rows']:
        if r['box_ok'] and r['pt_ok']: rows.append((r['area_pt']-r['area_box'], r['area_pt'], d['recording_id'], r['t']))
rows.sort(reverse=True)
rng=np.random.default_rng(0)
top=[rows[i] for i in rng.choice(min(200,len(rows)),6,replace=False)]   # 6 from the 200 biggest growths
typ=[rows[i] for i in rng.choice(len(rows),3,replace=False)]             # 3 typical
pred=sam.sam_point_predictor(s3.CHECKPOINT)
tiles=[]
for tag,(dg,ar,rid,t) in [('BIG',x) for x in top]+[('typical',x) for x in typ]:
    cap=cv2.VideoCapture(vid.video_path(rid)); cap.set(cv2.CAP_PROP_POS_FRAMES,t); ok,fr=cap.read(); cap.release()
    p=spine.clean(rid)['pose'][t]; bl=float(np.nanmedian(tego.body_length(spine.clean(rid)['pose'])))
    box=sam.keypoint_box(p,sam.PAD_BL*bl); pts=p[list(pp.PROMPTS)]; pts=pts[np.isfinite(pts).all(axis=1)]
    m,_=pred(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB),box,points=pts)
    fr[m]=(0.6*fr[m]+0.4*np.array([40,40,230])).astype(np.uint8)
    for q in pts: cv2.circle(fr,tuple(int(v) for v in q),5,(0,255,255),-1)
    for q in p[list(pp.EARS)]:
        if np.isfinite(q).all(): cv2.circle(fr,tuple(int(v) for v in q),4,(255,120,0),-1)
    cv2.rectangle(fr,(0,0),(640,28),(0,0,0),-1)
    cv2.putText(fr,f'{tag}: area {ar:.2f} bl^2 (+{dg:.2f} vs box)',(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    tiles.append(cv2.resize(fr,(400,300)))
grid=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,9,3)])
cv2.imwrite('work/scratch/big_masks.jpg',grid); print('ok')

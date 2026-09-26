import sys, os, numpy as np, cv2, glob
sys.path.insert(0,'/home/tul26194/vieb-pipeline'); sys.path.insert(1,'/home/tul26194/recur'); sys.path.insert(0,'/home/tul26194/vieb-pipeline/scripts')
import agreement as ag
from recur.render import video as vid
from vieb.io import spine
rng=np.random.default_rng(0)
files=sorted(glob.glob('work/agreement/*.npz'))
want={'dlc_suspect':3,'unresolved':3,'sam_suspect':2,'agree':1}
tiles=[]
for cname,k in want.items():
    ci=ag.CLASSES.index(cname); got=0
    for f in rng.permutation(files):
        z=np.load(f); idx=np.flatnonzero(z['cls']==ci)
        if idx.size==0: continue
        i=int(rng.choice(idx)); t=int(z['key_t'][i]); rid=os.path.basename(f)[:-4]
        masks=ag._unpack(np.load(f'work/stabilise7/rec/{rid}.npz'))
        cap=cv2.VideoCapture(vid.video_path(rid)); cap.set(cv2.CAP_PROP_POS_FRAMES,t); ok,fr=cap.read(); cap.release()
        if not ok: continue
        if t in masks:
            x0,y0,m=masks[t]; sub=fr[y0:y0+m.shape[0],x0:x0+m.shape[1]]; sub[m]=(0.6*sub[m]+0.4*np.array([40,40,230])).astype(np.uint8)
        p=spine.clean(rid)['pose'][t]
        if np.isfinite(p).all(): vid.draw_skeleton(fr,p,cv2)
        cv2.rectangle(fr,(0,0),(640,30),(0,0,0),-1)
        cv2.putText(fr,f"{cname}  in {z['kp_in'][i]:.2f} d {z['dist'][i]:.2f}bl ang {z['angle'][i]:.0f}",(6,21),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
        tiles.append(cv2.resize(fr,(400,300))); got+=1
        if got==k: break
while len(tiles)%3: tiles.append(np.zeros((300,400,3),np.uint8))
grid=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
cv2.imwrite('work/scratch/agree_examples.jpg',grid); print('ok',len(tiles))

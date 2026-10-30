"""曲线级对比：旧 50ms 响应与新中心修正的音高中心误差、颤音保留与原唱细节移植。"""
import sys, json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scipy.ndimage import median_filter, uniform_filter1d, gaussian_filter1d
import shadowvox.register as R
from shadowvox.align import estimate_shift
from shadowvox.natural import runs
base=ROOT/'out/naturalness/qinghua/register_strength'
d=np.load(base/'curves.npz')
rep=json.loads((base/'report.json').read_text(encoding='utf-8'))
cap={}
orig=R.register_shift
def spy(dry,target,voiced,strength,m=450):
    cap.update(dry=dry.copy(),target=target.copy(),voiced=voiced.copy()); return orig(dry,target,voiced,strength,m)
R.register_shift=spy
def old_shift(dry,reference,voiced,strength,mx=450):
    target,reliable,_=R.fold_register(dry,reference,voiced)
    shift=np.zeros(len(dry))
    for s,e in runs(reliable):
        if e-s<10: continue
        diff=target[s:e]-dry[s:e]
        delta=uniform_filter1d(median_filter(diff,size=3,mode='nearest'),size=5,mode='nearest')
        dev=median_filter(np.abs(diff-delta),size=21,mode='nearest')
        conf=np.clip((160-dev)/100,0,1)
        shift[s:e]=gaussian_filter1d(np.clip(delta,-mx,mx)*strength*conf,sigma=1,mode='constant')
    shift[~reliable]=0
    return shift,target
def ctr(x,mask):
    y=np.zeros(len(x))
    for s,e in runs(mask):
        if e-s>=30: y[s:e]=median_filter(median_filter(x[s:e],15,mode='nearest'),21,mode='nearest')
    return y
result={}
for strength in (.75,1.):
    estimate_shift(d['source_f0'],d['reference_f0'],strength=strength,max_shift_cents=450,
                   offset_override=rep['after_rhythm_offset_seconds'],pitch_mode='register')
    dry,target,voiced=cap['dry'],cap['target'],cap['voiced']
    new,rel,_=orig(dry,target,voiced,strength)
    old,tf=old_shift(dry,target,voiced,strength)
    m=np.zeros(len(dry),bool)
    for s,e in runs(rel):
        if e-s>=30: m[s+10:e-10]=True     # 只评估完整乐句内部，避开边界
    tc,dc=ctr(tf,rel),ctr(dry,rel)
    fd,fr=(dry-dc)[m],(tf-tc)[m]
    row={}
    for name,sh in (('原干声',np.zeros(len(dry))),('旧50ms',old),('新中心',new)):
        o=dry+sh; fo=(o-ctr(o,rel))[m]; ce=np.abs(ctr(o,rel)-tc)[m]
        row[name]=dict(center_median=round(float(np.median(ce)),1),center_p90=round(float(np.percentile(ce,90)),1),
                       frame_median=round(float(np.median(np.abs(o-tf)[m])),1),
                       detail_std_ratio=round(float(np.std(fo)/np.std(fd)),2),
                       corr_singer=round(float(np.corrcoef(fo,fd)[0,1]),2),
                       corr_reference=round(float(np.corrcoef(fo,fr)[0,1]),2))
    result[f'{strength:.0%}']=dict(frames=int(m.sum()),**row)
print(json.dumps(result,ensure_ascii=False,indent=1))
(ROOT/'out/center_eval').mkdir(parents=True,exist_ok=True)
(ROOT/'out/center_eval/curve_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

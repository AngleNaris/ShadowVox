"""有限节奏修正：可靠局部锚点 + 明确禁止移动区域。100fps 网格。"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from .f0 import cent
from .natural import runs

FPS = 100


def segments(f0):
    return [(a, b) for a, b in runs(f0 > 0) if b-a >= 6]


def content_cost(dry, ref, a, b):
    """比较起音后的共同浊音走势；只扣除八度差，不抹去旋律差。"""
    da, db = a
    ra, rb = b
    if min(db-da, rb-ra)/max(db-da, rb-ra) < .3:
        return None
    n = min(45, db-da, rb-ra)
    if n < 6:
        return None
    difference = cent(dry[da:da+n])-cent(ref[ra:ra+n])
    difference -= 1200*np.rint(np.median(difference)/1200)
    gap = float(np.median(np.abs(difference)))
    shape = float(np.median(np.abs(difference-np.median(difference))))
    if gap > 240 or shape > 140:
        return None
    return gap/240*.35 + shape/140*.25


def match_onsets(dry, ref, offset, limit=.15):
    ds, rs = segments(dry), segments(ref)
    score = np.full((len(ds), len(rs)), np.inf)
    for i, d in enumerate(ds):
        for j, r in enumerate(rs):
            delta = d[0]/FPS-r[0]/FPS-offset
            if abs(delta) > limit+1e-8:
                continue
            cost = content_cost(dry, ref, d, r)
            if cost is not None:
                score[i,j] = abs(delta)/limit*.4 + cost
    pairs = []
    if not score.size:
        return ds, rs, pairs
    for i in range(len(ds)):
        j = int(np.argmin(score[i]))
        if not np.isfinite(score[i,j]) or np.argmin(score[:,j]) != i:
            continue
        row, col = np.sort(score[i]), np.sort(score[:,j])
        if (len(row)>1 and row[1]-row[0]<.06) or (len(col)>1 and col[1]-col[0]<.06):
            continue
        if pairs and j <= pairs[-1][1]:
            continue
        pairs.append((i,j))
    # 孤立短音缺乏相邻起音支持，不作为时间锚点。
    supported = []
    for i,j in pairs:
        near = sum(abs(ds[k][0]-ds[i][0]) <= 300 for k,_ in pairs)
        if near >= 2 or ds[i][1]-ds[i][0] >= 100:
            supported.append((i,j))
    return ds, rs, supported


def plan(dry_f0, ref_f0, max_shift=.15, max_rate=.10, offset_override=None):
    from .rhythm import onset_offset
    n = len(dry_f0)
    candidate = onset_offset(dry_f0, ref_f0) if offset_override is None else float(offset_override)
    ds, rs, pairs = match_onsets(dry_f0, ref_f0, candidate, max_shift)
    reliable_global = (len(pairs)>=4 and len(pairs)>=.2*min(len(ds),len(rs))
                       and ds[pairs[-1][0]][0]-ds[pairs[0][0]][0]>=300)
    offset = candidate if reliable_global or offset_override is not None else 0.
    if offset != candidate:
        ds, rs, pairs = match_onsets(dry_f0, ref_f0, offset, max_shift)
    # 整体时移未获确认时，单个偶合起音不足以区分录音延迟与局部节奏。
    if not reliable_global and offset_override is None and len(pairs) < 2:
        pairs = []
    accepted = {i for i,_ in pairs}
    allowed = np.zeros(n, bool)
    protected = np.zeros(n, bool)
    for i,(a,b) in enumerate(ds):
        if i in accepted:
            allowed[max(0,a-60):min(n,b+60)] = True
        else:
            protected[max(0,a-4):min(n,b+4)] = True
    # 参考有持续歌声、源没有：整段禁止移动，不能借相邻句填空。
    idx = np.rint(np.arange(n)-offset*FPS).astype(int)
    rv = np.zeros(n, bool)
    inside = (idx>=0)&(idx<len(ref_f0))
    rv[inside] = ref_f0[idx[inside]]>0
    for a,b in runs(rv & (dry_f0<=0)):
        if b-a>=30:
            protected[max(0,a-4):min(n,b+4)] = True
    allowed &= ~protected
    if n:
        allowed[:4] = False
        allowed[-4:] = False
    lead = np.zeros(n)
    anchors = [(ds[i][0], ds[i][0]/FPS-rs[j][0]/FPS-offset) for i,j in pairs]
    for a,b in runs(allowed):
        points = [(t,v) for t,v in anchors if a<=t<b]
        if not points:
            allowed[a:b] = False
            continue
        t,v = zip(*points)
        lead[a:b] = np.interp(np.arange(a,b), [a-1,*t,b], [0,*v,0])
    # 保护区两侧回到零，并限制瞬时时间伸缩率。
    ceiling = distance_transform_edt(allowed)/FPS*max_rate
    lead = np.clip(lead, -np.minimum(ceiling,max_shift), np.minimum(ceiling,max_shift))
    step = max_rate/FPS
    for i in range(1,n):
        lead[i] = np.clip(lead[i],lead[i-1]-step,lead[i-1]+step)
    for i in range(n-2,-1,-1):
        lead[i] = np.clip(lead[i],lead[i+1]-step,lead[i+1]+step)
    lead[~allowed] = 0
    diffs = np.abs([v for _,v in anchors])
    stats = {"algorithm":"guarded-v1", "applied":bool(np.max(np.abs(lead),initial=0)>.001),
             "onset_pairs":len(pairs), "source_onsets":len(ds), "skipped_onsets":len(ds)-len(pairs),
             "global_offset_reliable":bool(reliable_global), "onset_offset_before_seconds":float(offset),
             "placement_offset_seconds":float(offset), "max_shift_seconds":max_shift, "max_rate":max_rate,
             "max_applied_shift_ms":float(np.max(np.abs(lead),initial=0)*1000),
             "onset_dev_p50_ms":float(np.median(diffs)*1000) if len(diffs) else 0.,
             "onset_dev_p90_ms":float(np.percentile(diffs,90)*1000) if len(diffs) else 0.,
             "protected_intervals_seconds":[[a/FPS,b/FPS] for a,b in runs(protected)],
             "active_intervals_seconds":[[a/FPS,b/FPS] for a,b in runs(allowed)]}
    return lead, allowed, stats


def align(dry, dry_f0, ref_f0, sr, max_rate=.10, log=print):
    from .rhythm import warp_wsola
    lead, allowed, stats = plan(dry_f0, ref_f0, max_rate=max_rate)
    if not stats["applied"]:
        log(f"[节奏] 无可靠可修片段，保留原时间位置；跳过 {stats['skipped_onsets']} 个起音")
        return dry.copy(), stats
    grid = np.arange(len(lead))
    warped = warp_wsola(dry,sr,lambda t:float(np.interp(t*FPS,grid,lead)))
    # 仅接受区域参与处理，禁止区逐采样保留；40ms 淡入淡出避免拼接突变。
    blend = np.clip(distance_transform_edt(allowed)/4,0,1)
    frame = np.minimum((np.arange(len(dry))*FPS/sr).astype(int),len(allowed)-1)
    mix = blend[frame]
    y = (dry*(1-mix)+warped*mix).astype(np.float32)
    y[~allowed[frame]] = dry[~allowed[frame]]
    log(f"[节奏] 接受 {stats['onset_pairs']} 个起音，跳过 {stats['skipped_onsets']} 个；"
        f"实际最大移动 {stats['max_applied_shift_ms']:.0f}ms，禁止区保持原位")
    return y, stats

"""Resumable export queue: renders every job of tools/jobs.py in chunks (16-bit PNG), encodes each chunk to
ProRes 422 HQ (10-bit, BT.709), deletes the PNGs, and when a job is complete concatenates the segments (+ black
frames outside the used range) into EXPORT/final/<job>.mov.  Finished chunks / jobs are never redone.

  python3 render_queue.py run    [job ...] [--budget MIN] [--workers N]   # stop starting chunks after MIN minutes
  python3 render_queue.py verify job [frames]  [--outs 1.5]                  # test frames vs the approved preview
  python3 render_queue.py status
Env: WORK (pipeline dir), A (uploads), EXPORT (output dir, default WORK/../export)."""
import os, sys, time, glob, shutil, subprocess, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobs import JOBS, WORK
import numpy as np, cv2

EXPORT = os.environ.get('EXPORT', os.path.join(os.path.dirname(WORK.rstrip('/')), 'export'))
PY = sys.executable
BYNAME = {j['name']: j for j in JOBS}
ENC = ['-c:v', 'prores_ks', '-profile:v', '3', '-vendor', 'apl0',
       '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv422p10le',
       '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-r', '25']


def chunks(j):
    u0, u1 = j['used']; c = j['chunk']
    return [(r, min(r + c, u1)) for r in range(u0, u1, c)]


def launch(j, r0, r1, cdir, outs):
    os.makedirs(cdir, exist_ok=True)
    env = dict(os.environ)
    for kv in j['env']:
        k, v = kv.split('=', 1); env[k] = v
    env['OUTS'] = str(outs)
    kind = j['kind']; cmd = [PY] + j['cmd']
    if kind in ('dsc', 'dso'):
        env['STILL'] = os.path.join(cdir, 'x'); env['RANGE'] = f'{r0},{r1}'
    elif kind == 'cn':
        env['PNGDIR'] = cdir; env['PNGNAME'] = 'x'; env['RANGE'] = f'{r0},{r1}'
    elif kind == 'f01':
        cmd += [cdir, str(j['f0'] + r0), str(j['f0'] + r1)]
    elif kind == 'sil':
        cmd += [cdir, str(r0), str(r1)]
    log = open(os.path.join(cdir, 'log.txt'), 'a')
    return subprocess.Popen(cmd, cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT)


def frame_path(j, cdir, k):
    if j['kind'] in ('dsc', 'dso', 'cn'):
        return os.path.join(cdir, f'x_{k:04d}.png')
    return os.path.join(cdir, j.get('pat', 'x_{:04d}.png').format(j['f0'] + k if j['kind'] == 'f01' else k))


def blackpoint(f, bp=1250.0):
    a = cv2.imread(f, -1).astype(np.float32)
    cv2.imwrite(f, (np.clip((a - bp) * 65535.0 / (65535.0 - bp), 0, 65535) + 0.5).astype(np.uint16))


def encode_chunk(j, r0, r1, cdir, seg):
    lst = os.path.join(cdir, 'list.txt')
    with open(lst, 'w') as fh:
        for k in range(r0, r1):
            fh.write(f"file '{frame_path(j, cdir, k)}'\nduration 0.04\n")
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst] + ENC +
                   ['-frames:v', str(r1 - r0), seg + '.tmp.mov'], check=True)
    os.replace(seg + '.tmp.mov', seg)


def black_seg(j, n, path):
    w, h = j['layer']['size']
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'color=black:s={w}x{h}:r=25,format=rgb48le',
                    '-frames:v', str(n)] + ENC + [path], check=True)


def finish(j):
    jd = os.path.join(EXPORT, j['name']); fin = os.path.join(EXPORT, 'final', j['name'] + '.mov')
    if os.path.exists(fin):
        return fin
    parts = []; u0, u1 = j['used']
    if u0 > 0:
        p = os.path.join(jd, 'black_head.mov'); black_seg(j, u0, p); parts.append(p)
    parts += [os.path.join(jd, f'seg_{r0:04d}.mov') for r0, _ in chunks(j)]
    if u1 < j['n']:
        p = os.path.join(jd, 'black_tail.mov'); black_seg(j, j['n'] - u1, p); parts.append(p)
    os.makedirs(os.path.dirname(fin), exist_ok=True)
    lst = os.path.join(jd, 'concat.txt')
    open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', fin + '.tmp.mov'], check=True)
    os.replace(fin + '.tmp.mov', fin)
    nf = int(subprocess.run(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v', '-show_entries',
                             'stream=nb_read_packets', '-of', 'csv=p=0', fin], capture_output=True, text=True).stdout.strip())
    print(f'[{time.strftime("%H:%M")}] FINISHED {j["name"]}: {nf} frames (expected {j["n"]}) -> {fin}', flush=True)
    for p in parts:
        os.remove(p)
    return fin


def run(names, budget, workers):
    t0 = time.time(); todo = []
    for j in [BYNAME[n] for n in names] if names else JOBS:
        jd = os.path.join(EXPORT, j['name'])
        if os.path.exists(os.path.join(EXPORT, 'final', j['name'] + '.mov')):
            continue
        for r0, r1 in chunks(j):
            if not os.path.exists(os.path.join(jd, f'seg_{r0:04d}.mov')):
                todo.append((j, r0, r1))
    running = []
    print(f'[{time.strftime("%H:%M")}] {len(todo)} chunks to do', flush=True)
    while todo or running:
        MEMCAP = float(os.environ.get('MEMCAP_GB', '6.2'))   # all processes share one memory limit (OOM kills otherwise)
        while todo and len(running) < workers and (time.time() - t0) / 60 < budget and \
                sum(it[1].get('mem', 2.5) for it in running) + todo[0][0].get('mem', 2.5) <= MEMCAP or (todo and not running and (time.time() - t0) / 60 < budget):
            j, r0, r1 = todo.pop(0); cdir = os.path.join(EXPORT, j['name'], f'c_{r0:04d}')
            k0 = r0                                        # resume inside an interrupted chunk: keep its finished frames
            while k0 < r1 and os.path.exists(frame_path(j, cdir, k0)) and os.path.exists(frame_path(j, cdir, k0 + 1) if k0 + 1 < r1 else frame_path(j, cdir, k0)):
                k0 += 1
            k0 = max(r0, k0 - 1)                           # the last written frame may be truncated: redo it
            if k0 == r0:
                shutil.rmtree(cdir, ignore_errors=True)
            running.append((launch(j, k0, r1, cdir, j['layer']['outs']), j, r0, r1, cdir, time.time()))
            print(f'[{time.strftime("%H:%M")}] start {j["name"]} {r0}-{r1}' + (f' (resuming at {k0})' if k0 > r0 else ''), flush=True)
        if not todo or (time.time() - t0) / 60 >= budget:
            todo = [] if (time.time() - t0) / 60 >= budget else todo
        time.sleep(5)
        for it in list(running):
            p, j, r0, r1, cdir, ts = it
            if p.poll() is None:
                continue
            running.remove(it)
            miss = [k for k in range(r0, r1) if not os.path.exists(frame_path(j, cdir, k))]
            if p.returncode or miss:
                print(f'!! {j["name"]} {r0}-{r1} failed (rc {p.returncode}, {len(miss)} missing) -- see {cdir}/log.txt', flush=True)
                continue
            if j.get('post') == 'blackpoint':
                for k in range(r0, r1):
                    blackpoint(frame_path(j, cdir, k))
            encode_chunk(j, r0, r1, cdir, os.path.join(EXPORT, j['name'], f'seg_{r0:04d}.mov'))
            shutil.rmtree(cdir)
            print(f'[{time.strftime("%H:%M")}] done {j["name"]} {r0}-{r1} in {(time.time() - ts) / 60:.1f} min '
                  f'({(time.time() - ts) / (r1 - r0):.1f} s/frame)', flush=True)
            if all(os.path.exists(os.path.join(EXPORT, j['name'], f'seg_{a:04d}.mov')) for a, _ in chunks(j)):
                finish(j)
    print(f'[{time.strftime("%H:%M")}] queue stopped after {(time.time() - t0) / 60:.0f} min', flush=True)


def verify(name, frames, outs):
    j = BYNAME[name]; vd = os.path.join(EXPORT, 'verify', name); shutil.rmtree(vd, ignore_errors=True)
    procs = []
    for k in frames:
        cd = os.path.join(vd, f'k{k:04d}'); procs.append((launch(j, k, k + 1, cd, outs), k, cd))
    for p, k, cd in procs:
        p.wait()
    cap = cv2.VideoCapture(os.path.join(WORK, 'out', j['preview']))
    rows = []; res = []
    for p, k, cd in procs:
        f = frame_path(j, cd, k); a = cv2.imread(f, -1)
        if a is None:
            print(name, k, 'MISSING (see', cd + '/log.txt)'); continue
        a = (a / 257).astype(np.uint8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, b = cap.read()
        a_ = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
        sa, sb = (cv2.GaussianBlur(cv2.resize(x, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1) for x in (a_, b))
        c = np.corrcoef(sa.ravel(), sb.ravel())[0, 1]; d = np.abs(sa - sb).mean(); m = (sa.mean(), sb.mean())
        res.append((k, round(float(c), 3), round(float(d), 2), round(float(m[0]), 1), round(float(m[1]), 1)))
        rows.append(np.hstack([cv2.resize(a_, (640, 360)), cv2.resize(b, (640, 360))]))
    if rows:
        cv2.imwrite(os.path.join(vd, 'compare.jpg'), np.vstack(rows))
    print(name, 'frame, corr, mean|diff|, mean new, mean preview:', res, flush=True)
    return res


def status():
    for j in JOBS:
        jd = os.path.join(EXPORT, j['name'])
        fin = os.path.exists(os.path.join(EXPORT, 'final', j['name'] + '.mov'))
        segs = sum(os.path.exists(os.path.join(jd, f'seg_{a:04d}.mov')) for a, _ in chunks(j))
        print(f"{j['name']:24s} {'FINAL' if fin else f'{segs}/{len(chunks(j))} chunks'}")


if __name__ == '__main__':
    a = sys.argv[1:]
    def getopt(k, d):
        if k in a:
            i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return v
        return d
    budget = float(getopt('--budget', '1e9')); workers = int(getopt('--workers', '2')); outs = float(getopt('--outs', '1.5'))
    if a[0] == 'run':
        run(a[1:], budget, workers)
    elif a[0] == 'verify':
        n = BYNAME[a[1]]['n']; u0, u1 = BYNAME[a[1]]['used']
        fr = [int(x) for x in a[2].split(',')] if len(a) > 2 else [u0 + (u1 - u0) // 6, (u0 + u1) // 2, u1 - 1 - (u1 - u0) // 6]
        verify(a[1], fr, outs)
    elif a[0] == 'status':
        status()

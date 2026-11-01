"""Final-cut shot table for the 4K / LED export (MV_v2, approved 31 Oct 2026).
Every job renders the SAME frames as its approved 1080p preview (same length, frame i = preview frame i), so the
ProRes drops onto the edit's layer with "Replace Footage". Frames outside `used` (+ handles) are rendered black.
WORK = folder with the pipeline scripts + out/ + depthjobs/ ; A = folder with the original uploads (data_archive/sources).
"""
import os
WORK = os.environ.get('WORK', '/home/claude/mv')
A = os.environ.get('A', '/mnt/user-data/attachments/f42362b3-e77e-5554-9434-580f66a1e90e')
MAIN = dict(size=(4608, 2592), outs=3.6)          # 16:9 at the LED width; crop 144 px top + bottom in AE (4608x2304)
SIDE = dict(size=(832, 1456), outs=832 / 720)     # native LED side panel (design 720x1260)


def P(name):
    """preset file -> list of KEY=VAL (later entries override earlier ones)"""
    for d in (WORK, os.path.join(WORK, 'presets')):
        f = os.path.join(d, name)
        if os.path.exists(f):
            return [t for t in open(f).read().split('#')[0].split() if '=' in t]
    raise FileNotFoundError(name)


FLAT = 'ORBIT= MATTE= TILT=0 YAW=0 DLOHI=0,1 PULSEK=0 NOSPECKS=1 THIN= FIT=cover'.split()
OVL = FLAT + 'TRANS=1 TRANS_MODE=hands IN_XMIN=0 T0ABS=1 OUTK=1.0'.split()
LAND = P('land_wob.env') + [f'GAINFILE={WORK}/data_land_gain2.npy']
RG = 'R_UP=0,0,1'.split()

# kind: how the script is called.  dsc = depth_sheet_clip (STILL prefix, RANGE env); cn = clip_new (PNGDIR, RANGE env);
# dso = depth_sheet_objects (STILL, RANGE); f01 = script outdir f0 f1 (stateless or own warm-up); sil = motion_sil3
JOBS = [
 dict(name='side_rose_transitions', layer=SIDE, n=493, kind='f01', chunk=100, post='blackpoint',
      cmd=['rose916.py', f'{A}/5e893272-Red_rose_SF.obj', f'{A}/a2ae3c42-image.jpg', f'{A}/50b48692-image.jpg', f'{A}/d4d39c86-image.jpg'],
      env=P('rose916_sf.env') + ['R_TRANS=1', 'DESIGN=720x1260'], pat='rose916_{:04d}.png'),
 dict(name='side_rose_loop', layer=SIDE, n=493, kind='f01', chunk=100, post='blackpoint',
      cmd=['rose916.py', f'{A}/5e893272-Red_rose_SF.obj', f'{A}/a2ae3c42-image.jpg', f'{A}/50b48692-image.jpg', f'{A}/d4d39c86-image.jpg'],
      env=P('rose916_sf.env') + ['DESIGN=720x1260'], pat='rose916_{:04d}.png'),
 dict(name='dancer_v7', preview='full_new/dancer_full_1080p_v7.mp4', n=375, used=(1, 369), kind='dsc', chunk=40,
      cmd=['depth_sheet_clip.py', 'out/depth_dancer_v3', '0', '375', 'x.mp4', 'depthjobs/dancer_g75.mp4', '0'],
      env=P('dancer_preset_v6.env') + 'TRANS=1 CLIPLEN=15 T0ABS=1 PITCH=3.6 GAIN=1.0'.split()),
 dict(name='blossom_v4', preview='full_new/blossom_full_1080p_v4.mp4', n=337, kind='cn', chunk=40,
      cmd=['clip_new.py', 'out/blossom_footage_v2.mp4', 'out/depth_blossom2s', '0', '337', 'depthsubj', 'x.mp4'],
      env=P('blossom_preset_trans.env') + ['MATTE_DIR=out/blossom_matte4']),
 dict(name='landscape_transitions', preview='full_new/landscape_loop15_transitions.mp4', n=375, kind='f01', chunk=40,
      cmd=['landLoop_wob.py'], env=LAND + ['LOOPMODE=trans'], pat='f{:04d}.png'),
 dict(name='landscape_seamless', preview='full_new/landscape_loop15_seamless.mp4', n=375, used=(0, 177), kind='f01', chunk=40,
      cmd=['landLoop_wob.py'], env=LAND, pat='f{:04d}.png'),
 dict(name='landscape_wobble', preview='full_new/landscape_wobble_from150.mp4', n=225, f0=150, used=(7, 189), kind='f01', chunk=40,
      cmd=['landLoop_wob.py'], env=LAND + 'LOOPMODE=trans WOB_T0=6.62'.split(), pat='f{:04d}.png'),
 dict(name='curtains_wobble', preview='full_new/curtains_wobble_1m57.mp4', n=445, f0=305, kind='f01', chunk=40,
      cmd=['landLoop_wob.py'], env=LAND + 'LOOPMODE=trans WOB_T0=12.2 NO_IN=1 OUT_MIN=15'.split(), pat='f{:04d}.png'),
 dict(name='snow', preview='full_new/snow_full_1080p.mp4', n=369, used=(24, 368), kind='cn', chunk=40,
      cmd=['clip_new.py', f'{A}/60b9fb00-snow.mp4', 'none', '0', '369', 'snow', 'x.mp4'], env=P('snow_preset.env')),
 dict(name='pigeons', preview='pigeons_full_720p_v2.mp4', n=288, used=(0, 176), kind='dso', chunk=40,
      cmd=['depth_sheet_objects.py', 'out/depth_pigeons_v3', '0', '288', 'x.mp4', f'{A}/b264125e-12772675_1920_1080_25fps.mp4', '0'],
      env=P('pigeon_preset.env') + ['TRANS=1']),
 dict(name='falling_leaf', preview='full_new/leaf_full_720p.mp4', n=97, kind='cn', chunk=49,
      cmd=['clip_new.py', 'out/leaf_footage_25fps_1920.mp4', 'out/depth_leaf', '0', '97', 'leaf', 'x.mp4'], env=P('leaf_preset.env')),
 dict(name='reflections1', preview='full_new/reflections1_full_1080p.mp4', n=94, kind='dsc', chunk=47,
      cmd=['depth_sheet_clip.py', 'out/depth_hflat', '0', '94', 'x.mp4', 'depthjobs/refl1_fc.mp4', '0'],
      env=P('dancer_preset_v6.env') + OVL + 'DARKCUT=0.04,0.12 LUMSCALE=0.86 GAIN=0.95 TWK=4 FLOW_DIR=-0.7,0.7 ODIR=-0.81,0.59 TRANS_IN=1.6,1.6 TRANS_OUT=1.4,1.6 OUTPAD=0.1 CLIPLEN=3.76'.split()),
 dict(name='reflections2', preview='full_new/reflections2_full_1080p.mp4', n=290, used=(6, 266), kind='dsc', chunk=40,
      cmd=['depth_sheet_clip.py', 'out/depth_hflat', '0', '290', 'x.mp4', 'depthjobs/refl2_f.mp4', '0'],
      env=P('dancer_preset_v6.env') + OVL + 'DARKCUT=0.04,0.12 LUMSCALE=0.8 GAIN=1.2 TWK=3 FLOW_DIR=-0.99,0.12 ODIR=-1,0.05 TRANS_IN=2.6,2.4 TRANS_OUT=2.2,2.6 OUTPAD=0.25 CLIPLEN=11.6'.split()),
 dict(name='hands', preview='full_new/hands_full_1080p_v3.mp4', n=209, used=(6, 192), kind='dsc', chunk=40,
      cmd=['depth_sheet_clip.py', 'out/depth_hands', '0', '209', 'x.mp4', 'out/hands_footage_25fps_1920.mp4', '0'], env=P('hands_preset.env')),
 dict(name='couple_dance', preview='full_new/dance_full_1080p.mp4', n=215, used=(3, 204), kind='sil', chunk=101,
      cmd=['motion_sil3.py', f'{A}/e2159d18-coupleshot_2.mp4', '0', '215'], env=P('dance_full.env'), pat='dance_{:04d}.png'),
 dict(name='ripples', preview='full_new/ripples_7s5.mp4', n=188, used=(6, 178), kind='cn', chunk=40,
      cmd=['clip_new.py', 'out/water_footage_25fps_1920.mp4', 'out/depth_water', '0', '188', 'water', 'x.mp4'],
      env=P('water_preset.env') + 'CLIPLEN=7.52 SOFT_K=0.55'.split()),
 dict(name='fireworks', preview='full_new/fireworks_full_720p.mp4', n=370, used=(17, 356), kind='cn', chunk=40,
      cmd=['clip_new.py', 'out/fireworks_footage_25fps_1920.mp4', 'out/depth_fireworks', '0', '370', 'fireworks', 'x.mp4'],
      env=P('fireworks_preset.env')),
 dict(name='rose_bouquet', preview='full_new/rose_LOCKED_1080p.mp4', n=276, used=(1, 271), kind='dsc', chunk=40,
      cmd=['depth_sheet_clip.py', 'out/depth_rose', '0', '276', 'x.mp4', f'{A}/02f87d17-7675962-hd_1920_1080_25fps.mp4', '0'],
      env=P('rose_preset_LOCKED.env')),
 dict(name='leaf3', preview='full_new/leaf3_full_1080p_v3.mp4', n=149, used=(21, 127), kind='cn', chunk=40,
      cmd=['clip_new.py', 'out/leaf3_footage_25fps_1920.mp4', 'out/depth_leaf3', '0', '149', 'scene', 'x.mp4'],
      env=P('leaf3_preset_trans.env')),
 dict(name='ballerina_fix', preview='full_new/ballerina_fix_full_1080p.mp4', n=300, used=(2, 284), kind='dsc', chunk=40,
      cmd=['depth_sheet_clip.py', 'out/depth_balf', '0', '300', 'x.mp4', 'depthjobs/balf_full_g.mp4', '0'], env=P('balf_full.env')),
 dict(name='rose3d_transitions', preview='full_new/rose3d_trans_10s_v2.mp4', n=246, kind='f01', chunk=41,
      cmd=['rose3d_loop.py', f'{A}/6e565724-rosa.obj', f'{A}/7e7846bc-image.jpg'], env=P('rose3d_trans.env'), pat='rose3d_{:04d}.png'),
 dict(name='rose3d_loop', preview='full_new/rose3d_loop2_10s.mp4', n=246, kind='f01', chunk=41,
      cmd=['rose3d_loop.py', f'{A}/6e565724-rosa.obj', f'{A}/7e7846bc-image.jpg'], env=P('rose3d_loop2.env'), pat='rose3d_{:04d}.png'),
 dict(name='fire', preview='full_new/fire_4s20_v2.mp4', n=120, kind='f01', chunk=40,
      cmd=['fire_clip.py', f'{A}/9d8441a2-fire.mp4'], env=P('fire_full.env') + 'GAIN=1.7 HOTW=0.72,0.85 FIRE_TL=4.8'.split(), pat='fire_{:04d}.png'),
]
MEM = dict(blossom_v4=3.6, snow=3.0, ripples=3.0, fireworks=3.0, leaf3=3.0, falling_leaf=2.5, curtains_wobble=3.5,
           landscape_transitions=3.5, landscape_seamless=3.5, landscape_wobble=3.5)       # peak GB per 4K worker (measured / est.)
for j in JOBS:
    j.setdefault('mem', MEM.get(j['name'], 2.8))
    j.setdefault('layer', MAIN); j.setdefault('f0', 0); j.setdefault('used', (0, j['n']))

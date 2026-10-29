"""Hybrid montage: grid displacement during the shots + feedback flow ("pixel propagation") at the handovers.
Each handover: the composite is pushed along a swirling curl-noise flow every frame and fed back, so the outgoing
shot smears into streaks, the incoming shot is fed into the same current, then the flow relaxes and it resolves."""
import numpy as np, cv2, opensimplex
import common
common.FPS = 25
import gridmontage as G
from common import W, H, Writer, bloom, luma

SLOT, TR, NS = G.SLOT, G.TR, G.NS
TOTAL = NS * SLOT + TR
FW, FH = 64, 36                                       # flow field resolution (upsampled)
fx = np.linspace(0, 5.0, FW); fy = np.linspace(0, 2.8, FH)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def flow(t, amp):
    """curl of an animated noise potential -> divergence-free swirling velocity (px/frame)"""
    psi = opensimplex.noise3array(fx, fy, np.array([t * 0.12]))[0] + \
          0.5 * opensimplex.noise3array(fx * 2.1 + 9, fy * 2.1, np.array([t * 0.2]))[0]
    dy, dx = np.gradient(psi)
    vx, vy = dy, -dx
    vx = cv2.resize(vx.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    vy = cv2.resize(vy.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    n = max(np.abs(vx).max(), np.abs(vy).max(), 1e-6)
    return (vx / n * amp).astype(np.float32), (vy / n * amp).astype(np.float32)


def handover_weight(g):
    """0 during the shots, peaks at each handover (incl. the assemble-in and the final dissolve)"""
    w = 0.0
    for k in range(NS + 1):
        c = k * SLOT + TR * 0.5
        w = max(w, np.exp(-((g - c) / 26.0) ** 2))
    return w


if __name__ == '__main__':
    wr = Writer('out/G_flowmontage.mp4', 'out/stills_flow', [20, 60, 112, 130, 150, 240, 250, 370, 500, 520])
    fb = np.zeros((H, W, 3), np.float32)
    for g in range(TOTAL):
        t = g / 25.0
        w = handover_weight(g)
        comp = G.render(g)                            # grid-displaced composite (tiles switching at handovers)
        vx, vy = flow(t, 1.2 + 12.0 * w)              # gentle ripple in the shots, strong current at handovers
        # melt the grid itself along the flow (stronger at handovers)
        k = 1.5 + 10.0 * w
        sc = np.float32(k / max(1.2 + 12.0 * w, 1e-6))
        comp_w = cv2.remap(comp.astype(np.float32), (xx - vx * sc).astype(np.float32), (yy - vy * sc).astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # feedback: advect previous buffer along the flow, decay, inject the current composite
        fb = cv2.remap(fb, (xx - vx).astype(np.float32), (yy - vy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        inj = 0.35 - 0.27 * w                         # small injection at the peak -> long streaks build up
        fb = fb * (1 - inj) + comp_w * inj
        fb = np.maximum(fb, fb * 0)                   # (keep non-negative)
        mix = np.clip(w * 1.1, 0, 0.95)
        out = comp_w * (1 - mix) + fb * mix
        out = 1 - (1 - out) * (1 - fb * 0.25 * w)     # streak glow on top during handovers
        out = bloom(np.clip(out, 0, 1), 0.75, 0.25 + 0.25 * w, 10)
        wr.write(np.clip(out, 0, 1))
    wr.close()

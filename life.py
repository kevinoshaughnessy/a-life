#!/usr/bin/env python3
"""A short animated film: one human life, from dividing cells to a deathbed surrounded by family."""
import math, os, random, sys, subprocess, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool

W, H = 1280, 720            # design space: every scene is laid out on this grid
OUT_W, OUT_H = 1920, 1080  # delivered resolution
OS = OUT_W / W
SS = 4                      # drawing buffer scale (5120x2880), downsampled to 1080p for anti-aliasing
FPS = 30
DUR = 111.5
NF = int(DUR * FPS)
BASE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(BASE, 'assets', 'fonts', 'Poppins-Medium.ttf')


# ---------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x

def lerp(a, b, t):
    return a + (b - a) * t

def ease(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)

def lerpc(c1, c2, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))

def interp(table, x):
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, v0), (x1, v1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            u = (x - x0) / (x1 - x0)
            return lerpc(v0, v1, u) if isinstance(v0, tuple) else lerp(v0, v1, u)

def bump(x, c, height, width):
    return height * math.exp(-((x - c) / width) ** 2)


class Canvas:
    def __init__(self, img):
        self.img = img
        self.d = ImageDraw.Draw(img, 'RGBA')

    def circle(self, x, y, r, fill, outline=None, width=0):
        s = SS
        self.d.ellipse([(x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s], fill=fill,
                       outline=outline, width=max(0, int(round(width * s))))

    def ellipse(self, x, y, rx, ry, fill, outline=None, width=0):
        s = SS
        self.d.ellipse([(x - rx) * s, (y - ry) * s, (x + rx) * s, (y + ry) * s], fill=fill,
                       outline=outline, width=max(0, int(round(width * s))))

    def line(self, pts, width, fill, caps=True):
        s = SS
        self.d.line([(px * s, py * s) for px, py in pts], fill=fill,
                    width=max(1, int(round(width * s))), joint='curve')
        if caps:
            for px, py in (pts[0], pts[-1]):
                self.circle(px, py, width / 2, fill)

    def poly(self, pts, fill):
        self.d.polygon([(px * SS, py * SS) for px, py in pts], fill=fill)

    def rect(self, x0, y0, x1, y1, fill, r=0):
        s = SS
        r = min(r, (x1 - x0) / 2 - 0.5, (y1 - y0) / 2 - 0.5)
        if r > 0.5:
            self.d.rounded_rectangle([x0 * s, y0 * s, x1 * s, y1 * s], radius=r * s, fill=fill)
        else:
            self.d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], fill=fill)

    def arc(self, x, y, rx, ry, a0, a1, fill, width):
        s = SS
        self.d.arc([(x - rx) * s, (y - ry) * s, (x + rx) * s, (y + ry) * s], a0, a1,
                   fill=fill, width=max(1, int(round(width * s))))

    def chord(self, x, y, rx, ry, a0, a1, fill):
        s = SS
        self.d.chord([(x - rx) * s, (y - ry) * s, (x + rx) * s, (y + ry) * s], a0, a1, fill=fill)

    def pieslice(self, x, y, rx, ry, a0, a1, fill):
        s = SS
        self.d.pieslice([(x - rx) * s, (y - ry) * s, (x + rx) * s, (y + ry) * s], a0, a1, fill=fill)

    def text(self, x, y, txt, font, fill, anchor='lm'):
        self.d.text((x * SS, y * SS), txt, font=font, fill=fill, anchor=anchor)


def vgrad(top, bot, h=H * SS, w=W * SS):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    col = np.array(top, np.float32)[None, :] * (1 - t) + np.array(bot, np.float32)[None, :] * t
    return np.broadcast_to(col[:, None, :], (h, w, 3))

def radial(cx, cy, radius, power=2.0, h=H * SS, w=W * SS):
    yy, xx = np.ogrid[0:h, 0:w]
    d = np.sqrt((xx - cx * SS) ** 2 + (yy - cy * SS) ** 2) / (radius * SS)
    return (np.clip(1 - d, 0, 1) ** power).astype(np.float32)

def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def finish(img, zoom=1.0, cx=640, cy=360):
    w, h = W * SS / zoom, H * SS / zoom
    cx = clamp(cx, w / 2 / SS, W - w / 2 / SS)
    cy = clamp(cy, h / 2 / SS, H - h / 2 / SS)
    box = (cx * SS - w / 2, cy * SS - h / 2, cx * SS + w / 2, cy * SS + h / 2)
    return img.resize((OUT_W, OUT_H), Image.LANCZOS, box=box)

_cache = {}
def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


# ---------------------------------------------------------------- the person
SKIN = (238, 196, 166)
SKIN_D = (222, 174, 144)
HEIGHT_CM = [(0, 50), (1, 75), (2, 87), (3, 95), (4, 102), (6, 116), (8, 128), (10, 138), (12, 150),
             (14, 160), (16, 166), (18, 169), (20, 170), (55, 170), (70, 168), (80, 165), (90, 161)]
HEADS = [(1, 4.3), (2, 4.6), (4, 5.1), (6, 5.6), (10, 6.2), (14, 6.8), (18, 7.1), (90, 7.1)]
LEGF = [(1, 0.36), (2, 0.38), (6, 0.43), (12, 0.46), (18, 0.47)]
ARMF = [(1, 0.31), (2, 0.32), (8, 0.35), (18, 0.37)]
SW = [(1, 0.62), (2, 0.64), (10, 0.7), (18, 0.8)]
HW = [(1, 0.6), (10, 0.6), (18, 0.72)]
LEGW = [(1, 0.42), (4, 0.38), (10, 0.34), (18, 0.32)]
ARMW = [(1, 0.32), (4, 0.28), (10, 0.25), (18, 0.23)]
HEADRX = [(1, 0.47), (4, 0.45), (10, 0.42), (18, 0.4)]
EYEY = [(1, 0.13), (6, 0.09), (18, 0.05)]
HAIRLEN = [(1, 0.06), (3, 0.18), (6, 0.45), (14, 0.65), (20, 0.62), (50, 0.5), (62, 0.25), (72, 0.1)]
SHIRT = [(2, (246, 186, 70)), (7, (226, 92, 82)), (13, (72, 136, 190)), (19, (56, 150, 140)),
         (32, (92, 108, 160)), (50, (150, 92, 118)), (68, (176, 146, 112)), (90, (168, 140, 112))]
PANTS = [(2, (92, 146, 206)), (7, (64, 96, 156)), (13, (50, 62, 92)), (19, (54, 58, 74)),
         (32, (58, 58, 70)), (50, (82, 74, 86)), (68, (108, 102, 98)), (90, (108, 102, 98))]
HAIR = [(1, (120, 80, 50)), (40, (98, 64, 42)), (52, (128, 108, 96)), (64, (170, 165, 160)), (76, (230, 228, 224))]
SHOE = [(2, (214, 84, 74)), (10, (90, 80, 100)), (18, (70, 60, 60))]


def height_px(age):
    return interp(HEIGHT_CM, age) * 2.3


def params(age):
    return dict(
        H=height_px(age), heads=interp(HEADS, age), legf=interp(LEGF, age), armf=interp(ARMF, age),
        sw=interp(SW, age), hw=interp(HW, age) + 0.07 * ease((age - 38) / 22),
        legw=interp(LEGW, age), armw=interp(ARMW, age), headrx=interp(HEADRX, age),
        eyey=interp(EYEY, age), hairlen=interp(HAIRLEN, age), shirt=interp(SHIRT, age),
        pants=interp(PANTS, age), hair=interp(HAIR, age), shoe=interp(SHOE, age),
        short=age < 15, glasses=ease((age - 46) / 4), stoop=ease((age - 62) / 24),
        cane=ease((age - 76) / 3), wrinkle=ease((age - 58) / 14), cheek=1 - ease((age - 8) / 8),
        bun=ease((age - 64) / 5), toddler=age < 3.2, bigeyes=age < 6,
    )



RIBBON = (240, 110, 130)

def smooth(pts, n=8):
    """Catmull-Rom through pts."""
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            u = k / n
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * u + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * u * u
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * u ** 3) for j in (0, 1)))
    out.append(pts[-1])
    return out

def hair_cap(c, hcx, hcy, rx, ry, h, col, look=0.0, male=False, part=1):
    """Hair over the top of a front-facing head, with a soft side-swept hairline (no straight fringe)."""
    cx = hcx - look * rx * 0.08
    f = hcx + look * rx * 0.3
    cap = [(cx + rx * 1.09 * math.cos(a), hcy - 0.03 * h + ry * 1.07 * math.sin(a)) for a in np.linspace(math.pi, 2 * math.pi, 30)]
    s_ = part
    if male:
        line = [(cx + s_ * rx * 1.05, hcy - 0.1 * ry), (f + s_ * 0.6 * rx, hcy - 0.5 * ry), (f + s_ * 0.25 * rx, hcy - 0.62 * ry),
                (f - s_ * 0.3 * rx, hcy - 0.5 * ry), (cx - s_ * rx * 1.05, hcy - 0.1 * ry)]
    else:
        line = [(cx + s_ * rx * 1.08, hcy + 0.06 * ry), (f + s_ * 0.74 * rx, hcy - 0.42 * ry), (f + s_ * 0.32 * rx, hcy - 0.66 * ry),
                (f - s_ * 0.12 * rx, hcy - 0.44 * ry), (f - s_ * 0.6 * rx, hcy - 0.24 * ry), (cx - s_ * rx * 1.08, hcy + 0.1 * ry)]
    if line[0][0] < line[-1][0]:
        line = line[::-1]
    c.poly(cap + smooth(line), col)
    hl = tuple(min(255, int(v * 1.18 + 14)) for v in col[:3])
    c.arc(cx - s_ * rx * 0.15, hcy - 0.18 * ry, rx * 0.72, ry * 0.62, 215, 260, hl + (130,), h * 0.03)

def pigtails(c, hcx, hcy, rx, h, col, look=0.0):
    for side in (-1, 1):
        px = hcx + side * rx * 1.13 - look * rx * 0.1
        c.ellipse(px, hcy + 0.16 * h, h * 0.15, h * 0.24, col)
        c.circle(px - side * h * 0.02, hcy - 0.04 * h, h * 0.07, RIBBON)


# ---------------------------------------------------------------- body parts: shaped limbs, hands, shoes, clothes
def tone(col, k):
    """Darker (k < 1) or lighter (k > 1) version of a colour, keeping any alpha."""
    out = tuple(max(0, min(255, int(v * k + (8 if k > 1 else 0)))) for v in col[:3])
    return out + tuple(col[3:])

def seg(c, A, B, wa, wb, col):
    """A tapered capsule from A (half-width wa) to B (half-width wb)."""
    dx, dy = B[0] - A[0], B[1] - A[1]
    L = math.hypot(dx, dy)
    if L < 1e-3:
        c.circle(A[0], A[1], max(wa, wb), col)
        return
    nx, ny = -dy / L, dx / L
    c.poly([(A[0] + nx * wa, A[1] + ny * wa), (B[0] + nx * wb, B[1] + ny * wb),
            (B[0] - nx * wb, B[1] - ny * wb), (A[0] - nx * wa, A[1] - ny * wa)], col)
    c.circle(A[0], A[1], wa, col)
    c.circle(B[0], B[1], wb, col)

def along(A, B, f):
    return (A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f)

def unit(A, B):
    dx, dy = B[0] - A[0], B[1] - A[1]
    L = math.hypot(dx, dy) + 1e-9
    return dx / L, dy / L

def draw_hand(c, T, d, L, w, col, ts=1, fingers=True):
    """A hand centred on T pointing along unit vector d: palm and fingers, a thumb on side ts, finger gaps."""
    ux, uy = d
    nx, ny = -uy, ux
    wrist = (T[0] - ux * L * 0.4, T[1] - uy * L * 0.4)
    tips = (T[0] + ux * L * 0.42, T[1] + uy * L * 0.42)
    tb = (T[0] - ux * L * 0.18 + nx * w * 0.75 * ts, T[1] - uy * L * 0.18 + ny * w * 0.75 * ts)
    tt = (tb[0] + (ux * 0.62 + nx * ts * 0.5) * L * 0.42, tb[1] + (uy * 0.62 + ny * ts * 0.5) * L * 0.42)
    seg(c, tb, tt, w * 0.36, w * 0.28, col)                     # thumb
    seg(c, wrist, tips, w * 0.82, w * 0.74, col)                # palm and fingers
    if fingers and L > 9:
        sh = tone(col, 0.82) + (150,)
        for k in (-1, 0, 1):
            o = k * w * 0.42
            p0 = (T[0] + ux * L * 0.14 + nx * o, T[1] + uy * L * 0.14 + ny * o)
            p1 = (T[0] + ux * L * 0.5 + nx * o, T[1] + uy * L * 0.5 + ny * o)
            c.line([p0, p1], max(0.8, w * 0.07), sh, caps=False)

def arm(c, S, E, T, aw, sleeve, skin, short=False, hand=True, hand_L=None, xb=None, cuff=True):
    """Tapered upper arm and forearm with an elbow, sleeve (long or short, with a hem or cuff), and a hand at T."""
    L = hand_L if hand_L is not None else aw * 2.6
    d = unit(E, T)
    W = (T[0] - d[0] * L * 0.38, T[1] - d[1] * L * 0.38)       # wrist
    if short:
        seg(c, S, E, aw * 0.52, aw * 0.42, skin)
        seg(c, E, W, aw * 0.42, aw * 0.3, skin)
        sl = along(S, E, 0.48)
        seg(c, S, sl, aw * 0.6, aw * 0.56, sleeve)
        n = unit(S, E)
        c.line([(sl[0] - n[1] * aw * 0.56, sl[1] + n[0] * aw * 0.56), (sl[0] + n[1] * aw * 0.56, sl[1] - n[0] * aw * 0.56)],
               max(1, aw * 0.08), tone(sleeve, 0.82), caps=False)
    else:
        seg(c, S, E, aw * 0.56, aw * 0.46, sleeve)
        seg(c, E, W, aw * 0.46, aw * 0.36, sleeve)
        if cuff:
            seg(c, along(E, W, 0.84), W, aw * 0.39, aw * 0.38, tone(sleeve, 0.86))
        # a crease at the inside of the elbow
        n = unit(S, E)
        c.arc(E[0], E[1], aw * 0.3, aw * 0.22, math.degrees(math.atan2(n[1], n[0])) + 60,
              math.degrees(math.atan2(n[1], n[0])) + 130, tone(sleeve, 0.8) + (150,), max(1, aw * 0.06))
    if hand:
        ts = 1
        if xb is not None:
            nx = -d[1]
            ts = 1 if nx * (xb - T[0]) > 0 else -1
        draw_hand(c, T, d, L, aw * 0.5, skin, ts)

def shoe_front(c, cx, ground, frx, side, col):
    """A shoe seen from the front: toe turned slightly out, with a sole and a toe highlight."""
    rot_ellipse(c, cx + side * frx * 0.12, ground - frx * 0.4, frx, frx * 0.46, side * 0.12, col)
    c.arc(cx + side * frx * 0.12, ground - frx * 0.36, frx * 0.98, frx * 0.42, 20, 160, tone(col, 0.6), max(1, frx * 0.12))
    c.ellipse(cx + side * frx * 0.35, ground - frx * 0.58, frx * 0.3, frx * 0.13, tone(col, 1.35) + (160,))

def shoe_side(c, ax, ay, fl, facing, col):
    """A shoe in profile with heel, rounded toe and a sole; ay is the sole line."""
    f = facing
    pts = smooth([(ax - f * fl * 0.28, ay), (ax - f * fl * 0.3, ay - fl * 0.3), (ax - f * fl * 0.05, ay - fl * 0.36),
                  (ax + f * fl * 0.3, ay - fl * 0.28), (ax + f * fl * 0.62, ay - fl * 0.2), (ax + f * fl * 0.76, ay - fl * 0.06),
                  (ax + f * fl * 0.7, ay)], n=5)
    c.poly(pts, col)
    c.line([(ax - f * fl * 0.28, ay - fl * 0.035), (ax + f * fl * 0.72, ay - fl * 0.035)], max(1, fl * 0.08), tone(col, 0.6), caps=False)

def leg_front(c, hipj, knee, ankle, lw, col, trousers=True):
    """Thigh, knee and calf, tapering to the ankle (trousers keep a straight hem)."""
    calf = along(knee, ankle, 0.32)
    seg(c, hipj, knee, lw * 0.56, lw * 0.43, col)
    seg(c, knee, calf, lw * 0.43, lw * 0.46, col)
    seg(c, calf, ankle, lw * 0.46, lw * (0.38 if trousers else 0.28), col)
    if trousers:
        n = unit(knee, ankle)
        c.line([(ankle[0] - n[1] * lw * 0.38, ankle[1] + n[0] * lw * 0.38), (ankle[0] + n[1] * lw * 0.38, ankle[1] - n[0] * lw * 0.38)],
               max(1, lw * 0.07), tone(col, 0.78), caps=False)
        c.line([along(hipj, knee, 0.3), along(knee, ankle, 0.85)], max(0.8, lw * 0.04), tone(col, 0.86) + (120,), caps=False)
    c.arc(knee[0], knee[1], lw * 0.26, lw * 0.14, 200, 340, tone(col, 0.82) + (120,), max(0.8, lw * 0.05))

def torso_outline(xb, sh_y, hip_y, sw, hw, h, female):
    """Shaped torso: sloped shoulders, a waist, and a hem at the hips."""
    waist_y = sh_y + (hip_y - sh_y) * 0.6
    hem_y = hip_y + 0.08 * h
    wx = hw * (0.84 if female else 0.95)
    left = [(xb - hw * 1.04, hem_y), (xb - wx, waist_y), (xb - sw * 0.9, sh_y + 0.46 * h), (xb - sw * 1.0, sh_y + 0.27 * h),
            (xb - sw * 0.93, sh_y + 0.11 * h), (xb - sw * 0.6, sh_y + 0.01 * h), (xb - 0.16 * h, sh_y - 0.03 * h)]
    right = [(2 * xb - p[0], p[1]) for p in left[::-1]]
    return smooth(left, n=5) + smooth(right, n=5), hem_y, waist_y

def clothes_details(c, R, xb, sh_y, hip_y, sw, hw, h, col, collar, hem_y, waist_y, skin_d):
    """Neckline or collar, hem, and a few soft fold lines."""
    dk = tone(col, 0.8)
    if collar == 'shirt':
        c.chord(*R(xb, sh_y - 0.02 * h), h * 0.14, h * 0.12, 0, 180, skin_d)
        lc = tone(col, 1.25)
        for s_ in (-1, 1):
            c.poly([R(xb + s_ * 0.17 * h, sh_y - 0.03 * h), R(xb + s_ * 0.01 * h, sh_y + 0.14 * h),
                    R(xb + s_ * 0.22 * h, sh_y + 0.1 * h)], lc)
        c.line([R(xb, sh_y + 0.13 * h), R(xb, hem_y - 0.02 * h)], max(1, h * 0.012), dk + (160,), caps=False)
        for k in range(4):
            bx, by = R(xb + h * 0.03, sh_y + 0.24 * h + k * (hem_y - sh_y - 0.3 * h) / 3.5)
            c.circle(bx, by, max(0.8, h * 0.017), tone(col, 1.3))
    elif collar == 'cardigan':
        c.poly([R(xb - 0.15 * h, sh_y - 0.02 * h), R(xb + 0.15 * h, sh_y - 0.02 * h), R(xb, sh_y + 0.42 * h)], (236, 228, 214))
        c.line([R(xb - 0.15 * h, sh_y - 0.02 * h), R(xb, sh_y + 0.42 * h), R(xb + 0.15 * h, sh_y - 0.02 * h)],
               max(1, h * 0.03), tone(col, 1.12), caps=False)
        c.line([R(xb, sh_y + 0.42 * h), R(xb, hem_y)], max(1, h * 0.015), dk, caps=False)
        for k in range(3):
            bx, by = R(xb - h * 0.035, sh_y + 0.5 * h + k * 0.28 * h)
            c.circle(bx, by, max(0.8, h * 0.022), tone(col, 0.7))
    elif collar == 'suit':
        c.poly([R(xb - h * 0.16, sh_y), R(xb + h * 0.16, sh_y), R(xb, sh_y + h * 0.55)], (246, 246, 244))
        for s_ in (-1, 1):
            c.poly([R(xb + s_ * 0.17 * h, sh_y - 0.02 * h), R(xb + s_ * 0.02 * h, sh_y + 0.56 * h),
                    R(xb + s_ * 0.24 * h, sh_y + 0.24 * h)], tone(col, 0.82))
    else:  # crew neck
        c.chord(*R(xb, sh_y - 0.03 * h), h * 0.17, h * 0.13, 0, 180, skin_d)
        c.arc(*R(xb, sh_y - 0.03 * h), h * 0.18, h * 0.14, 10, 170, dk, max(1, h * 0.03))
    # hem
    c.line([R(xb - hw * 1.02, hem_y - h * 0.02), R(xb, hem_y), R(xb + hw * 1.02, hem_y - h * 0.02)], max(1, h * 0.025), dk, caps=False)
    # folds: under the arms and at the waist
    fa = dk + (90,)
    for s_ in (-1, 1):
        c.line([R(xb + s_ * sw * 0.86, sh_y + 0.5 * h), R(xb + s_ * sw * 0.55, sh_y + 0.66 * h)], max(1, h * 0.02), fa, caps=False)
        c.line([R(xb + s_ * hw * 0.8, waist_y + 0.1 * h), R(xb + s_ * hw * 0.45, waist_y + 0.2 * h)], max(1, h * 0.02), fa, caps=False)


NOW = 0.0          # film time of the frame being drawn; drives blinking and idle motion

def person_seed(x, male=False):
    return abs(int(round(x / 40.0))) * 1.7 + (3.1 if male else 0.0)

def blink_now(seed):
    p = 3.3 + (seed * 0.37) % 1.9
    return ((NOW + seed * 1.3) % p) < 0.13

def brow_colour(hair):
    return tone(hair, 0.72) if sum(hair[:3]) < 480 else (150, 142, 138)

BROWS = {  # inner dy, middle dy, outer dy, overall lift (in head heights)
    'soft': (0.0, -0.012, 0.004, 0.0), 'happy': (0.0, -0.03, 0.0, -0.02), 'sad': (-0.045, -0.015, 0.03, 0.0),
    'worried': (-0.03, -0.01, 0.015, 0.0), 'surprised': (0.0, -0.03, 0.0, -0.05)}

def draw_person(c, x, ground, age, eyes='open', mouth='smile', hands=None, walk=None, style=None):
    P = params(age)
    if style:
        P.update(style)
    eyes, mouth = P.get('eyes', eyes), P.get('mouth', mouth)
    seed = person_seed(x, P.get('male', False))
    Hh = P['H']
    h = Hh / P['heads']
    stoop, cane = P['stoop'], P['cane']
    lean = math.radians(4.0) * stoop + P.get('wobble', 0.0)
    sway = liftL = liftR = bob = 0.0
    if walk is not None:
        s = math.sin(walk)
        sway = s * h * 0.06
        liftL, liftR = max(0, s) * h * 0.16, max(0, -s) * h * 0.16
        bob = -abs(s) * h * 0.05
    # idle: slow breathing and a gentle shift of weight
    idle = 0.0 if walk is not None else P.get('idle', 1.0)
    breath = math.sin(2 * math.pi * NOW / 3.6 + seed) * idle
    shift = math.sin(2 * math.pi * NOW / 7.3 + seed * 2.3) * idle
    lean += math.radians(0.5) * shift
    sway += shift * h * 0.03
    bob -= (breath * 0.5 + 0.5) * h * 0.02
    cs, sn = math.cos(lean), math.sin(lean)

    def R(px, py):
        dx, dy = px - x, py - ground
        return (x + dx * cs - dy * sn, ground + dx * sn + dy * cs)

    xb = x + sway
    top = ground - Hh + bob + stoop * 0.22 * h
    cy = top + h * 0.5
    rx, ry = h * P['headrx'], h * 0.5
    sh_y = top + h * 1.2
    hip_y = ground - Hh * P['legf'] + bob + (breath * 0.5 + 0.5) * h * 0.02
    sw, hw = h * P['sw'] * (1 - 0.06 * stoop), h * P['hw']
    lw, aw = h * P['legw'], h * P['armw']
    foot_y = ground - lw * 0.35
    shirt, pants, hair = P['shirt'], P['pants'], P['hair']

    if P.get('veil'):
        c.poly([R(xb - rx * 0.9, cy - ry * 0.7), R(xb + rx * 0.9, cy - ry * 0.7),
                R(xb + hw * 2.0, hip_y + 0.4 * h), R(xb - hw * 2.0, hip_y + 0.4 * h)], (255, 255, 255, 110))
    male = P.get('male', False)
    frx = max(lw * 0.62, Hh * 0.034 + 2)
    girl_dress = P.get('girl') and age < 11 and not P.get('nolegs')
    # legs, knees, ankles and shoes
    if not (P.get('nolegs') or P.get('dress')):
        for side, lift in ((-1, liftL), (1, liftR)):
            hipj = R(xb + side * hw * 0.48, hip_y + 0.12 * h)
            knee = R(xb + side * hw * 0.44, hip_y + (ground - hip_y) * 0.5 - lift * 0.5)
            ank = R(x + side * hw * 0.44, ground - frx * 0.62 - lift)
            leg_front(c, hipj, knee, ank, lw, pants, trousers=not girl_dress)
            sx_, sy_ = R(x + side * hw * 0.44, ground - lift)
            shoe_front(c, sx_, sy_, frx, side, P['shoe'])
    c.poly([R(xb - hw, hip_y - 0.1 * h), R(xb + hw, hip_y - 0.1 * h), R(xb + hw * 0.98, hip_y + 0.2 * h),
            R(xb, hip_y + 0.3 * h), R(xb - hw * 0.98, hip_y + 0.2 * h)], pants)
    if girl_dress:                                   # a little girl's dress, flared, with a hem and folds
        knee_y = hip_y + (ground - hip_y) * 0.44
        hem = [R(xb + hw * 1.62 * math.cos(a_), knee_y + h * 0.05 * math.sin(a_)) for a_ in np.linspace(0, math.pi, 12)]
        c.poly([R(xb - hw * 0.98, hip_y - 0.16 * h), R(xb + hw * 0.98, hip_y - 0.16 * h)] + hem, shirt)
        c.line(hem, max(1, h * 0.03), tone(shirt, 0.8), caps=False)
        for k_ in (-0.9, -0.3, 0.3, 0.9):
            c.line([R(xb + k_ * hw * 0.6, hip_y), R(xb + k_ * hw * 1.3, knee_y - 0.02 * h)], max(1, h * 0.018),
                   tone(shirt, 0.84) + (140,), caps=False)
    if P.get('dress'):
        dc = P['dress']
        sk = [R(xb - hw * 0.95, hip_y - 0.2 * h), R(xb + hw * 0.95, hip_y - 0.2 * h)]
        sk += [R(xb + hw * 2.3 * math.cos(a_), ground + 2 + 6 * math.sin(a_)) for a_ in np.linspace(0, math.pi, 14)]
        c.poly(sk, dc)
        for k in (-1, 1):
            c.line([R(xb + k * hw * 0.4, hip_y), R(xb + k * hw * 1.2, ground - 4)], 1.4, tuple(int(v * 0.9) for v in dc[:3]), caps=False)
    # neck, shaped torso, collar, hem and folds
    seg(c, R(xb, cy + 0.3 * h), R(xb, sh_y + 0.05 * h), h * 0.115, h * 0.125, SKIN_D)
    pts, hem_y, waist_y = torso_outline(xb, sh_y, hip_y, sw, hw, h, not male)
    c.poly([R(*p) for p in pts], shirt)
    collar = P.get('collar') or ('suit' if P.get('tie') else 'shirt' if male else
                                 'cardigan' if age >= 64 else 'crew')
    clothes_details(c, R, xb, sh_y, hip_y, sw, hw, h, shirt, collar, hem_y, waist_y, SKIN_D)
    if not male and age >= 13 and not P.get('nolegs_bust_off'):
        bk = ease((age - 13) / 5)
        for s_ in (-1, 1):                               # soft shading under the bust, and light across the top
            bx_, by_ = R(xb + s_ * sw * 0.42, sh_y + 0.44 * h)
            c.arc(bx_, by_, sw * 0.3, h * 0.12 * (0.6 + 0.4 * bk), 20, 160, tone(shirt, 0.84) + (int(120 * bk),), max(1, h * 0.022))
            c.arc(bx_, by_ + h * 0.02, sw * 0.26, h * 0.1, 210, 300 if s_ < 0 else 330, tone(shirt, 1.1) + (int(90 * bk),),
                  max(1, h * 0.02))
    if P.get('tie'):
        c.poly([R(xb - h * 0.05, sh_y + 0.06 * h), R(xb + h * 0.05, sh_y + 0.06 * h), R(xb + h * 0.06, sh_y + 0.5 * h),
                R(xb, sh_y + 0.6 * h), R(xb - h * 0.06, sh_y + 0.5 * h)], P['tie'])
        bx, by = R(xb - sw * 0.55, sh_y + 0.35 * h)
        flower(c, bx, by, h / 75)

    # arms (and cane)
    joints = {}
    armlen = Hh * P['armf']
    hand_L = h * (0.52 if age < 6 else 0.6)
    for side in (-1, 1):
        S = R(xb + side * (sw - aw * 0.5), sh_y + aw * 0.6)
        T = R(xb + side * (sw + h * 0.1), sh_y + aw * 0.6 + armlen * 0.97)
        if P.get('arms_up'):
            T = R(xb + side * (sw + h * 0.5), sh_y - h * 0.05)
        if P.get('bouquet'):
            T = R(xb + side * h * 0.14, hip_y - 0.22 * h)
        elbow_down = False
        if hands and side in hands:
            T = hands[side]
            if len(T) == 3:
                elbow_down, T = True, T[:2]
        if side == 1 and cane > 0:
            ch = R(xb + sw + h * 0.6, hip_y + 0.02 * h)
            T = (lerp(T[0], ch[0], cane), lerp(T[1], ch[1], cane))
        d = math.hypot(T[0] - S[0], T[1] - S[1]) + 1e-6
        bend = math.sqrt(max(0.0, (armlen / 2) ** 2 - (d / 2) ** 2))
        px, py = -(T[1] - S[1]) / d, (T[0] - S[0]) / d
        if (py < 0) if elbow_down else (px * side < 0):
            px, py = -px, -py
        E = ((S[0] + T[0]) / 2 + px * bend * 0.8, (S[1] + T[1]) / 2 + py * bend * 0.8)
        if side == 1 and cane > 0:
            col = (104, 74, 52, int(255 * cane))
            top_c = (T[0], T[1] - aw * 0.3)
            c.line([top_c, (T[0] + h * 0.05, ground)], h * 0.085, col, caps=False)
            c.arc(T[0] - h * 0.13, T[1] - aw * 0.3, h * 0.13, h * 0.11, 180, 360, col, h * 0.085)
        arm(c, S, E, T, aw, shirt, SKIN, short=P['short'], hand_L=hand_L, xb=xb)
        joints[side] = (S, E, T, aw, SKIN if P['short'] else shirt)
        joints[('arm', side)] = dict(E=E, T=T, aw=aw, sleeve=shirt, skin=SKIN, short=P['short'], L=hand_L, xb=xb)
    joints.update(sh_y=sh_y, hip_y=hip_y, h=h, xb=xb, sw=sw, top=top)
    if P.get('bouquet'):
        bx, by = R(xb, hip_y - 0.36 * h)
        for a in np.linspace(0, 2 * math.pi, 7)[:-1]:
            c.circle(bx + math.cos(a) * h * 0.2, by + math.sin(a) * h * 0.14 + h * 0.04, h * 0.1, (110, 160, 110))
        rng = random.Random(4)
        for k in range(11):
            a, rr = rng.uniform(0, 2 * math.pi), rng.uniform(0, h * 0.2)
            c.circle(bx + math.cos(a) * rr, by + math.sin(a) * rr * 0.7, h * 0.085,
                     [(250, 200, 210), (255, 255, 250), (246, 170, 150), (250, 226, 160)][k % 4])
        c.line([(bx - h * 0.05, by + h * 0.15), (bx, by + h * 0.45)], h * 0.05, (110, 150, 100))
        flower(c, bx, by - h * 0.02, h / 50)

    # head
    hcx, hcy = R(xb, cy)
    if P['hairlen'] > 0:
        c.ellipse(hcx, hcy - 0.03 * h, rx * 1.12, ry * 1.06, hair)
        c.rect(hcx - rx * 1.12, hcy, hcx + rx * 1.12, hcy + P['hairlen'] * h, hair, r=rx * 0.5)
    if P['bun'] > 0:
        c.circle(hcx, hcy - ry * 0.98, h * 0.2, hair + (int(255 * P['bun']),))
    look = clamp(P.get('look', 0.0), -1, 1)
    ldown = P.get('lookdown', 0.0)
    fx = hcx + look * rx * 0.32                 # where the face points
    if P.get('girl') and 1.6 <= age < 10:
        pigtails(c, hcx, hcy, rx, h, hair, look)
    for side in (-1, 1):
        if look * side > 0.35:                  # far ear hidden as the head turns
            continue
        c.circle(hcx + side * rx * (0.97 - 0.2 * abs(look)) - look * rx * 0.1, hcy + 0.06 * h, h * 0.085, SKIN_D)
    if abs(look) > 0.03:                        # back of the head, peeking out behind the face
        c.ellipse(hcx - look * rx * 0.2, hcy - 0.05 * h, rx * 1.04, ry * 0.98, hair)
    c.ellipse(hcx, hcy, rx, ry, SKIN)
    hair_cap(c, hcx, hcy, rx, ry, h, hair, look, male=P.get('male', False), part=P.get('part', 1))
    if P.get('girl') and age < 1.6:
        c.circle(hcx + rx * 0.45, hcy - ry * 0.8, h * 0.06, RIBBON)
    if P.get('veil'):
        for k in range(7):
            a = math.pi * (1.05 + 0.9 * k / 6)
            c.circle(hcx + math.cos(a) * rx * 1.02, hcy - 0.04 * h + math.sin(a) * ry * 0.98, h * 0.065,
                     [(255, 255, 250), (250, 200, 210)][k % 2])
    if P.get('hat'):
        hc = P['hat']
        c.poly([(hcx - rx * 0.62, hcy - ry * 0.62), (hcx + rx * 0.62, hcy - ry * 0.62), (hcx + rx * 0.1, hcy - ry * 1.9)], hc)
        c.line([(hcx - rx * 0.4, hcy - ry * 0.95), (hcx + rx * 0.45, hcy - ry * 1.0)], h * 0.04, (255, 255, 255, 200), caps=False)
        c.circle(hcx + rx * 0.1, hcy - ry * 1.9, h * 0.07, (255, 240, 160))

    ey = hcy + P['eyey'] * h + ldown * 0.07 * h
    eox = rx * 0.42 * (1 - 0.22 * abs(look))
    er = h * (0.052 if P['bigeyes'] else 0.042)
    ink = (58, 42, 42)
    brow = P.get('brow') or ('sad' if (mouth in ('neutral', 'sad') and eyes == 'closed') else
                             'happy' if mouth == 'big' else 'surprised' if mouth == 'o' else 'soft')
    bi, bm, bo, lift_ = BROWS[brow]
    bcol = brow_colour(hair)
    blink = eyes == 'open' and blink_now(seed)
    for side in (-1, 1):
        exx = fx + side * eox
        if eyes == 'happy':                     # eyes squeezed into happy arcs
            c.arc(exx, ey + h * 0.025, h * 0.06, h * 0.05, 200, 340, ink, h * 0.026)
        elif blink:
            c.line([(exx - h * 0.055, ey + h * 0.005), (exx + h * 0.055, ey + h * 0.005)], h * 0.024, ink)
        elif eyes == 'open' and ldown < 0.5:
            c.circle(exx, ey, er, ink)
            c.circle(exx + er * 0.35, ey - er * 0.35, er * 0.3, (255, 255, 255, 200))
        else:  # closed / downcast
            c.arc(exx, ey - h * 0.01, h * 0.065, h * 0.035, 15, 165, ink, h * 0.022)
        if P['cheek'] > 0.02:
            c.circle(fx + side * rx * 0.6, ey + 0.13 * h, h * 0.075, (240, 130, 120, int(90 * P['cheek'] + 25)))
        if P['wrinkle'] > 0:
            wa = int(140 * P['wrinkle'])
            c.arc(exx + side * h * 0.12, ey, h * 0.05, h * 0.05, 300 if side > 0 else 150, 60 if side > 0 else 210,
                  (150, 110, 100, wa), h * 0.014)
        by = ey - h * (0.135 if P['bigeyes'] else 0.125) + lift_ * h
        c.line(smooth([(exx - side * h * 0.07, by + bi * h), (exx, by + bm * h), (exx + side * h * 0.075, by + bo * h)], n=4),
               h * (0.02 if age < 4 else 0.026), bcol + ((170,) if age < 4 else ()))
    if abs(look) > 0.15:                        # nose shows in three-quarter view
        nx_ = fx + look * rx * 0.25
        c.arc(nx_, ey + 0.1 * h, h * 0.045, h * 0.05, 270 if look > 0 else 90, 450 if look > 0 else 270,
              (196, 140, 116), h * 0.02)
    else:
        c.arc(fx, ey + 0.085 * h, h * 0.03, h * 0.025, 30, 150, (206, 150, 126, 190), h * 0.018)
    my = ey + 0.2 * h
    mcol = (160, 84, 84)
    if mouth == 'smile':
        c.arc(fx, my - 0.03 * h, h * 0.09, h * 0.06, 20, 160, mcol, h * 0.026)
    elif mouth == 'big':                        # open, laughing smile
        c.chord(fx, my - 0.05 * h, h * 0.1, h * 0.09, 0, 180, (126, 52, 60))
        c.rect(fx - h * 0.075, my - 0.05 * h, fx + h * 0.075, my - 0.03 * h, (252, 248, 244))
        c.ellipse(fx, my + 0.012 * h, h * 0.045, h * 0.02, (226, 124, 130))
    elif mouth == 'sad':
        c.arc(fx, my + 0.04 * h, h * 0.075, h * 0.055, 205, 335, mcol, h * 0.026)
    elif mouth == 'o':
        c.ellipse(fx, my, h * 0.035, h * 0.045, (126, 52, 60))
    elif mouth == 'blow':
        c.circle(fx, my, h * 0.035, (150, 76, 76))
    else:
        c.line([(fx - 0.05 * h, my), (fx + 0.05 * h, my)], h * 0.022, (160, 96, 92))
    if P['glasses'] > 0:
        gc = (70, 58, 58, int(255 * P['glasses']))
        for side in (-1, 1):
            c.circle(fx + side * eox, ey, h * 0.105, None, outline=gc, width=h * 0.022)
        c.line([(fx - eox + h * 0.1, ey), (fx + eox - h * 0.1, ey)], h * 0.02, gc, caps=False)
    joints.update(hcx=hcx, hcy=hcy, ey=ey, rx=rx, fx=fx)
    return joints




def redraw_forearm(c, j, side):
    a = j[('arm', side)]
    E, T, aw, L = a['E'], a['T'], a['aw'], a['L']
    d = unit(E, T)
    W = (T[0] - d[0] * L * 0.38, T[1] - d[1] * L * 0.38)
    if a['short']:
        seg(c, E, W, aw * 0.42, aw * 0.3, a['skin'])
    else:
        seg(c, E, W, aw * 0.46, aw * 0.36, a['sleeve'])
        seg(c, along(E, W, 0.84), W, aw * 0.39, aw * 0.38, tone(a['sleeve'], 0.86))
    redraw_hand(c, j, side)

def redraw_hand(c, j, side):
    a = j[('arm', side)]
    d = unit(a['E'], a['T'])
    ts = 1 if -d[1] * (a['xb'] - a['T'][0]) > 0 else -1
    draw_hand(c, a['T'], d, a['L'], a['aw'] * 0.5, a['skin'], ts)

def draw_person_back(c, x, ground, age, hands=None, style=None):
    """A figure seen from behind (facing into the scene)."""
    P = params(age)
    if style:
        P.update(style)
    Hh = P['H']
    h = Hh / P['heads']
    top = ground - Hh
    cy = top + h * 0.5
    rx, ry = h * P['headrx'], h * 0.5
    sh_y = top + h * 1.2
    hip_y = ground - Hh * P['legf']
    sw, hw = h * P['sw'], h * P['hw']
    lw, aw = h * P['legw'], h * P['armw']
    shirt, pants, hair = P['shirt'], P['pants'], P['hair']
    I = lambda px_, py_: (px_, py_)
    frx = max(lw * 0.62, Hh * 0.034 + 2)
    for side in (-1, 1):
        hipj, knee = (x + side * hw * 0.48, hip_y + 0.12 * h), (x + side * hw * 0.44, hip_y + (ground - hip_y) * 0.5)
        ank = (x + side * hw * 0.44, ground - frx * 0.62)
        leg_front(c, hipj, knee, ank, lw, pants)
        rot_ellipse(c, x + side * hw * 0.44, ground - frx * 0.38, frx * 0.72, frx * 0.42, 0, P['shoe'])   # heels
    c.poly([(x - hw, hip_y - 0.1 * h), (x + hw, hip_y - 0.1 * h), (x + hw * 0.98, hip_y + 0.2 * h),
            (x, hip_y + 0.3 * h), (x - hw * 0.98, hip_y + 0.2 * h)], pants)
    # arms reach forward, so the torso hides them unless they are raised
    joints = {}
    armlen = Hh * P['armf']
    for side in (-1, 1):
        S = (x + side * (sw - aw * 0.5), sh_y + aw * 0.6)
        T = (hands or {}).get(side, (x + side * (sw + h * 0.1), sh_y + aw * 0.6 + armlen * 0.97))
        d = math.hypot(T[0] - S[0], T[1] - S[1]) + 1e-6
        bend = math.sqrt(max(0.0, (armlen / 2) ** 2 - (d / 2) ** 2))
        px, py = -(T[1] - S[1]) / d, (T[0] - S[0]) / d
        if px * side < 0:
            px, py = -px, -py
        E = ((S[0] + T[0]) / 2 + px * bend * 0.8, (S[1] + T[1]) / 2 + py * bend * 0.8)
        arm(c, S, E, T, aw, shirt, SKIN, short=P['short'], hand_L=h * 0.6, xb=2 * x - T[0])
        joints[side] = (S, E, T, aw)
    seg(c, (x, cy + 0.3 * h), (x, sh_y + 0.05 * h), h * 0.115, h * 0.125, SKIN_D)
    pts, hem_y, waist_y = torso_outline(x, sh_y, hip_y, sw, hw, h, not P.get('male', False))
    c.poly(pts, shirt)
    dk = tone(shirt, 0.8)
    c.arc(x, sh_y - 0.06 * h, h * 0.17, h * 0.09, 20, 160, dk, max(1, h * 0.03))               # collar at the back
    c.line([(x - hw * 1.02, hem_y - h * 0.02), (x, hem_y), (x + hw * 1.02, hem_y - h * 0.02)], max(1, h * 0.025), dk, caps=False)
    for s_ in (-1, 1):                                                                              # shoulder-blade folds
        c.line([(x + s_ * sw * 0.55, sh_y + 0.3 * h), (x + s_ * sw * 0.3, sh_y + 0.55 * h)], max(1, h * 0.02), dk + (90,), caps=False)
        c.line([(x + s_ * hw * 0.8, waist_y + 0.1 * h), (x + s_ * hw * 0.45, waist_y + 0.2 * h)], max(1, h * 0.02), dk + (90,), caps=False)
    for side in (-1, 1):
        if P['hairlen'] <= 0:
            c.circle(x + side * rx * 0.97, cy + 0.06 * h, h * 0.085, SKIN_D)
    c.ellipse(x, cy, rx, ry, SKIN_D)
    if P['hairlen'] > 0:
        c.ellipse(x, cy - 0.03 * h, rx * 1.12, ry * 1.06, hair)
        c.rect(x - rx * 1.12, cy, x + rx * 1.12, cy + P['hairlen'] * h, hair, r=rx * 0.5)
    else:
        c.ellipse(x, cy - 0.08 * h, rx * 1.04, ry * 0.93, hair)
    return joints


# ---------------------------------------------------------------- side-view walk cycle
def _dark(col, k=0.82):
    return tuple(int(v * k) for v in col[:3])

def draw_walker(c, x, ground, age, ph, facing=1, style=None, arms_up=False, carry=False, stride=None,
                near_hand=None, far_hand=None, bump=0.0, kick=0.0, out_geom=None):
    """A figure seen from the side walking towards `facing`. Feet are planted on the floor during
    each step (two-bone leg IK), so the body moves over them instead of hovering."""
    P = params(age)
    if style:
        P.update(style)
    Hh = P['H']
    h = Hh / P['heads']
    lw, aw = h * P['legw'], h * P['armw']
    depth = h * P['sw'] * 0.62
    shirt, pants, hair = P['shirt'], P['pants'], P['hair']
    armlen = Hh * P['armf']
    sleeve = SKIN if P['short'] else shirt
    fl = max(Hh * 0.075, lw * 1.25)                      # foot length grows with height
    ah = fl * 0.3                                        # ankle height above the floor
    stand_hip = ground - Hh * P['legf']
    Lmax = (ground - ah) - stand_hip                     # hip-to-ankle when standing straight
    thigh = shin = Lmax * 0.5 * 1.002
    stride = stride if stride is not None else Lmax * 0.55
    lift = Lmax * 0.16
    hx = x - facing * depth * 0.1

    feet = []
    for off in (0.0, 0.5):                               # near leg, far leg
        sph = (ph / (2 * math.pi) + off) % 1.0
        if sph < 0.6:                                    # stance: planted, sliding back under the body
            u = sph / 0.6
            feet.append((hx + facing * stride * (0.5 - u), ground - ah, True))
        else:                                            # swing: lifted forward
            u = (sph - 0.6) / 0.4
            feet.append((hx + facing * stride * (-0.5 + ease(u)), ground - ah - lift * math.sin(math.pi * u), False))
    dxm = max([abs(fx - hx) for fx, fy, st in feet if st] + [0.0])
    L = (thigh + shin) * 0.995
    hip_y = (ground - ah) - math.sqrt(max(L * L - dxm * dxm, (0.8 * L) ** 2))
    drop = hip_y - stand_hip
    seed = person_seed(x, P.get('male', False))
    top = ground - Hh + drop - (math.sin(2 * math.pi * NOW / 3.6 + seed) * 0.5 + 0.5) * h * 0.02 * P.get('idle', 0.0)
    cy, sh_y = top + 0.5 * h, top + 1.2 * h

    def leg(foot, far):
        fx, fy, _ = foot
        dx, dy = fx - hx, fy - hip_y
        d = math.hypot(dx, dy)
        if d > thigh + shin - 0.5:                       # out of reach: pull the foot in
            k = (thigh + shin - 0.5) / d
            fx, fy, dx, dy, d = hx + dx * k, hip_y + dy * k, dx * k, dy * k, thigh + shin - 0.5
        ca = clamp((thigh * thigh + d * d - shin * shin) / (2 * thigh * d), -1, 1)
        ang = math.acos(ca)
        base = math.atan2(dy, dx)
        a1 = base - ang * facing                          # bend the knee forwards
        kx, ky = hx + thigh * math.cos(a1), hip_y + thigh * math.sin(a1)
        if (kx - hx) * facing < (fx - hx) * facing * 0.5 - 1:
            a1 = base + ang * facing
            kx, ky = hx + thigh * math.cos(a1), hip_y + thigh * math.sin(a1)
        col = _dark(pants) if far else pants
        girlish = P.get('girl') and age < 11
        leg_front(c, (hx, hip_y), (kx, ky), (fx, fy), lw, col, trousers=not girlish)
        sc = _dark(P['shoe']) if far else P['shoe']
        shoe_side(c, fx, fy + ah, fl, facing, sc)

    def arm(swing, far):
        S = (x - facing * depth * 0.05, sh_y + aw * 0.6)
        a = 0.38 * swing
        if arms_up:
            a = 1.75 + 0.18 * swing
        if carry and not far:
            a = 0.06 * swing
        up, lo = armlen * 0.5, armlen * 0.5
        E = (S[0] + facing * up * math.sin(a), S[1] + up * math.cos(a))
        a2 = a + (0.15 if carry and not far else 0.35)
        T = (E[0] + facing * lo * math.sin(a2), E[1] + lo * math.cos(a2))
        tgt = far_hand if far else near_hand
        if callable(tgt):
            tgt = tgt(geom_live)
        if tgt is not None:                              # reach for a point; the elbow hangs down
            T = tgt
            d = math.hypot(T[0] - S[0], T[1] - S[1]) + 1e-6
            if d > armlen * 0.98:
                T = (S[0] + (T[0] - S[0]) * armlen * 0.98 / d, S[1] + (T[1] - S[1]) * armlen * 0.98 / d)
                d = armlen * 0.98
            bend = math.sqrt(max(0.0, (armlen / 2) ** 2 - (d / 2) ** 2))
            px_, py_ = -(T[1] - S[1]) / d, (T[0] - S[0]) / d
            if py_ < 0:
                px_, py_ = -px_, -py_
            E = ((S[0] + T[0]) / 2 + px_ * bend, (S[1] + T[1]) / 2 + py_ * bend)
        k = 0.82 if far else 1.0
        globals()['arm'](c, S, E, T, aw, _dark(shirt, k), _dark(SKIN, 0.9 if far else 1.0), short=P['short'],
                         hand_L=h * (0.52 if age < 6 else 0.6), xb=T[0] - facing * 10)
        return T

    geom_live = {}
    near_swing = math.sin(ph)                          # arms swing opposite to the legs
    arm(near_swing, True)
    leg(feet[1], True)
    leg(feet[0], False)
    c.rect(x - depth * 0.9, hip_y - 0.1 * h, x + depth * 0.9, hip_y + 0.28 * h, pants, r=depth * 0.4)
    f_ = facing
    waist_y = sh_y + (hip_y - sh_y) * 0.6
    female_adult = (not P.get('male', False)) and age >= 13
    bust = ease((age - 13) / 5) if female_adult else 0.0
    X = lambda dx, y: (x + f_ * dx, y)
    back = [X(-depth * 0.4, sh_y - 0.03 * h), X(-depth * 0.95, sh_y + 0.2 * h), X(-depth * 0.86, waist_y),
            X(-depth * 1.0, hip_y + 0.08 * h)]
    if bump > 0:
        # a round belly whose fullest point sits just below the navel; the top falls from the bust onto it
        bcx, bcy = x + f_ * depth * 0.95, sh_y + (hip_y - sh_y) * 0.68
        brx, bry = depth * 1.42, (hip_y - sh_y) * 0.39
        g_bel = dict(bcx=bcx, bcy=bcy, brx=brx, bry=bry, f=f_)
        arc_ = [(bcx + f_ * brx * math.cos(t_), bcy + bry * math.sin(t_)) for t_ in np.linspace(-1.25, 1.05, 22)]
        front = [X(depth * 0.88, hip_y + 0.16 * h)] + arc_[::-1] + [
            X(depth * (1.0 + 0.4 * bust), sh_y + 0.46 * h), X(depth * 0.92, sh_y + 0.16 * h), X(depth * 0.5, sh_y - 0.02 * h)]
        torso = smooth(back, n=5) + smooth([back[-1], X(-depth * 0.2, hip_y + 0.16 * h), X(depth * 0.88, hip_y + 0.16 * h)], n=4)[1:] \
            + smooth(front, n=3)[1:]
    else:
        g_bel = None
        front = [X(depth * 0.98, hip_y + 0.08 * h), X(depth * 0.84, waist_y)]
        if bust > 0:
            front += [X(depth * (0.9 + 0.04 * bust), sh_y + 0.66 * h), X(depth * (1.0 + 0.36 * bust), sh_y + 0.46 * h),
                      X(depth * 0.96, sh_y + 0.22 * h)]
        else:
            front += [X(depth * 1.02, sh_y + 0.36 * h)]
        front += [X(depth * 0.5, sh_y - 0.02 * h)]
        torso = smooth(back + front, n=5)
    c.poly(torso, shirt)
    if bump > 0:
        bcx, bcy, brx, bry = g_bel['bcx'], g_bel['bcy'], g_bel['brx'], g_bel['bry']
        shade = [(bcx + f_ * brx * math.cos(t_), bcy + bry * math.sin(t_)) for t_ in np.linspace(0.2, 1.05, 10)]
        shade += [(bcx + f_ * brx * 0.62 * math.cos(t_), bcy + bry * 0.86 * math.sin(t_)) for t_ in np.linspace(1.05, 0.2, 10)]
        c.poly(shade, tone(shirt, 0.88) + (150,))                          # the underside of the belly in shadow
        c.arc(bcx - f_ * brx * 0.06, bcy - bry * 0.04, brx * 0.86, bry * 0.86, -70 if f_ > 0 else 180,
              -15 if f_ > 0 else 250, tone(shirt, 1.12) + (150,), max(1, h * 0.025))     # light on the top of the curve
        hem = [(bcx + f_ * brx * 0.98 * math.cos(t_), bcy + bry * 0.98 * math.sin(t_)) for t_ in np.linspace(0.75, 1.0, 5)]
        c.line(smooth([X(-depth * 0.95, hip_y + 0.1 * h), X(depth * 0.3, hip_y + 0.18 * h)] + hem[::-1], n=4),
               max(1, h * 0.025), tone(shirt, 0.8), caps=False)                         # hem, following the curve
        if kick > 0.02:
            t_k = -0.55
            kx_, ky_ = bcx + f_ * brx * math.cos(t_k), bcy + bry * math.sin(t_k)
            r_ = h * 0.1 * kick
            c.circle(kx_ + f_ * r_ * 0.2 + 1.5, ky_ + 2.5, r_, tone(shirt, 0.86))
            c.circle(kx_ + f_ * r_ * 0.2, ky_, r_, shirt)
    else:
        c.line(smooth([X(-depth * 1.0, hip_y + 0.06 * h), X(0, hip_y + 0.09 * h), X(depth * 0.98, hip_y + 0.06 * h)], n=4),
               max(1, h * 0.025), tone(shirt, 0.8), caps=False)
    if bust > 0:                                         # a soft fold under the bust
        c.arc(*X(depth * (0.62 + 0.2 * bust), sh_y + 0.5 * h), depth * 0.42, h * 0.14, 20 if f_ > 0 else 70,
              110 if f_ > 0 else 160, tone(shirt, 0.82) + (130,), max(1, h * 0.02))
    else:
        c.line([X(-depth * 0.2, sh_y + 0.5 * h), X(depth * 0.3, sh_y + 0.62 * h)], max(1, h * 0.02),
               tone(shirt, 0.8) + (90,), caps=False)
    geom_live.update(belly=g_bel, sh_y=sh_y, hip_y=hip_y, h=h, depth=depth)
    if out_geom is not None:
        out_geom.update(geom_live)
    if P.get('girl') and age < 11:
        knee_y = hip_y + (ground - hip_y) * 0.42
        c.poly([(x - depth * 0.95, hip_y - 0.14 * h), (x + depth * 0.95, hip_y - 0.14 * h),
                (x + depth * 1.55, knee_y), (x - depth * 1.55, knee_y)], shirt)
    c.line([(x + facing * h * 0.03, cy + 0.3 * h), (x, sh_y + 0.08 * h)], h * 0.22, SKIN_D)
    # head in profile
    rx, ry = h * P['headrx'] * 0.98, h * 0.5
    hx2, hy = x + facing * h * 0.06, cy
    if P['hairlen'] > 0.2 and not (P.get('girl') and age < 10):
        xa, xb = hx2 - facing * rx * 1.08, hx2 + facing * rx * 0.15
        c.rect(min(xa, xb), hy - 0.1 * h, max(xa, xb), hy + P['hairlen'] * h, hair, r=rx * 0.4)
    if P.get('girl') and 1.6 <= age < 10:
        c.ellipse(hx2 - facing * rx * 1.05, hy + 0.12 * h, h * 0.16, h * 0.24, hair)
    if P.get('bun', 0) > 0:
        c.circle(hx2 - facing * rx * 0.7, hy - ry * 0.75, h * 0.2, hair + (int(255 * P['bun']),))
    c.ellipse(hx2, hy, rx, ry, SKIN)
    c.circle(hx2 + facing * rx * 0.96, hy + 0.08 * h, h * 0.07, SKIN)          # nose
    # hair: covers the back and top, with a curved hairline at the forehead and ear
    cap = [(math.cos(a) * rx * 1.07, -0.03 * h + math.sin(a) * ry * 1.06)
           for a in np.linspace(math.radians(118), math.radians(318), 26)]
    edge = smooth([(rx * 0.72, -ry * 0.62), (rx * 0.42, -ry * 0.42), (rx * 0.1, -ry * 0.12),
                   (-rx * 0.12, ry * 0.18), (-rx * 0.5, ry * 0.62)])
    pts = [(hx2 + facing * px_, hy + py_) for px_, py_ in cap + edge]
    c.poly(pts, hair)
    if P.get('girl') and age < 10:
        c.circle(hx2 - facing * rx * (0.86 if age >= 1.6 else 0.3), hy - (0.06 if age >= 1.6 else 0.42) * h, h * 0.07, RIBBON)
    c.circle(hx2 - facing * rx * 0.12, hy + 0.06 * h, h * 0.08, SKIN_D)      # ear
    ey = hy + P['eyey'] * h
    exw = hx2 + facing * rx * 0.52
    w_mouth, w_eyes = P.get('mouth', 'smile'), P.get('eyes', 'open')
    if w_eyes == 'happy':
        c.arc(exw, ey + h * 0.025, h * 0.055, h * 0.05, 200, 340, (58, 42, 42), h * 0.026)
    elif w_eyes == 'closed' or blink_now(seed):
        c.line([(exw - h * 0.05, ey + h * 0.005), (exw + h * 0.05, ey + h * 0.005)], h * 0.024, (58, 42, 42))
    else:
        c.circle(exw, ey, h * (0.05 if P['bigeyes'] else 0.04), (58, 42, 42))
    w_brow = P.get('brow') or ('happy' if w_mouth == 'big' else 'surprised' if w_mouth == 'o' else 'soft')
    bi, bm, bo, lift_ = BROWS[w_brow]
    by = ey - h * 0.13 + lift_ * h
    c.line(smooth([(exw + facing * h * 0.075, by + bi * h), (exw, by + bm * h), (exw - facing * h * 0.07, by + bo * h)], n=4),
           h * (0.02 if age < 4 else 0.026), brow_colour(hair))
    mxw, myw = hx2 + facing * rx * 0.66, ey + 0.2 * h
    if w_mouth == 'big':
        c.chord(mxw, myw - 0.045 * h, h * 0.075, h * 0.085, 0, 180, (126, 52, 60))
        c.rect(mxw - h * 0.05, myw - 0.045 * h, mxw + h * 0.055, myw - 0.028 * h, (252, 248, 244))
    elif w_mouth == 'o':
        c.ellipse(mxw + facing * h * 0.02, myw, h * 0.03, h * 0.042, (126, 52, 60))
    else:
        c.arc(hx2 + facing * rx * 0.58, ey + 0.17 * h, h * 0.07, h * 0.05, 20 if facing > 0 else 90, 90 if facing > 0 else 160,
              (160, 84, 84), h * 0.024)
    if P['cheek'] > 0.02:
        c.circle(hx2 + facing * rx * 0.42, ey + 0.13 * h, h * 0.07, (240, 130, 120, int(90 * P['cheek'] + 25)))
    if P['glasses'] > 0:
        gc = (70, 58, 58, int(255 * P['glasses']))
        c.circle(hx2 + facing * rx * 0.55, ey, h * 0.1, None, outline=gc, width=h * 0.022)
        c.line([(hx2 + facing * rx * 0.45, ey), (hx2 - facing * rx * 0.1, ey)], h * 0.02, gc, caps=False)
    T = arm(-near_swing, False)
    if carry:
        c.rect(T[0] - 22, T[1] + 8, T[0] + 22, T[1] + 48, (150, 86, 70), r=5)
        c.rect(T[0] - 22, T[1] + 24, T[0] + 22, T[1] + 28, (124, 70, 58))
        c.line([(T[0] - 7, T[1] + 9), (T[0] - 7, T[1] + 2), (T[0] + 7, T[1] + 2), (T[0] + 7, T[1] + 9)], 3, (110, 64, 52), caps=False)
        draw_hand(c, T, (0, 1), h * 0.6, aw * 0.5, SKIN, ts=facing)
    return T


# ---------------------------------------------------------------- scene 1: cells
def build_cells():
    rng = random.Random(11)
    R0, ZR = 82.0, 122.0
    gens, radii, parents = [[(0.0, 0.0, 0.3)]], [R0], [None]
    for g in range(1, 6):
        r = R0 * 0.5 ** (g / 3)
        nxt, par = [], []
        for i, (px, py, pa) in enumerate(gens[-1]):
            a = pa + math.pi / 2 + rng.uniform(-0.5, 0.5)
            dx, dy = math.cos(a) * r * 0.95, math.sin(a) * r * 0.95
            nxt += [[px + dx, py + dy, a], [px - dx, py - dy, a]]
            par += [i, i]
        mind = r * 1.42
        for _ in range(120):
            for i in range(len(nxt)):
                for j in range(i + 1, len(nxt)):
                    dx, dy = nxt[j][0] - nxt[i][0], nxt[j][1] - nxt[i][1]
                    dist = math.hypot(dx, dy) + 1e-6
                    if dist < mind:
                        push = (mind - dist) * 0.25
                        nxt[i][0] -= dx / dist * push; nxt[i][1] -= dy / dist * push
                        nxt[j][0] += dx / dist * push; nxt[j][1] += dy / dist * push
            for p in nxt:
                p[0] *= 0.99; p[1] *= 0.99
                dd = math.hypot(p[0], p[1])
                lim = ZR - r - 8
                if dd > lim:
                    p[0] *= lim / dd; p[1] *= lim / dd
        gens.append([tuple(p) for p in nxt]); radii.append(r); parents.append(par)
    return gens, radii, parents, ZR

DIV_T = [0.8 + k * 1.0 for k in range(5)]
DIV_D = 0.8

def cells_bg():
    arr = vgrad((40, 20, 40), (14, 8, 22)).astype(np.float32)
    g = radial(640, 360, 420, 2.2)
    arr = arr + g[..., None] * np.array([150, 70, 70], np.float32)
    return to_img(arr)

def scene_cells(t):
    gens, radii, parents, ZR = cached('cells', build_cells)
    img = cached('cells_bg', cells_bg).copy()
    c = Canvas(img)
    # drifting particles
    rng = random.Random(3)
    for i in range(70):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        vx, vy = rng.uniform(-6, 6), rng.uniform(-10, -2)
        r = rng.uniform(0.8, 2.4)
        a = int(60 + 60 * math.sin(t * rng.uniform(0.8, 2) + i))
        c.circle((x0 + vx * t) % W, (y0 + vy * t) % H, r, (255, 210, 200, a))

    z = 1.0 + 0.22 * ease(t / 6.5)
    rot = 0.06 * t
    cx, cy = 640, 360
    def P(x, y):
        xr = x * math.cos(rot) - y * math.sin(rot)
        yr = x * math.sin(rot) + y * math.cos(rot)
        return cx + xr * z, cy + yr * z

    # current state
    g, u = 0, 0.0
    for k in range(5, 0, -1):
        if t >= DIV_T[k - 1]:
            g, u = k, ease((t - DIV_T[k - 1]) / DIV_D)
            break
    cells = []
    if g == 0:
        x, y, _ = gens[0][0]
        cells.append((x, y, radii[0]))
    else:
        for i, (x, y, _) in enumerate(gens[g]):
            px, py, _ = gens[g - 1][parents[g][i]]
            cells.append((lerp(px, x, u), lerp(py, y, u), lerp(radii[g - 1], radii[g], u)))

    # zona pellucida
    zx, zy = P(0, 0)
    c.circle(zx, zy, ZR * z + 6, (255, 180, 170, 22))
    c.circle(zx, zy, ZR * z, (255, 190, 180, 18), outline=(255, 215, 205, 110), width=2.4)
    breath = 1 + 0.012 * math.sin(t * 2.2)
    for i, (x, y, r) in enumerate(cells):
        wob = 1.2 * math.sin(t * 1.7 + i * 1.3)
        X, Y = P(x + wob, y + wob * 0.6)
        R_ = r * z * breath
        c.circle(X, Y, R_, (255, 192, 180, 92), outline=(255, 228, 218, 210), width=1.6)
    for i, (x, y, r) in enumerate(cells):
        wob = 1.2 * math.sin(t * 1.7 + i * 1.3)
        X, Y = P(x + wob, y + wob * 0.6)
        R_ = r * z * breath
        c.circle(X - R_ * 0.08, Y - R_ * 0.06, R_ * 0.32, (196, 104, 136, 150))
        c.circle(X - R_ * 0.13, Y - R_ * 0.12, R_ * 0.1, (255, 205, 215, 170))
    out = finish(img)
    fade = ease(t / 1.2)
    if fade < 1:
        out = Image.blend(Image.new('RGB', out.size), out, fade)
    return out



def _branch_window(img, w, h, fol):
    c = Canvas(img)
    bark = (104, 76, 58)
    br = [(-10, 50), (50, 66), (110, 70), (160, 92)]
    c.line(br, 8, bark)
    twigs = [((50, 66), (72, 34)), ((110, 70), (138, 44)), ((110, 70), (128, 104))]
    for a, b in twigs:
        c.line([a, b], 4, bark)
    rng = random.Random(5)
    for (x, y) in [p for _, p in twigs] + br[1:]:
        for k in range(5):
            c.circle(x + rng.uniform(-13, 13), y + rng.uniform(-11, 9), rng.uniform(6, 11), fol[k % 3])

def _blossom_window(img, w, h):
    _branch_window(img, w, h, FOLIAGE['spring'])

def _leafy_window(img, w, h):
    _branch_window(img, w, h, FOLIAGE['summer'])

# ---------------------------------------------------------------- scene 2: baby
def baby_bg():
    arr = vgrad((253, 238, 224), (244, 212, 202)).astype(np.float32)
    g = radial(640, 330, 600, 1.6)
    arr = arr + g[..., None] * np.array([6, 10, 14], np.float32)
    img = to_img(arr)
    c = Canvas(img)
    c.circle(640, 395, 262, (206, 222, 242))
    rng = random.Random(5)
    for _ in range(46):
        a, rr = rng.uniform(0, 2 * math.pi), math.sqrt(rng.uniform(0.02, 1)) * 238
        c.circle(640 + math.cos(a) * rr, 395 + math.sin(a) * rr, rng.uniform(2, 4), (255, 255, 255, 170))
    c.circle(640, 395, 262, None, outline=(186, 204, 230), width=6)
    return img

def scene_baby(t):
    lt = t - 9.0
    img = cached('baby_bg', baby_bg).copy()
    c = Canvas(img)
    br = 1 + 0.025 * math.sin(2 * math.pi * lt / 1.7)
    # swaddle body
    c.ellipse(640, 452, 112 * br, 136, (232, 214, 168))
    c.ellipse(640, 448, 106 * br, 130, (250, 233, 184))
    c.poly([(536, 400), (742, 470), (738, 500), (544, 430)], (240, 218, 160))
    c.poly([(744, 392), (560, 520), (548, 500), (736, 372)], (244, 224, 170))
    # hood
    c.ellipse(640, 300, 86, 86, (236, 220, 174))
    # tiny hands
    c.circle(612, 372, 13, SKIN); c.circle(668, 375, 13, SKIN)
    # head
    hx, hy = 640, 302
    c.circle(hx - 60, hy + 6, 11, SKIN_D); c.circle(hx + 60, hy + 6, 11, SKIN_D)
    c.ellipse(hx, hy, 62, 60, SKIN)
    c.arc(hx + 2, hy - 55, 13, 13, 180, 470, (150, 100, 66), 4)
    ink = (70, 50, 50)
    c.arc(hx - 22, hy + 4, 11, 7, 10, 170, ink, 3)
    c.arc(hx + 22, hy + 4, 11, 7, 10, 170, ink, 3)
    c.circle(hx - 36, hy + 22, 11, (240, 140, 130, 110)); c.circle(hx + 36, hy + 22, 11, (240, 140, 130, 110))
    c.ellipse(hx, hy + 16, 4, 3, SKIN_D)
    c.ellipse(hx, hy + 32, 6, 4 + 1.2 * math.sin(2 * math.pi * lt / 1.7), (200, 120, 116))
    rng = random.Random(14)
    for k in range(22):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        petal(c, (x0 + 22 * lt) % (W + 20) - 10, (y0 + 30 * lt + 10 * math.sin(lt * 1.5 + k)) % (H + 20) - 10,
              rng.uniform(1.2, 2.0), FOLIAGE['spring'][k % 3] + (210,))
    return finish(img, zoom=1.0 + 0.07 * ease(lt / 4.0), cy=380)


# ---------------------------------------------------------------- scene 3: crawling
def heart(c, x, y, s, col):
    c.circle(x - s * 0.5, y, s * 0.55, col)
    c.circle(x + s * 0.5, y, s * 0.55, col)
    c.poly([(x - s * 1.02, y + s * 0.12), (x + s * 1.02, y + s * 0.12), (x, y + s * 1.25)], col)

def rot_ellipse(c, cx, cy, rx, ry, ang, fill):
    ca, sa = math.cos(ang), math.sin(ang)
    pts = [(cx + rx * math.cos(a) * ca - ry * math.sin(a) * sa, cy + rx * math.cos(a) * sa + ry * math.sin(a) * ca)
           for a in np.linspace(0, 2 * math.pi, 48)]
    c.poly(pts, fill)

def crawl_bg():
    arr = vgrad((252, 234, 216), (242, 214, 198)).astype(np.float32)
    img = to_img(arr)
    c = Canvas(img)
    c.rect(0, 380, W, 392, (250, 246, 240))
    for k in range(0, W, 46):
        c.rect(k + 18, 392, k + 28, 560, (246, 226, 210))
    win = to_img(np.ascontiguousarray(vgrad((160, 206, 240), (214, 234, 248), h=230 * SS, w=210 * SS)))
    _blossom_window(win, 210, 230)
    img.paste(win, (760 * SS, 90 * SS))
    for (a, b, cc, dd) in [(750, 80, 980, 90), (750, 320, 980, 332), (750, 90, 760, 320), (970, 90, 980, 320), (860, 90, 870, 320)]:
        c.rect(a, b, cc, dd, (250, 246, 240))
    c.rect(0, 560, W, H, (214, 176, 140))
    c.ellipse(640, 610, 560, 56, (196, 220, 214))
    c.ellipse(640, 610, 530, 46, (214, 232, 226))
    # toy blocks
    for (x, y, col, ch) in [(150, 520, (236, 120, 110), 'A'), (198, 520, (110, 170, 220), 'B'), (174, 474, (246, 200, 90), 'C')]:
        c.rect(x - 22, y - 22, x + 22, y + 22, col, r=4)
        c.text(x, y, ch, ImageFont.truetype(FONT, 24 * SS), (255, 255, 255), anchor='mm')
    return img

def draw_crawler(c, x, gy, ph):
    onesie, onesie_d = (150, 205, 185), (124, 178, 160)
    bob = -3 * abs(math.sin(ph))
    gy2 = gy + bob
    sx, sy = x + 38, gy2 - 72
    hx_, hy_ = x - 46, gy2 - 68
    def limbs(near):
        col = onesie if near else onesie_d
        skin = SKIN if near else SKIN_D
        p_arm = ph if near else ph + math.pi
        p_leg = ph + math.pi if near else ph
        off = 0 if near else -8
        hx = sx + 18 + off + 24 * math.sin(p_arm)
        hy = gy - 9 - max(0, math.cos(p_arm)) * 14
        mid = ((sx + hx) / 2, (sy + hy) / 2)
        c.line([(sx + off, sy), mid], 24, col)
        c.line([mid, (hx, hy)], 19, skin)
        c.ellipse(hx + 4, hy + 2, 12, 9, skin)
        kx = hx_ + 14 + off + 20 * math.sin(p_leg)
        ky = gy - 13 - max(0, math.cos(p_leg)) * 10
        c.line([(hx_ + off, hy_), (kx, ky)], 30, col)
        fx = kx - 56
        c.line([(kx, ky), (fx, gy - 11)], 24, col)
        c.ellipse(fx - 8, gy - 9, 12, 8, skin)
    c.ellipse(x - 10, gy - 1, 112, 6, (0, 0, 0, 34))
    limbs(False)
    c.ellipse(x, gy2 - 80, 64, 38, onesie)
    c.circle(x - 50, gy2 - 74, 38, onesie)
    limbs(True)
    hx, hy = x + 74, gy2 - 108 - 3 * math.sin(ph * 0.5)
    c.circle(hx - 8, hy + 6, 10, SKIN_D)
    c.circle(hx, hy, 44, SKIN)
    c.circle(hx - 12, hy + 6, 9, SKIN_D)
    c.arc(hx - 6, hy - 46, 12, 12, 160, 430, (150, 100, 66), 4)
    c.chord(hx - 4, hy - 4, 44, 42, 180, 280, (170, 120, 80))
    c.circle(hx + 20, hy - 4, 5.5, (58, 42, 42))
    c.circle(hx + 21.5, hy - 5.5, 1.8, (255, 255, 255))
    c.circle(hx + 16, hy + 16, 10, (240, 130, 120, 110))
    c.ellipse(hx + 43, hy + 4, 5, 6, SKIN)
    c.arc(hx + 30, hy + 16, 9, 7, 20, 120, (160, 84, 84), 3)

def scene_crawl(t):
    lt = t - 12.5
    img = cached('crawl_bg', crawl_bg).copy()
    c = Canvas(img)
    # ball the baby is heading for
    bx, by = 1060, 556
    c.ellipse(bx, by + 6, 40, 7, (0, 0, 0, 30))
    for k, col in enumerate([(236, 96, 90), (250, 206, 90), (100, 160, 220), (250, 250, 246)]):
        c.pieslice(bx, by - 32, 36, 36, k * 90 + lt * 10, k * 90 + 90 + lt * 10, col)
    x = lerp(300, 830, ease(lt / 4.2) * 0.4 + 0.6 * clamp(lt / 4.2))
    draw_crawler(c, x, 560, lt * 2 * math.pi * 1.1)
    return finish(img)


# ---------------------------------------------------------------- the seasons
SEASON_EDGES = [(25.75, 'spring'), (44.75, 'summer'), (61.6, 'autumn'), (83.0, 'winter'), (1e9, 'spring')]
FOLIAGE = {'spring': [(250, 196, 212), (255, 234, 240), (242, 168, 196)],
           'summer': [(96, 160, 90), (124, 182, 100), (78, 140, 80)],
           'autumn': [(232, 142, 60), (206, 92, 52), (242, 184, 72)],
           'winter': [(232, 142, 60), (206, 92, 52), (242, 184, 72)]}

def season_at(t):
    for edge, name in SEASON_EDGES:
        if t < edge:
            return name

def winter_k(t):
    return 0.0 if t >= 83.0 else ease((t - 61.4) / 1.6)

def petal(c, x, y, s, col):
    c.ellipse(x, y, 4.5 * s, 2.6 * s, col)

def butterfly(c, x, y, ph, col):
    f = 0.25 + 0.75 * abs(math.sin(ph))
    c.ellipse(x - 7 * f, y - 3, 7 * f, 6, col); c.ellipse(x + 7 * f, y - 3, 7 * f, 6, col)
    c.ellipse(x - 5 * f, y + 5, 5 * f, 4, col); c.ellipse(x + 5 * f, y + 5, 5 * f, 4, col)
    c.line([(x, y - 6), (x, y + 8)], 2, (60, 50, 50), caps=False)

def window_season(dw, t, w, h, tree_g=None):
    """Seasonal view through a window: a neighbour's branch, an optional young tree, and weather."""
    sea, wk = season_at(t), winter_k(t)
    fol = FOLIAGE[sea]
    la = int(240 * (1 - wk))
    bark = (104, 76, 58)
    # young tree, planted when the daughter was born
    if tree_g is not None:
        g = tree_g
        tx, tb, th = 160, h, 30 + 170 * g
        dw.line([(tx, tb), (tx, tb - th)], 3 + 12 * g, bark, caps=False)
        R_ = 12 + 46 * g
        for k, (ox, oy) in enumerate([(-0.7, -0.1), (0.7, -0.15), (-0.35, -0.7), (0.4, -0.65)]):
            dw.line([(tx, tb - th * 0.75), (tx + ox * R_ * 1.3, tb - th + oy * R_)], 2 + 4 * g, bark, caps=False)
        if la > 0:
            for k, (ox, oy, rr) in enumerate([(0, 0, 1.0), (-0.55, 0.25, 0.75), (0.55, 0.22, 0.75),
                                              (-0.3, -0.35, 0.7), (0.32, -0.32, 0.7)]):
                dw.circle(tx + ox * R_ * 1.3, tb - th + oy * R_ * 1.2, R_ * rr, fol[k % 3] + (la,))
        if wk > 0:
            for (ox, oy) in [(-0.7, -0.1), (0.7, -0.15), (-0.35, -0.7), (0.4, -0.65), (0, -1.0)]:
                dw.ellipse(tx + ox * R_ * 1.3, tb - th + oy * R_ - 3, 7 + 6 * g, 3, (250, 252, 255, int(230 * wk)))
    # the neighbour's branch reaching in from the left
    br = [(-10, 40), (44, 54), (96, 60), (146, 78)]
    dw.line(br, 7, bark)
    twigs = [((44, 54), (64, 26)), ((96, 60), (122, 38)), ((96, 60), (112, 92)), ((146, 78), (170, 70))]
    for a, b in twigs:
        dw.line([a, b], 3.5, bark)
    if la > 0:
        rng = random.Random(8)
        for (x, y) in [p for _, p in twigs] + br[1:]:
            for k in range(4):
                dw.circle(x + rng.uniform(-12, 12), y + rng.uniform(-10, 8), rng.uniform(6, 10), fol[k % 3] + (la,))
    if wk > 0:
        for (a, b) in list(zip(br, br[1:])) + twigs:
            dw.line([(a[0], a[1] - 4), (b[0], b[1] - 4)], 3, (250, 252, 255, int(240 * wk)), caps=False)
    # weather
    rng = random.Random(30)
    if sea == 'spring':
        for k in range(14):
            x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
            petal(dw, (x0 + 18 * t) % (w + 20) - 10, (y0 + 24 * t + 6 * math.sin(t * 2 + k)) % (h + 20) - 10, 1.0,
                  fol[k % 3] + (220,))
    elif sea == 'summer':
        for k in range(2):
            x = (rng.uniform(0, w) + 30 * t) % (w + 60) - 30
            y = 40 + 30 * k + 4 * math.sin(t * 3 + k)
            fl = 4 * math.sin(t * 9 + k)
            dw.line([(x - 8, y - fl), (x, y), (x + 8, y - fl)], 2, (70, 60, 70), caps=False)
    if sea in ('autumn', 'winter') and wk < 1:
        for k in range(12):
            x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
            x = (x0 + 12 * math.sin(t * 1.4 + k) + 10 * t) % (w + 20) - 10
            y = (y0 + 30 * t) % (h + 20) - 10
            dw.ellipse(x, y, 5, 3 + 2 * math.sin(t * 4 + k), fol[k % 3] + (int(230 * (1 - wk)),))
    if wk > 0:
        for k in range(46):
            x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
            x = (x0 + 8 * math.sin(t + k)) % w
            y = (y0 + 34 * t) % (h + 10) - 5
            dw.circle(x, y, rng.uniform(1.2, 2.6), (255, 255, 255, int(230 * wk)))
        dw.rect(0, h - 10 * wk, w, h, (244, 248, 252))

CAPTIONS = [(9.3, 'Spring', (70, 56, 56)), (25.8, 'Summer', (255, 246, 236)),
            (44.8, 'Autumn', (70, 56, 56)), (62.3, 'Winter', (70, 56, 56)), (83.4, 'Spring', (70, 56, 56))]

def draw_caption(img, t):
    for (s, txt, col) in CAPTIONS:
        a = ease((t - s) / 0.6) * (1 - ease((t - s - 2.6) / 0.7))
        if a > 0:
            font = cached('capfont', lambda: ImageFont.truetype(
                os.path.join(BASE, 'assets', 'fonts', 'texgyrepagella-italic.otf'), int(46 * OS)))
            d = ImageDraw.Draw(img, 'RGBA')
            d.text((56 * OS, (676 - 10 * a) * OS), txt, font=font, fill=col + (int(235 * a),), anchor='ls')
            tw = d.textlength(txt, font=font)
            y = (690 - 10 * a) * OS
            d.line([(58 * OS, y), (58 * OS + tw * a, y)], fill=col + (int(150 * a),), width=max(1, int(OS)))
    return img


# ---------------------------------------------------------------- scene 4: the room (five visits)
WALL = [(0.0, (250, 234, 214)), (0.3, (242, 238, 228)), (0.6, (238, 232, 220)),
        (0.8, (246, 216, 186)), (0.92, (220, 180, 168)), (1.0, (186, 160, 168))]
WALL2 = [(0.0, (226, 236, 220)), (0.3, (222, 234, 218)), (0.6, (216, 230, 212)),
         (0.8, (232, 216, 184)), (0.92, (206, 184, 168)), (1.0, (174, 160, 170))]
SKY_T = [(0.0, (255, 196, 168)), (0.25, (146, 194, 236)), (0.5, (112, 172, 232)), (0.75, (238, 172, 112)),
         (0.9, (196, 112, 112)), (1.0, (70, 62, 112))]
SKY_B = [(0.0, (255, 232, 196)), (0.25, (204, 228, 246)), (0.5, (192, 222, 246)), (0.75, (250, 212, 152)),
         (0.9, (250, 170, 120)), (1.0, (190, 112, 122))]
WX0, WY0, WX1, WY1 = 840, 168, 1070, 400
NOTCHES = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
GROUND = 600
PX_PER_M = 230.0   # height_px uses 2.3 px per cm
YELLOW = (250, 204, 60)
GGC = dict(shirt=(232, 112, 96), hair=(150, 100, 60), hairlen=0.55, bun=0, girl=True)       # great-grandchild
GRANDD = dict(hair=(196, 150, 90), hairlen=0.62, shirt=(206, 146, 96), bun=0)   # granddaughter

def flower(c, x, y, s=1.0, stem=0.0, lean=0.0):
    if stem:
        c.line([(x, y), (x + lean, y + stem)], 2.4 * s, (96, 150, 86))
        c.ellipse(x + lean * 0.5 + 5 * s, y + stem * 0.55, 6 * s, 3 * s, (110, 166, 96))
    for k in range(7):
        a = 2 * math.pi * k / 7
        c.ellipse(x + math.cos(a) * 6 * s, y + math.sin(a) * 6 * s, 4.6 * s, 4.6 * s, YELLOW)
    c.circle(x, y, 4 * s, (150, 96, 40))

def vase_flower(c, x, y, s=1.0, col=(120, 160, 200)):
    flower(c, x - 1, y - 30 * s, s, stem=26 * s, lean=1)
    c.rect(x - 8 * s, y - 14 * s, x + 8 * s, y + 2 * s, col, r=4 * s)

def father_style(age):
    return dict(MAN, H=height_px(age) * 1.07, bun=0, hair=lerpc((70, 48, 34), (206, 206, 202), ease((age - 42) / 26)),
                shirt=lerpc((96, 128, 156), (130, 116, 104), ease((age - 48) / 16)), pants=(62, 66, 80))

def daughter_style(age):
    return dict(hair=(150, 100, 60), bun=0, shirt=lerpc((130, 190, 140), (236, 160, 90), ease((age - 8) / 8)),
                pants=(80, 90, 140), girl=True, part=-1)

def faded(img, alpha, draw):
    """Draw onto a transparent layer, then composite it at the given opacity."""
    if alpha <= 0:
        return None
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    out = draw(Canvas(layer))
    if alpha >= 1:
        img.paste(layer, (0, 0), layer)
    else:
        img.paste(layer, (0, 0), layer.getchannel('A').point(lambda v: int(v * alpha)))
    return out

def teacups(c, t, a1, a2):
    """A small side table with one or two cups of tea; a1/a2 are each cup's opacity."""
    tc = (176, 128, 98)
    c.rect(874, 470, 886, 598, tc)
    c.ellipse(880, 598, 28, 5, tc)
    c.ellipse(880, 470, 58, 11, (196, 146, 112))
    for (cx, a) in ((856, a1), (906, a2)):
        if a <= 0:
            continue
        A = int(255 * a)
        c.ellipse(cx, 466, 20, 5, (250, 248, 244, A))
        c.chord(cx, 452, 14, 13, 0, 180, (250, 248, 244, A))
        c.arc(cx + 15, 455, 6, 5, 270, 450, (250, 248, 244, A), 3)
        for k in range(2):
            pts = [(cx - 4 + k * 8 + 4 * math.sin(t * 2.4 + y * 0.12 + k), y) for y in np.linspace(446, 414, 8)]
            c.line(pts, 2, (255, 255, 255, int(110 * a)), caps=False)

def portrait(cc, x0, y0, hair, shirt, bg, bun=False, glasses=False):
    cc.rect(x0, y0, x0 + 80, y0 + 94, (120, 86, 66), r=3)
    cc.rect(x0 + 8, y0 + 8, x0 + 72, y0 + 86, bg)
    cc.chord(x0 + 40, y0 + 94, 30, 30, 180, 360, shirt)
    if bun:
        cc.circle(x0 + 40, y0 + 26, 8, hair)
    cc.circle(x0 + 40, y0 + 42, 15, SKIN)
    cc.chord(x0 + 40, y0 + 40, 16, 15, 180, 360, hair)
    cc.arc(x0 + 40, y0 + 46, 6, 4, 20, 160, (160, 84, 84), 1.6)
    if glasses:
        for dx in (-6, 6):
            cc.circle(x0 + 40 + dx, y0 + 41, 4.5, None, outline=(70, 58, 58), width=1.2)

# visit timings
A0, A1 = 16.0, 26.0           # first steps, growing up
B0, B1 = 34.5, 39.0           # newlyweds paint the room
C0, C1 = 44.5, 57.0           # family years, daughter leaves, two teacups
D0, D1 = 60.0, 68.5           # he fades, winter, alone with one cup
E0, E1 = 83.0, 90.0           # epilogue: spring again


PAINT0, PAINT_F, PAINT_N, COLW = 34.6, 1.15, 3, 84    # start, strokes per second, strokes, roller width
PAINTERS = [(442, 1), (862, -1)]                        # first strip, direction: they work towards each other
ROLL_TOP, ROLL_BOT = 132, 560
STAND = 58                                              # each stands beside their roller

def roller_state(t, col0, d):
    """Roller centre, height, whether it is on the wall, and the stroke index."""
    u = max(0.0, (t - PAINT0) * PAINT_F)
    k = int(u)
    if k >= PAINT_N:
        return col0 + d * COLW * (PAINT_N - 1), ROLL_BOT, False, PAINT_N, 1.0
    f = u - k
    cx = col0 + d * COLW * k
    if f < 0.55:                                      # rolling down the wall
        return cx, lerp(ROLL_TOP, ROLL_BOT, ease(f / 0.55)), True, k, f / 0.55
    if k == PAINT_N - 1:                              # last strip done: rollers stay down
        return cx, ROLL_BOT, False, k, 1.0
    g = ease((f - 0.55) / 0.45)                       # lift off, move up and across to the next strip
    nxt = cx + d * COLW if k < PAINT_N - 1 else cx
    return lerp(cx, nxt, g), lerp(ROLL_BOT, ROLL_TOP, g), False, k, 1.0

def paint_mask(t):
    xs = np.arange(W * SS)[None, :] / SS
    ys = np.arange(H * SS)[:, None] / SS
    m = np.broadcast_to((xs < 400) | (xs > 904), (H * SS, W * SS)).copy()   # the sides are already done
    for col0, d in PAINTERS:
        cx, ry, on, k, _ = roller_state(t, col0, d)
        for j in range(min(k + (0 if on else 1), PAINT_N)):
            c0 = col0 + d * COLW * j
            m |= np.broadcast_to(np.abs(xs - c0) < COLW / 2, m.shape)
        if on:
            c0 = col0 + d * COLW * k
            m |= (np.abs(xs - c0) < COLW / 2) & (ys < ry + 9)
    return m[..., None]


TIMELAPSE = [(20.0, 25.0), (44.5, 51.0), (54.0, 57.0), (63.0, 68.0)]
TL_RATE, TL_RAMP = 1.4, 0.5

def _tl_integral(t, t0, t1, ramp=TL_RAMP):
    span = t1 - t0 - ramp
    rate = max(1, round(TL_RATE * span)) / span          # a whole number of days per window
    if t <= t0:
        return 0.0
    a = min(t, t0 + ramp) - t0
    v = rate * a * a / (2 * ramp)
    if t > t0 + ramp:
        v += rate * (min(t, t1 - ramp) - (t0 + ramp))
    if t > t1 - ramp:
        b = min(t, t1) - (t1 - ramp)
        v += rate * (b - b * b / (2 * ramp))
    return v

def days_passed(t):
    return sum(_tl_integral(t, a, b) for a, b in TIMELAPSE)

def timelapse_on(t):
    return max([ease((t - a) / TL_RAMP) * (1 - ease((t - (b - TL_RAMP)) / TL_RAMP)) for a, b in TIMELAPSE] + [0.0])

def room_state(t):
    S = dict(visit='A', paint=None, mom=1.0, dad=None, kid=None, kid_x=500, kid_alpha=0.0, dad_alpha=0.0,
             photo=0.0, readout=0.0, flowers=False, cups=(0.0, 0.0), table=False)
    if t < 30.0:
        S['mom'] = (lerp(1.0, 1.3, clamp((t - A0) / 4.0)) if t < 20.0 else
                    lerp(1.3, 20.0, (t - 20.0) / 5.0) if t < 25.0 else lerp(20.0, 21.0, clamp(t - 25.0)))
        S['readout'] = ease((t - 20.1) / 0.4) * (1 - ease((t - 25.3) / 0.4))
    elif t < 41.0:
        S.update(visit='B', mom=29.0, dad=31.0, dad_alpha=1.0, paint=clamp((t - 35.0) / 3.0), flowers=True)
    elif t < 58.5:
        S.update(visit='C', flowers=True, table=True)
        if t < 51.0:
            S['mom'] = lerp(33.0, 50.0, (t - C0) / 6.5)
            S['kid'] = lerp(1.0, 18.0, clamp((t - C0) / 6.5))
            S['kid_alpha'] = 1.0
        elif t < 54.0:
            u = (t - 51.0) / 3.0
            S['mom'] = lerp(50.0, 55.0, u)
            S['kid'] = 18.0
            S['leaving'] = u
            S['kid_alpha'] = 1 - ease((t - 53.1) / 0.6)
        else:
            S['mom'] = lerp(55.0, 68.0, clamp((t - 54.0) / 3.0))
            S['cups'] = (ease((t - 54.0) / 0.5), ease((t - 54.0) / 0.5))
        S['dad'] = S['mom'] + 2.0
        S['dad_alpha'] = 1.0
    elif t < 75.0:
        S.update(visit='D', flowers=True, table=True)
        S['mom'] = (lerp(72.0, 75.0, (t - D0) / 3.0) if t < 63.0 else lerp(75.0, 90.0, clamp((t - 63.0) / 5.0)))
        S['dad'] = min(S['mom'], 75.0) + 2.0
        S['dad_alpha'] = 1 - ease((t - 61.3) / 1.2)
        S['photo'] = ease((t - 62.2) / 0.8)
        S['cups'] = (1.0, S['dad_alpha'])
    else:
        S.update(visit='E', mom=90.0, flowers=True, photo=1.0, photo2=1.0)
    return S

def scene_room(t):
    S = room_state(t)
    age = S['mom']
    visit = S['visit']
    L = 0.18 if visit == 'E' else clamp(age / 90)
    wall_old, wall_new = interp(WALL, L), interp(WALL2, L)
    wk = winter_k(t)
    if visit == 'A':
        arr = np.array(vgrad(wall_old, tuple(int(v * 0.93) for v in wall_old)), dtype=np.float32)
        wall = wall_old
    elif visit == 'B':
        a_old = vgrad(wall_old, tuple(int(v * 0.93) for v in wall_old))
        a_new = vgrad(wall_new, tuple(int(v * 0.93) for v in wall_new))
        arr = np.where(paint_mask(t), a_new, a_old).astype(np.float32)
        wall = wall_new
    else:
        arr = np.array(vgrad(wall_new, tuple(int(v * 0.93) for v in wall_new)), dtype=np.float32)
        wall = wall_new
    # window
    # days race by during the growing-up and growing-old stretches: the sun crosses again and again
    dp = (L + days_passed(t)) % 1.0
    night = timelapse_on(t) * (1 - math.sin(math.pi * dp)) ** 3 * 0.5
    sky_t = lerpc(lerpc(interp(SKY_T, L), (150, 160, 184), 0.5 * wk), (36, 40, 86), night)
    sky_b = lerpc(lerpc(interp(SKY_B, L), (206, 210, 222), 0.5 * wk), (70, 64, 110), night)
    sky = np.ascontiguousarray(vgrad(sky_t, sky_b, h=(WY1 - WY0) * SS, w=(WX1 - WX0) * SS))
    win = to_img(sky)
    dw = Canvas(win)
    sx = lerp(-30, WX1 - WX0 + 30, dp)
    sy = (WY1 - WY0) + 24 - math.sin(math.pi * dp) * (WY1 - WY0) * 0.8
    suncol = interp([(0, (255, 214, 150)), (0.35, (255, 250, 220)), (0.7, (255, 214, 140)), (1, (255, 130, 90))], L)
    dw.circle(sx, sy, 48, suncol + (60,)); dw.circle(sx, sy, 24, suncol)
    for i, (cx0, cy0, s) in enumerate([(40, 70, 1.0), (160, 130, 0.8), (110, 40, 0.6)]):
        cx_ = (cx0 + (t * 6 + days_passed(t) * 170) * (1 + i * 0.3)) % (WX1 - WX0 + 120) - 60
        ca = (255, 255, 255, int(150 * (1 - ease((L - 0.85) / 0.15)) * (1 - night)))
        dw.ellipse(cx_, cy0, 34 * s, 12 * s, ca); dw.ellipse(cx_ + 18 * s, cy0 - 7 * s, 20 * s, 12 * s, ca)
    tree = None
    if visit in ('C', 'D'):
        tree = clamp((age - 33) / 40) ** 0.8
    elif visit == 'E':
        tree = 1.0
    window_season(dw, t, WX1 - WX0, WY1 - WY0, tree)
    img = to_img(arr)
    img.paste(win, (WX0 * SS, WY0 * SS))
    c = Canvas(img)
    fr = (250, 246, 238)
    c.rect(WX0 - 12, WY0 - 12, WX1 + 12, WY0, fr); c.rect(WX0 - 12, WY1, WX1 + 12, WY1 + 12, fr)
    c.rect(WX0 - 12, WY0, WX0, WY1, fr); c.rect(WX1, WY0, WX1 + 12, WY1, fr)
    c.rect((WX0 + WX1) / 2 - 5, WY0, (WX0 + WX1) / 2 + 5, WY1, fr)
    c.rect(WX0, (WY0 + WY1) / 2 - 5, WX1, (WY0 + WY1) / 2 + 5, fr)
    c.rect(WX0 - 24, WY1 + 10, WX1 + 24, WY1 + 22, (236, 228, 216), r=3)
    if S['flowers']:
        vase_flower(c, WX1 - 30, WY1 + 10)
    # doorway + frame
    c.rect(214, 110, 362, GROUND, tuple(int(v * 0.62) for v in wall))
    c.rect(214, 110, 362, 140, tuple(int(v * 0.55) for v in wall))
    jamb = (244, 236, 222)
    c.rect(196, 96, 380, 112, jamb); c.rect(196, 96, 214, GROUND, jamb); c.rect(360, 96, 384, GROUND, jamb)
    c.line([(380, 112), (380, GROUND)], 1.5, (210, 198, 184), caps=False)
    floor = lerpc((204, 158, 118), (150, 112, 96), ease((L - 0.7) / 0.3))
    c.rect(0, GROUND, W, H, floor)
    for k in range(5):
        y = GROUND + 18 + k * k * 6
        c.line([(0, y), (W, y)], 1.4, tuple(int(v * 0.9) for v in floor), caps=False)
    c.rect(0, GROUND - 10, W, GROUND + 2, (250, 246, 240))
    c.rect(214, GROUND - 10, 362, GROUND + 2, tuple(int(v * 0.62) for v in wall))
    # measuring tape in metres
    font = cached('font', lambda: ImageFont.truetype(FONT, 13 * SS))
    ink = (96, 84, 78)
    c.rect(388, GROUND - 2.0 * PX_PER_M - 4, 404, GROUND - 10, (250, 228, 150))
    for cm in range(0, 201, 10):
        y = GROUND - cm / 100 * PX_PER_M
        if y > GROUND - 10:
            continue
        major = cm % 50 == 0
        c.line([(388, y), (388 + (16 if major else 7), y)], 1.6 if major else 1.0, ink, caps=False)
        if major and cm > 0:
            c.text(410, y, f'{cm / 100:.1f} m', font, ink)
    # pencil notches: hers grey, her daughter's red, her great-granddaughter's blue
    for m in NOTCHES:
        a = 1.0 if visit == 'E' else ease((age - m) / 0.35)
        if a > 0:
            y = GROUND - height_px(m)
            c.line([(362, y), (384, y)], 2.0, (112, 98, 92, int(230 * a)), caps=False)
    if visit in ('C', 'D', 'E'):
        kid_age = S['kid'] if S['kid'] is not None else 18.0
        for m in NOTCHES:
            a = ease((kid_age - m) / 0.35)
            if a > 0:
                y = GROUND - height_px(m) + 3
                c.line([(362, y), (376, y)], 2.4, (200, 72, 72, int(240 * a)), caps=False)
    if visit == 'E':
        a = ease((t - 86.55) / 0.3)
        if a > 0:
            y = GROUND - height_px(7) - 3
            c.line([(362, y), (384, y)], 2.6, (70, 110, 200, int(250 * a)), caps=False)
    # photos on the wall
    if S['photo'] > 0:
        def photos(cc):
            portrait(cc, 640 if visit == 'E' else 690, 120, (190, 188, 184), (96, 128, 156), (222, 228, 236), glasses=True)
            if visit == 'E':
                portrait(cc, 744, 120, (232, 230, 226), (176, 146, 112), (232, 222, 236), bun=True, glasses=True)
            cc.rect(628 if visit == 'E' else 680, 222, 836 if visit == 'E' else 780, 230, (150, 110, 86), r=2)
            vase_flower(cc, 735 if visit == 'E' else 752, 222, 0.9, col=(200, 200, 214))
        faded(img, S['photo'], photos)
    c.ellipse(620, GROUND + 26, 210, 22, (186, 112, 96))
    c.ellipse(620, GROUND + 26, 192, 16, (214, 150, 120))
    if S['table']:
        teacups(c, t, *S['cups'])

    # ---------------- people
    hp = height_px(age)
    if ROOM_MODE is not None:
        return pregnancy_people(img, c, ROOM_MODE)
    if visit == 'A':
        if t < 19.7:
            u = clamp((t - A0) / 3.4)
            x = lerp(300, 600, u)
            ph = (t - A0) * 2 * math.pi * 1.25
            a_side = 1 - ease((t - 19.4) / 0.3)
            c.ellipse(x, GROUND + 2, 26 + hp * 0.12, 7, (0, 0, 0, 34))
            faded(img, a_side, lambda cc: draw_walker(cc, x, GROUND, age, ph, 1, style={'girl': True, 'mouth': 'big'}, arms_up=True,
                                                         stride=0.6 * (300 / 3.4) / 1.25))
            if t > 19.4:
                faded(img, 1 - a_side, lambda cc: draw_person(cc, 600, GROUND, age, style={'girl': True}))
        else:
            x = 600
            if S['readout'] > 0:
                y = GROUND - hp
                c.line([(406, y), (x - 40, y)], 1.2, ink + (int(120 * S['readout']),), caps=False)
                c.text(x - 46, y - 12, f'{hp / PX_PER_M:.2f} m', font, ink + (int(255 * S['readout']),), anchor='rm')
            c.ellipse(x, GROUND + 2, 26 + hp * 0.12, 7, (0, 0, 0, 34))
            draw_person(c, x, GROUND, age, style={'girl': True})
        return finish(img)

    if visit == 'B':
        paint_dark = tuple(int(v * 0.68) for v in wall_new)
        mstyle = {'shirt': (226, 150, 120), 'pants': (70, 86, 120)}
        turn = ease((t - 37.35) / 0.2)
        people = [(29, mstyle, PAINTERS[0]), (31, father_style(31), PAINTERS[1])]
        if turn < 1:
            def painters_back(cc):
                for (pa, st, (col0, d)) in people:
                    cx, ry, on, k, _ = roller_state(t, col0, d)
                    px = cx - d * STAND
                    hp_ = st.get('H', height_px(pa))
                    gy = clamp(ry + 110, GROUND - hp_ * 0.95, GROUND - hp_ * 0.47 - 10)
                    gx = cx - d * 12
                    # roller and pole sit between them and the wall, so they go behind the figure
                    cc.line([(cx, ry + 10), (cx, ry + 18), (gx, gy + 40)], 5, (96, 96, 104), caps=False)
                    cc.rect(cx - COLW / 2 + 2, ry - 12, cx + COLW / 2 - 2, ry + 12, (80, 80, 88), r=9)
                    cc.rect(cx - COLW / 2 + 5, ry - 9, cx + COLW / 2 - 5, ry + 9, paint_dark, r=7)
                    cc.ellipse(px, GROUND + 2, 70, 7, (0, 0, 0, 34))
                    draw_person_back(cc, px, GROUND, pa, hands={d: (gx, gy), -d: (gx - d * 4, gy + 32)}, style=st)
            faded(img, 1 - turn, painters_back)
        if turn > 0:
            glance = ease((t - 37.6) / 0.35)
            def painters_front(cc):
                for i, (pa, st, (col0, d)) in enumerate(people):
                    cx, _, _, _, _ = roller_state(99, col0, d)
                    px = cx - d * STAND
                    st = dict(st, look=d * 0.75 * glance)
                    cc.ellipse(px, GROUND + 2, 70, 7, (0, 0, 0, 34))
                    j = draw_person(cc, px, GROUND, pa, style=dict(st, mouth='big' if glance > 0.6 else 'smile'))
                    T = j[d][2]
                    cc.line([(T[0], T[1]), (T[0] + d * 6, T[1] + 108)], 5, (96, 96, 104), caps=False)
                    cc.rect(T[0] + d * 6 - 33, T[1] + 101, T[0] + d * 6 + 33, T[1] + 125, (80, 80, 88), r=9)
                    cc.rect(T[0] + d * 6 - 30, T[1] + 104, T[0] + d * 6 + 30, T[1] + 122, paint_dark, r=7)
                    redraw_hand(cc, j, d)
                if glance > 0.6:
                    u = (t - 38.0) / 1.2
                    if 0 < u < 1:
                        heart(cc, 655, 190 - 50 * u, 9 + 5 * u, (240, 110, 130, int(220 * math.sin(math.pi * u))))
            faded(img, turn, painters_front)
        return finish(img, zoom=1.0 + 0.04 * ease((t - B0) / 4))

    if visit == 'E':
        gx = 540
        mark = ease((t - 86.0) / 0.6) * (1 - ease((t - 87.0) / 0.5))
        child_look = -0.6 if 86.8 < t < 87.6 else (0.6 if t >= 87.6 else 0.0)
        tip = (391, GROUND - height_px(7) - 3)
        gh = {-1: (lerp(486, tip[0] + 4, mark), lerp(420, tip[1], mark))} if mark > 0 else None
        c.ellipse(gx, GROUND + 2, 74, 7, (0, 0, 0, 34))
        jg = draw_person(c, gx, GROUND, 36, hands=gh, style=dict(GRANDD, look=-0.6))
        if mark > 0:
            T = jg[-1][2]
            c.line([(T[0] - 2, T[1] - 2), (T[0] - 12, T[1] - 14)], 4, (234, 196, 80), caps=False)
        hp7 = height_px(7)
        if t < 85.8:
            u = clamp((t - 84.0) / 1.6)
            if u > 0:
                x = lerp(290, 430, u)
                a_side = 1 - ease((t - 85.6) / 0.2)
                c.ellipse(x, GROUND + 2, 26 + hp7 * 0.12, 7, (0, 0, 0, 34))
                faded(img, a_side, lambda cc: draw_walker(cc, x, GROUND, 7, (t - 84.0) * 2 * math.pi * 1.5, 1, style=GGC,
                                                             stride=0.6 * (140 / 1.6) / 1.5))
                if t > 85.6:
                    faded(img, 1 - a_side, lambda cc: draw_person(cc, 430, GROUND, 7, style=GGC))
        else:
            c.ellipse(430, GROUND + 2, 26 + hp7 * 0.12, 7, (0, 0, 0, 34))
            draw_person(c, 430, GROUND, 7, style=dict(GGC, look=child_look, mouth='big' if t > 87.2 else 'smile'))
        out = finish(img)
        f = ease((t - 88.6) / 1.4)
        if f > 0:
            out = Image.blend(out, Image.new('RGB', out.size), f)
        return out

    # visits C and D: the family years
    mx = 620
    mstyle, mhands = {}, {}
    m_eyes, m_mouth = 'open', 'smile'
    ld = S.get('leaving')
    if visit == 'C':
        if S['kid'] is not None and S['kid'] < 9:
            mstyle['look'] = -0.45
        if ld is not None and ld < 0.55:
            mhands = {-1: (mx - 72, GROUND - hp * 0.86 + 10 * math.sin(t * 12))}
        if ld is not None:
            mstyle['look'] = -0.6
            mstyle['brow'] = 'worried'
        if S['kid'] is None:
            mhands = {1: (686, 428)}
            g = ease((t - 55.0) / 0.4) * (1 - ease((t - 56.4) / 0.4))
            mstyle['look'] = 0.7 * g
    else:
        if S['dad_alpha'] > 0.45:
            mhands = {1: (686, 428)}
        if 61.6 < t < 64.0:
            m_eyes, m_mouth = 'closed', 'sad'
            mstyle['look'] = 0.5 * ease((t - 61.8) / 0.5)
        elif 64.0 <= t < 66.5:
            m_mouth = 'neutral'
            mstyle['brow'] = 'worried'

    def draw_dad(cc):
        da = S['dad']
        st = father_style(da)
        hands = None
        if S['kid'] is None:
            hands = {-1: (690, 430)}
        if visit == 'C' and S['kid'] is None:
            g = ease((t - 55.0) / 0.4) * (1 - ease((t - 56.4) / 0.4))
            st['look'] = -0.7 * g
        if ld is not None:
            st['look'] = -0.4
        cc.ellipse(750, GROUND + 2, 26 + hp * 0.12, 7, (0, 0, 0, 34))
        return draw_person(cc, 750, GROUND, da, hands=hands, style=st)
    if S['dad'] is not None:
        faded(img, S['dad_alpha'], draw_dad)

    if S['kid'] is not None and S['kid_alpha'] > 0:
        def draw_kid(cc):
            ka = S['kid']
            khp = height_px(ka)
            if ld is None or ld < 0.233:
                kh = {}
                if ld is not None:
                    kh = {1: (570, GROUND - khp * 0.86 + 10 * math.sin(t * 12))}
                cc.ellipse(500, GROUND + 2, 26 + khp * 0.12, 7, (0, 0, 0, 34))
                return draw_person(cc, 500, GROUND, ka, hands=kh, style=dict(daughter_style(ka), look=0.3 if ld else 0))
            u = clamp((t - 51.8) / 1.8)
            x = lerp(500, 286, u)
            cc.ellipse(x, GROUND + 2, 26 + khp * 0.12, 7, (0, 0, 0, 34))
            return draw_walker(cc, x, GROUND, ka, (t - 51.7) * 2 * math.pi * 1.3, -1, style=daughter_style(ka), carry=True,
                               stride=0.6 * (214 / 1.8) / 1.3)
        faded(img, S['kid_alpha'], draw_kid)

    c.ellipse(mx, GROUND + 2, 26 + hp * 0.12, 7, (0, 0, 0, 34))
    draw_person(c, mx, GROUND, age, eyes=m_eyes, mouth=m_mouth, hands=mhands, style=mstyle)
    return finish(img)


# ---------------------------------------------------------------- scene 5: a first date
WOMAN = {'hairlen': 0.62}
MAN = {'hairlen': 0, 'hair': (70, 48, 34), 'sw': 0.92, 'hw': 0.68, 'male': True, 'part': -1}

def date_bg():
    arr = vgrad((96, 64, 96), (58, 40, 70)).astype(np.float32)
    img = to_img(arr)
    c = Canvas(img)
    c.rect(0, 600, W, H, (70, 46, 54))
    # a window onto the night street
    c.rect(880, 140, 1180, 420, (34, 34, 70), r=6)
    rng = random.Random(2)
    for _ in range(18):
        x0 = rng.uniform(890, 1160)
        c.rect(x0, rng.uniform(300, 380), x0 + rng.uniform(16, 40), 420, (24, 24, 50))
        c.rect(x0 + 4, rng.uniform(320, 400), x0 + 9, 412, (250, 210, 130, 140))
    c.rect(874, 134, 1186, 146, (130, 96, 90)); c.rect(874, 414, 1186, 426, (130, 96, 90))
    c.rect(1024, 140, 1032, 420, (130, 96, 90))
    return img

def string_lights(c, t):
    for k, x in enumerate(np.linspace(20, 1260, 26)):
        y = 70 + 60 * (1 - ((x - 640) / 640) ** 2)
        tw = 0.8 + 0.2 * math.sin(t * 3 + k)
        c.circle(x, y + 10, 18, (255, 210, 140, int(50 * tw)))
        c.circle(x, y + 10, 5, (255, 230, 170))
    pts = [(x, 70 + 60 * (1 - ((x - 640) / 640) ** 2)) for x in np.linspace(0, W, 40)]
    c.line(pts, 1.5, (40, 30, 40), caps=False)


def wine_glass(c, gx, base_y, level=0.5):
    """A stemmed wine glass: tulip bowl, stem and foot."""
    top, bot = base_y - 70, base_y - 36
    def hw(v):
        return 14 * math.sin(math.pi * (0.26 + 0.74 * v))
    vs = np.linspace(0, 1, 18)
    left = [(gx - hw(v), top + v * (bot - top)) for v in vs]
    right = [(gx + hw(v), top + v * (bot - top)) for v in vs]
    c.poly(left + right[::-1], (236, 242, 248, 70))
    wv = [v for v in vs if v >= level]
    c.poly([(gx - hw(v), top + v * (bot - top)) for v in wv] + [(gx + hw(v), top + v * (bot - top)) for v in wv[::-1]],
           (128, 18, 40, 235))
    c.ellipse(gx, top + level * (bot - top), hw(level), 2.2, (160, 40, 60, 235))
    glass = (238, 244, 250, 220)
    c.line(left, 1.4, glass, caps=False); c.line(right, 1.4, glass, caps=False)
    c.ellipse(gx, top, hw(0), 2.4, None, outline=glass, width=1.2)
    c.line([(gx - hw(0.15) + 4, top + 6), (gx - hw(0.5) + 4, top + 0.5 * (bot - top))], 2, (255, 255, 255, 150), caps=False)
    c.line([(gx, bot), (gx, base_y - 3)], 2.4, glass, caps=False)
    c.ellipse(gx, base_y - 2, 11, 3, glass)

def scene_date(t):
    lt = t - 25.5
    img = cached('date_bg', date_bg).copy()
    c = Canvas(img)
    string_lights(c, t)
    for k in range(12):
        fx = 900 + (k * 23 + 40 * math.sin(t * 0.7 + k * 1.3)) % 270
        fy = 170 + (k * 37 % 220) + 18 * math.cos(t * 0.9 + k)
        a = 0.5 + 0.5 * math.sin(t * 3 + k * 2)
        c.circle(fx, fy, 7, (220, 255, 140, int(60 * a))); c.circle(fx, fy, 2.2, (240, 255, 180, int(240 * a)))
    for x in (560, 730):
        c.rect(x - 62, 300, x + 62, 600, (122, 80, 62), r=18)
        c.rect(x - 50, 312, x + 50, 600, (140, 94, 72), r=14)
    p1, p2 = ease((lt - 1.0) / 0.7), ease((lt - 2.0) / 0.8)
    rest, reach, hold = (612, 470), (648, 434), (592, 404)
    her = (lerp(lerp(rest[0], reach[0], p1), hold[0], p2), lerp(lerp(rest[1], reach[1], p1), hold[1], p2))
    his = (lerp(660, 684, p2), lerp(434, 470, p2))
    gl = ease((lt - 0.7) / 0.5) * (1 - 0.5 * ease((lt - 2.6) / 0.6))
    jw = draw_person(c, 560, 668, 24, hands={1: her}, style=dict(WOMAN, shirt=(206, 70, 92), nolegs=True, look=0.75 * gl, mouth='big' if p2 > 0.5 else 'smile'))
    jm = draw_person(c, 730, 660, 26, hands={-1: his}, style=dict(MAN, shirt=(62, 84, 126), nolegs=True, look=-0.75 * gl, mouth='big' if p2 > 0.8 else 'smile'))
    c.poly([(470, 480), (810, 480), (850, 600), (430, 600)], (236, 228, 220))
    c.ellipse(640, 480, 172, 24, (248, 244, 238))
    fl = 1 + 0.12 * math.sin(t * 13) + 0.06 * math.sin(t * 29)
    c.circle(600, 432, 22, (255, 210, 140, 34)); c.circle(600, 432, 13, (255, 220, 150, 50))
    c.ellipse(600, 474, 18, 5, (200, 170, 120))
    c.rect(593, 440, 607, 474, (250, 244, 230))
    c.ellipse(600, 430 - 3 * fl, 5, 10 * fl, (255, 200, 90)); c.ellipse(600, 432, 2.5, 5, (255, 250, 220))
    for gx in (530, 760):
        wine_glass(c, gx, 474)
    for j, side in ((jw, 1), (jm, -1)):
        redraw_forearm(c, j, side)
    fx, fy = (his[0] - 6, his[1]) if p2 <= 0 else (her[0] + 6, her[1])   # passed hand to hand
    flower(c, fx - 2, fy - 26, 1.1, stem=24, lean=2)
    for j, side in ((jw, 1), (jm, -1)):
        redraw_hand(c, j, side)
    for k in range(4):
        u = (lt - 2.4 - k * 0.6) / 2.0
        if 0 < u < 1:
            heart(c, 640 + 26 * math.sin(u * 6 + k), 380 - u * 210, 9 + 6 * u, (240, 110, 130, int(220 * math.sin(math.pi * u))))
    return finish(img, zoom=1.0 + 0.08 * ease(lt / 4.5), cx=650, cy=390)


# ---------------------------------------------------------------- scene 6: the wedding
def wedding_bg():
    arr = vgrad((190, 222, 246), (252, 238, 226)).astype(np.float32)
    img = to_img(arr)
    c = Canvas(img)
    c.circle(1120, 110, 80, (255, 250, 220, 50)); c.circle(1120, 110, 44, (255, 248, 214))
    c.ellipse(260, 560, 420, 120, (170, 206, 150)); c.ellipse(1040, 570, 460, 130, (156, 196, 140))
    c.rect(0, 560, W, H, (138, 190, 120))
    c.poly([(600, 560), (680, 560), (900, H), (380, H)], (248, 244, 238))
    # flower arch
    wood = (250, 246, 238)
    c.rect(472, 300, 492, 610, wood); c.rect(788, 300, 808, 610, wood)
    c.arc(640, 300, 158, 158, 180, 360, wood, 20)
    rng = random.Random(12)
    pts = [(482, y) for y in np.linspace(600, 300, 16)] + [(798, y) for y in np.linspace(600, 300, 16)]
    pts += [(640 + 158 * math.cos(a), 300 + 158 * math.sin(a)) for a in np.linspace(math.pi, 2 * math.pi, 26)]
    for (x, y) in pts:
        for _ in range(2):
            c.ellipse(x + rng.uniform(-16, 16), y + rng.uniform(-10, 10), 9, 5, (110, 160, 100))
    for (x, y) in pts:
        c.circle(x + rng.uniform(-10, 10), y + rng.uniform(-8, 8), rng.uniform(7, 11),
                 rng.choice([(250, 196, 208), (255, 255, 250), (246, 170, 150), (236, 150, 176)]))
    return img


def guest_bust(c, x, y, s, hair, shirt, style, cheer=0.0, hat=(90, 110, 150)):
    """A seated wedding guest seen from behind: shoulders and head."""
    if cheer > 0:
        for side in (-1, 1):
            hx_, hy_ = x + side * 62 * s, y - 64 * s * cheer
            c.line([(x + side * 36 * s, y + 18 * s), (hx_, hy_)], 17 * s, shirt)
            c.circle(hx_, hy_ - 6 * s, 9 * s, SKIN_D)
    c.rect(x - 48 * s, y, x + 48 * s, y + 160 * s, shirt, r=30 * s)
    c.rect(x - 11 * s, y - 22 * s, x + 11 * s, y + 8 * s, SKIN_D)
    hy = y - 48 * s
    if style != 'long':
        for side in (-1, 1):
            c.circle(x + side * 29 * s, hy + 4 * s, 7 * s, SKIN_D)
    c.ellipse(x, hy, 29 * s, 33 * s, SKIN_D)
    if style == 'long':
        c.ellipse(x, hy - 3 * s, 33 * s, 36 * s, hair)
        c.rect(x - 33 * s, hy, x + 33 * s, hy + 62 * s, hair, r=15 * s)
    elif style == 'bun':
        c.ellipse(x, hy - 4 * s, 31 * s, 32 * s, hair)
        c.circle(x, hy - 28 * s, 13 * s, hair)
    elif style == 'hat':
        c.ellipse(x, hy - 7 * s, 30 * s, 29 * s, hair)
        c.ellipse(x, hy - 22 * s, 48 * s, 9 * s, hat)
        c.rect(x - 25 * s, hy - 52 * s, x + 25 * s, hy - 20 * s, hat, r=9 * s)
        c.rect(x - 25 * s, hy - 30 * s, x + 25 * s, hy - 24 * s, (236, 206, 120))
    elif style == 'bald':
        c.chord(x, hy + 2 * s, 30 * s, 30 * s, 10, 170, hair)
    else:
        c.ellipse(x, hy - 9 * s, 30 * s, 28 * s, hair)

GUESTS = [  # x, row (0 front, 1 back), hair, style, shirt
    (180, 1, (70, 48, 34), 'short', (90, 110, 150)), (300, 1, (196, 150, 90), 'long', (200, 140, 160)),
    (420, 1, (190, 186, 182), 'bun', (150, 120, 170)), (870, 1, (40, 30, 26), 'short', (70, 70, 80)),
    (990, 1, (120, 80, 50), 'long', (120, 170, 150)), (1110, 1, (180, 176, 170), 'bald', (110, 100, 90)),
    (120, 0, (150, 100, 60), 'long', (230, 170, 120)), (240, 0, (60, 44, 36), 'hat', (60, 70, 96)),
    (360, 0, (210, 190, 150), 'bun', (170, 200, 230)), (470, 0, (90, 60, 40), 'short', (120, 100, 140)),
    (810, 0, (220, 200, 160), 'long', (236, 190, 200)), (920, 0, (50, 36, 30), 'short', (80, 110, 90)),
    (1040, 0, (140, 136, 132), 'hat', (150, 110, 90)), (1160, 0, (110, 70, 44), 'long', (190, 150, 210)),
]
CHEERERS = {240, 920, 300}

def draw_guests(c, lt, kiss):
    for row in (1, 0):
        for i, (x, r, hair, style, shirt) in enumerate(GUESTS):
            if r != row:
                continue
            s = 1.0 if row == 0 else 0.82
            y = 668 if row == 0 else 628
            cheer = ease((lt - 2.3) / 0.4) * (1 - ease((lt - 4.2) / 0.5)) if x in CHEERERS else 0.0
            bob = 3 * math.sin(lt * 9 + i) * cheer + 1.2 * math.sin(lt * 1.3 + i)
            hatcol = [(90, 110, 150), (176, 90, 110), (110, 140, 100)][i % 3]
            guest_bust(c, x, y + bob, s, hair, shirt, style, cheer, hat=hatcol)

def scene_wedding(t):
    lt = t - 30.0
    img = cached('wedding_bg', wedding_bg).copy()
    c = Canvas(img)
    c.ellipse(645, 604, 150, 9, (0, 0, 0, 30))
    k = ease((lt - 1.8) / 0.7) * (1 - ease((lt - 3.6) / 0.6))
    turn = ease((lt - 1.5) / 0.5) * (1 - ease((lt - 4.0) / 0.5))
    # the officiant, under the arch behind the couple
    robe = (36, 36, 44)
    jo = draw_person(c, 645, 584, 58, hands={-1: (634, 330), 1: (657, 330)},
                     style=dict(MAN, H=height_px(58) * 1.05, shirt=robe, pants=robe, hair=(150, 146, 140), glasses=1,
                                look=-0.25 * (1 - k), bun=0, part=1))
    shy, hipy = jo['sh_y'], jo['hip_y']
    for dx in (-13, 13):
        c.line([(645 + dx, shy + 4), (645 + dx * 0.8, hipy + 40)], 8, (226, 204, 150), caps=False)
    c.rect(639, shy - 6, 651, shy + 8, (250, 250, 248))
    c.rect(621, 312, 670, 342, (110, 40, 40), r=3)
    c.rect(624, 314, 667, 318, (240, 232, 214))
    redraw_hand(c, jo, -1); redraw_hand(c, jo, 1)
    draw_person(c, 700 - 14 * k, 600, 29, hands={-1: (650 - 8 * k, 404)},
                style=dict(MAN, shirt=(46, 50, 66), pants=(46, 50, 66), tie=(150, 160, 196), shoe=(30, 30, 34),
                           eyes='closed' if k > 0.6 else ('happy' if lt > 4.1 else 'open'), mouth='big' if lt > 4.1 else 'smile', idle=1 - k, 
                           look=-0.85 * turn, wobble=-math.radians(4.2) * k, bun=0))
    draw_person(c, 590 + 14 * k, 600, 27, eyes='closed' if k > 0.6 else 'open',
                style=dict(WOMAN, shirt=(252, 252, 248), dress=(252, 252, 248), veil=True, bouquet=True,
                           eyes='closed' if k > 0.6 else ('happy' if lt > 4.1 else 'open'), mouth='big' if lt > 4.1 else 'smile', idle=1 - k, 
                           look=0.85 * turn, wobble=math.radians(4.2) * k))
    if k > 0.8:
        u = (lt - 2.4) / 1.4
        if 0 < u < 1:
            heart(c, 645, 190 - 60 * u, 10 + 6 * u, (240, 110, 130, int(230 * math.sin(math.pi * u))))
    rng = random.Random(21)
    for k in range(70):
        x0, y0, v = rng.uniform(0, W), rng.uniform(-H, H), rng.uniform(40, 90)
        y = (y0 + v * lt) % (H + 40) - 20
        x = x0 + 18 * math.sin(lt * 1.6 + k)
        col = [(250, 190, 206), (255, 255, 250), (246, 170, 150)][k % 3]
        c.ellipse(x, y, 4.5, 2.6 + 1.6 * math.sin(lt * 4 + k), col + (220,))
    draw_guests(c, lt, ease((lt - 1.8) / 0.7))
    for k, col in enumerate([(250, 190, 90), (240, 150, 190), (160, 200, 240)]):
        bx = 300 + k * 330 + 80 * math.sin(lt * 0.7 + k * 2)
        by = 420 - k * 60 + 40 * math.sin(lt * 1.1 + k)
        butterfly(c, bx, by, lt * 14 + k, col + (230,))
    return finish(img, zoom=1.0 + 0.05 * ease(lt / 5), cy=360)


# ---------------------------------------------------------------- scene 7: her own baby
def mother_bg():
    arr = vgrad((252, 238, 222), (242, 222, 208)).astype(np.float32)
    img = to_img(arr)
    c = Canvas(img)
    win = to_img(np.ascontiguousarray(vgrad((150, 200, 240), (220, 236, 248), h=250 * SS, w=220 * SS)))
    _leafy_window(win, 220, 250)
    img.paste(win, (110 * SS, 150 * SS))
    fr = (250, 246, 238)
    c.rect(102, 142, 338, 154, fr); c.rect(102, 396, 338, 408, fr); c.rect(102, 150, 114, 400, fr); c.rect(326, 150, 338, 400, fr)
    c.rect(214, 150, 226, 400, fr)
    vase_flower(c, 300, 398)
    c.poly([(114, 154), (330, 154), (760, 600), (420, 600)], (255, 246, 214, 60))
    c.rect(0, 600, W, H, (206, 164, 128))
    c.rect(0, 590, W, 602, (250, 246, 240))
    # crib
    cw = (236, 222, 206)
    c.rect(900, 420, 1180, 432, cw, r=4); c.rect(900, 560, 1180, 572, cw)
    c.rect(896, 400, 912, 600, cw, r=4); c.rect(1168, 400, 1184, 600, cw, r=4)
    for x in np.linspace(930, 1150, 9):
        c.rect(x - 3, 432, x + 3, 560, cw)
    # mobile
    c.line([(1040, 0), (1040, 230)], 1.5, (170, 150, 140), caps=False)
    c.line([(980, 230), (1100, 230)], 2, (170, 150, 140), caps=False)
    for x, col in ((980, (246, 190, 150)), (1040, (170, 206, 230)), (1100, (250, 216, 120))):
        c.line([(x, 230), (x, 270)], 1.2, (170, 150, 140), caps=False)
        c.circle(x, 282, 12, col)
    return img

def scene_mother(t):
    lt = t - 38.5
    img = cached('mother_bg', mother_bg).copy()
    c = Canvas(img)
    c.ellipse(660, 604, 150, 9, (0, 0, 0, 30))
    jd = draw_person(c, 720, 600, 32, hands={-1: (640, 284, 'down')},
                     style=dict(MAN, shirt=(96, 128, 156), pants=(62, 66, 80), look=-0.6, bun=0))
    rock = math.sin(lt * 2 * math.pi / 2.4)
    jm = draw_person(c, 600, 600, 30, eyes='closed', hands={-1: (562, 382), 1: (652, 360)},
                     style=dict(WOMAN, shirt=(232, 156, 140), pants=(70, 80, 110), wobble=math.radians(1.2) * rock, look=0.45))
    # the baby in her arms
    bx, by = 606 + 2 * rock, 352
    rot_ellipse(c, bx, by, 64, 30, math.radians(-18), (206, 222, 244))
    rot_ellipse(c, bx - 6, by + 4, 50, 22, math.radians(-18), (220, 232, 248))
    hx, hy = bx + 52, by - 22
    c.circle(hx, hy, 30, (206, 222, 244))
    c.circle(hx + 2, hy + 2, 21, SKIN)
    c.arc(hx - 5, hy + 2, 5, 3, 10, 170, (70, 50, 50), 2)
    c.arc(hx + 9, hy + 2, 5, 3, 10, 170, (70, 50, 50), 2)
    c.circle(hx - 9, hy + 10, 5, (240, 130, 120, 110)); c.circle(hx + 13, hy + 10, 5, (240, 130, 120, 110))
    redraw_hand(c, jm, 1)                        # her hand cradling under the baby
    # his hand resting on her shoulder
    redraw_hand(c, jd, -1)
    for k in range(4):
        u = (lt - 0.8 - k * 0.9) / 2.6
        if 0 < u < 1:
            heart(c, 660 + 22 * math.sin(u * 6 + k), 300 - u * 200, 7 + 5 * u, (240, 120, 140, int(200 * math.sin(math.pi * u))))
    return finish(img, zoom=1.0 + 0.06 * ease(lt / 5), cy=360)


# ---------------------------------------------------------------- scene 4: the deathbed
LAMP = (238, 352)

def bed_bg():
    arr = vgrad((206, 170, 156), (168, 134, 128)).astype(np.float32)
    img = to_img(arr)
    c = Canvas(img)
    # floor
    c.rect(0, 600, W, H, (150, 108, 92))
    c.rect(0, 590, W, 602, (196, 170, 156))
    # framed pictures on wall
    for (x0, y0, w_, h_) in [(470, 150, 70, 90), (560, 170, 92, 66)]:
        c.rect(x0, y0, x0 + w_, y0 + h_, (128, 92, 72), r=2)
        c.rect(x0 + 7, y0 + 7, x0 + w_ - 7, y0 + h_ - 7, (226, 210, 186))
    c.circle(505, 198, 13, (190, 150, 120))
    c.rect(567, 177, 645, 229, (196, 222, 240))
    c.rect(567, 214, 645, 229, (150, 196, 130))
    c.poly([(588, 196), (600, 196), (608, 229), (580, 229)], (252, 252, 248))
    c.rect(612, 196, 624, 229, (46, 50, 66))
    c.circle(594, 191, 6, SKIN); c.circle(618, 190, 6, SKIN)
    c.chord(594, 190, 7, 7, 180, 360, (98, 64, 42)); c.chord(618, 189, 6.5, 6.5, 180, 360, (70, 48, 34))
    # window at night
    wx0, wy0, wx1, wy1 = 1074, 150, 1226, 372
    sky = np.ascontiguousarray(vgrad((24, 28, 66), (74, 58, 104), h=(wy1 - wy0) * SS, w=(wx1 - wx0) * SS))
    win = to_img(sky)
    wc = Canvas(win)
    rng = random.Random(9)
    for _ in range(26):
        wc.circle(rng.uniform(0, wx1 - wx0), rng.uniform(0, wy1 - wy0), rng.uniform(0.8, 1.8), (255, 250, 230, 220))
    wc.circle(100, 52, 15, (250, 240, 210)); wc.circle(108, 47, 14, (24, 28, 66))
    img.paste(win, (wx0 * SS, wy0 * SS))
    c.rect(wx0, wy1 - 9, wx1, wy1, (226, 232, 244))
    fr = (210, 194, 178)
    c.rect(wx0 - 10, wy0 - 10, wx1 + 10, wy0, fr); c.rect(wx0 - 10, wy1, wx1 + 10, wy1 + 10, fr)
    c.rect(wx0 - 10, wy0, wx0, wy1, fr); c.rect(wx1, wy0, wx1 + 10, wy1, fr)
    c.rect((wx0 + wx1) / 2 - 4, wy0, (wx0 + wx1) / 2 + 4, wy1, fr)
    # curtain
    c.poly([(wx0 - 30, wy0 - 20), (wx0 + 18, wy0 - 20), (wx0 - 2, wy1 + 40), (wx0 - 36, wy1 + 40)], (150, 86, 92))
    # nightstand + lamp
    c.rect(170, 440, 296, 600, (126, 86, 64), r=4)
    c.rect(164, 432, 302, 446, (110, 74, 54), r=3)
    c.line([(182, 512), (284, 512)], 2, (100, 68, 50), caps=False)
    c.circle(233, 480, 4, (220, 190, 140)); c.circle(233, 548, 4, (220, 190, 140))
    c.rect(226, 380, 250, 432, (196, 160, 110))
    c.ellipse(238, 432, 26, 6, (176, 140, 96))
    c.poly([(204, 380), (272, 380), (258, 322), (218, 322)], (255, 226, 170))
    # glass of water, small photo frame
    c.rect(268, 406, 290, 432, (180, 140, 100))
    vase_flower(c, 190, 430, 1.0, col=(200, 200, 214))
    return img

def bed_grade():
    g = radial(LAMP[0], LAMP[1], 760, 1.4)
    gl = radial(LAMP[0], LAMP[1], 200, 2.0)
    m = 0.58 + 0.6 * g
    tint = np.array([1.0, 0.86, 0.72], np.float32)
    mult = m[..., None] * tint[None, None, :]
    add = gl[..., None] * np.array([90, 60, 20], np.float32)
    return mult, add, g

def draw_bed_and_elder(c, lt):
    wood, wood_d = (132, 90, 66), (112, 76, 56)
    c.rect(300, 326, 334, 600, wood, r=6); c.rect(294, 318, 340, 334, wood_d, r=4)
    c.rect(934, 418, 962, 600, wood, r=5); c.rect(930, 410, 966, 424, wood_d, r=3)
    c.rect(326, 466, 942, 522, (240, 236, 230), r=8)
    c.rect(330, 520, 940, 600, wood)
    c.line([(340, 560), (930, 560)], 2, wood_d, caps=False)
    # pillow
    c.ellipse(398, 458, 74, 25, (226, 222, 216)); c.ellipse(398, 454, 72, 23, (250, 248, 244))
    # head in profile, face to the ceiling
    hx, hy = 420, 432
    white = (236, 234, 230)
    c.ellipse(392, 440, 22, 15, white)
    c.circle(hx, hy, 31, SKIN)
    c.pieslice(hx, hy, 33, 33, 125, 245, white)
    c.arc(hx + 8, hy - 20, 8, 4, 200, 320, (200, 196, 190), 2)
    c.ellipse(hx + 9, hy - 28, 6, 7, SKIN)              # nose
    c.arc(hx + 6, hy - 14, 7, 4, 10, 170, (90, 64, 60), 2.2)  # closed eye
    c.ellipse(hx - 4, hy + 4, 5, 7, SKIN_D)              # ear
    c.line([(hx + 22, hy - 14), (hx + 25, hy - 8)], 2, (170, 110, 104))  # mouth
    # blanket with slow breathing
    breath = 3.0 * math.sin(2 * math.pi * lt / 4.6) * (1 - ease((lt - 10.0) / 3.0))
    xs = np.linspace(444, 944, 80)
    top = [466 - 6 - bump(x, 505, 20 + breath, 42) - bump(x, 600, 12, 60) - bump(x, 735, 17, 48)
           - bump(x, 878, 22, 32) for x in xs]
    pts = [(444, 470)] + list(zip(xs, top)) + [(952, 472), (954, 546)]
    pts += [(x, 546 + 4 * math.sin(x / 26)) for x in np.linspace(950, 444, 40)]
    c.poly(pts, (204, 128, 112))
    for x in (590, 700, 810):
        c.line([(x, 466 - bump(x, 600, 12, 60) - bump(x, 735, 17, 48) - 4), (x, 544)], 1.6, (182, 110, 98), caps=False)
    c.line([(x, y + 6) for x, y in zip(xs[:16], top[:16])], 13, (246, 242, 236))
    # elder's arm resting on the blanket
    seg(c, (470, 452), (528, 458), 7.5, 6.5, (222, 214, 232))
    seg(c, (528, 458), (572, 462), 6.5, 5.5, (222, 214, 232))
    seg(c, (566, 461.5), (572, 462), 5.8, 5.8, tone((222, 214, 232), 0.88))
    draw_hand(c, (586, 463), unit((528, 458), (586, 463)), 24, 6.6, SKIN, ts=-1)

FAMILY = [  # x, ground, age, style
    (520, 668, 58, {'shirt': (124, 92, 130), 'nolegs': True}),
    (648, 590, 61, {'hairlen': 0, 'hair': (150, 145, 140), 'shirt': (70, 92, 124), 'pants': (60, 60, 70), 'sw': 0.92, 'glasses': 1}),
    (770, 590, 27, {'hair': (196, 150, 90), 'hairlen': 0.78, 'shirt': (206, 146, 96)}),
    (880, 590, 31, {'hairlen': 0, 'hair': (70, 48, 34), 'shirt': (56, 128, 118), 'sw': 0.92, 'hw': 0.68}),
]
HAND_MEET = (936, 432)

def scene_bed(t):
    lt = t - 57.5
    img = cached('bed_bg', bed_bg).copy()
    c = Canvas(img)
    j_daughter = None
    for i, (x, g, a, st) in enumerate(FAMILY):
        hands = None
        if i == 0:
            hands = {1: (590, 457)}
        if i == 3:
            hands = {1: HAND_MEET}
        jd = draw_person(c, x, g, a, eyes='closed', mouth='neutral', hands=hands, style=st)
        if i == 0:
            j_daughter = jd[1]
    # great-grandchild beside the foot of the bed
    draw_person(c, 994, 600, 7, eyes='closed', mouth='neutral', hands={-1: (HAND_MEET[0] + 6, HAND_MEET[1] + 4)},
                style={'shirt': (232, 112, 96), 'hair': (150, 100, 60), 'hairlen': 0.55})
    draw_bed_and_elder(c, lt)
    rng = random.Random(40)
    for k in range(40):
        x0, y0 = rng.uniform(1074, 1226), rng.uniform(150, 372)
        x = 1074 + (x0 - 1074 + 6 * math.sin(t * 0.8 + k)) % 152
        y = 150 + (y0 - 150 + 22 * t) % 214
        c.circle(x, y, rng.uniform(1.0, 2.2), (255, 255, 255, 220))
    # daughter's hand resting over the elder's hand (drawn over the blanket)
    S, E, T, aw, col = j_daughter
    c.line([E, T], aw * 0.92, col)
    c.circle(T[0], T[1], aw * 0.55, SKIN)

    mult, add, g = cached('bed_grade', bed_grade)
    arr = np.asarray(img, dtype=np.float32) * mult + add
    out = finish(to_img(arr), zoom=1.0 + 0.06 * ease(lt / 9), cx=620, cy=380)
    # the light goes down last at the lamp
    f = ease((t - 64.3) / 2.6)
    f2 = ease((t - 66.1) / 1.2)
    if f > 0:
        gs = cached('glow_small', lambda: np.asarray(Image.fromarray((radial(LAMP[0], LAMP[1], 640, 1.6) * 255).astype(np.uint8)).resize((W, H)), dtype=np.float32) / 255)
        z = 1.0 + 0.06 * ease(lt / 9)
        m = (1 - f) + f * gs * 0.75 * (1 - f2)
        o = np.asarray(out, dtype=np.float32) * m[..., None]
        out = to_img(o)
    return out


# ---------------------------------------------------------------- close-up: hands across generations
OLD_SKIN, OLD_SKIN_D = (238, 212, 194), (222, 190, 170)
BABY_SKIN = (246, 206, 180)

def tube_outline(pts, hws, cap_end=True, cap_start=False, n_cap=10):
    """Outline of a tube along pts with half-widths hws, optionally rounded at the ends."""
    N = len(pts)
    normals = []
    for i in range(N):
        a = pts[max(0, i - 1)]
        b = pts[min(N - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) + 1e-9
        normals.append((-dy / L, dx / L))
    left = [(p[0] + n[0] * w, p[1] + n[1] * w) for p, n, w in zip(pts, normals, hws)]
    right = [(p[0] - n[0] * w, p[1] - n[1] * w) for p, n, w in zip(pts, normals, hws)]
    out = list(left)
    if cap_end:
        p, n, w = pts[-1], normals[-1], hws[-1]
        a0 = math.atan2(n[1], n[0])
        out += [(p[0] + w * math.cos(a0 - k), p[1] + w * math.sin(a0 - k)) for k in np.linspace(0, math.pi, n_cap)]
    out += right[::-1]
    if cap_start:
        p, n, w = pts[0], normals[0], hws[0]
        a0 = math.atan2(-n[1], -n[0])
        out += [(p[0] + w * math.cos(a0 - k), p[1] + w * math.sin(a0 - k)) for k in np.linspace(0, math.pi, n_cap)]
    return out, left, right

def shaded(c, outline, col, off=(1.5, 3.0), k_shade=0.8):
    """Fill with a darker copy offset underneath, which reads as a rounded underside."""
    c.poly([(x + off[0], y + off[1]) for x, y in outline], tone(col, k_shade))
    c.poly(outline, col)

def soft_shadow(c, outline, off=(5, 8), a=46):
    c.poly([(x + off[0], y + off[1]) for x, y in outline], (60, 30, 30, a))

def mother_hand(c, old, lt):
    """Her right hand from the right, seen from the thumb side, index finger reaching left."""
    skin = OLD_SKIN if old else SKIN
    skin_d = tone(skin, 0.86)
    crease = tone(skin, 0.74) + (170,)
    sleeve = (222, 214, 232) if old else (232, 156, 140)
    # curled middle/ring/little fingers, tucked under the palm
    curled = smooth([(800, 346), (836, 350), (884, 370), (902, 398), (884, 420), (846, 420), (812, 404), (798, 376), (800, 346)], n=5)
    soft_shadow(c, curled)
    shaded(c, curled, skin_d)
    c.arc(876, 404, 26, 16, 300, 400, (224, 184, 88), 5)                      # her ring, just showing
    # back of the hand, from the wrist to the knuckles
    back = smooth([(1012, 280), (944, 274), (870, 284), (806, 298), (796, 322), (808, 346), (870, 362),
                   (944, 388), (1012, 396)], n=6)
    soft_shadow(c, back)
    shaded(c, back, skin)
    for k in range(3):                                                          # tendons / knuckle bumps
        c.line(smooth([(1000, 300 + k * 26), (920, 300 + k * 18), (840, 302 + k * 12)], n=4), 2,
               tone(skin, 1.06) + (150,), caps=False)
    # index finger: three segments tapering to the tip, with pads underneath
    xs = [806, 750, 700, 668, 636, 606, 584]
    hws = [24, 23, 22, 21, 20, 19, 18]
    pts = [(x, 322 + (806 - x) * 0.065) for x in xs]
    outline, top, bot = tube_outline(pts, hws, cap_end=True)
    bot_pad = [(p[0], p[1] + 1.6 * math.sin(math.pi * (806 - p[0]) / 85)) for p in bot]
    outline = top + outline[len(top):len(outline) - len(bot)] + bot_pad[::-1]
    soft_shadow(c, outline)
    shaded(c, outline, skin)
    tip = pts[-1]
    for xj in (700, 636):                                                       # joint creases on top
        yj = 322 + (806 - xj) * 0.065 - 20
        c.arc(xj, yj + 4, 5, 9, 200, 330, crease, 2)
        c.arc(xj + 7, yj + 5, 4, 7, 200, 320, crease, 1.6)
    # fingernail on the top of the last segment, with a pale moon
    rot_ellipse(c, tip[0] + 6, tip[1] - 9, 18, 8.5, 0.065, (250, 224, 216))
    rot_ellipse(c, tip[0] + 18, tip[1] - 9, 6, 6, 0.065, (255, 240, 236))
    c.arc(tip[0] + 6, tip[1] - 9, 18, 8.5, 160, 340, tone(skin, 0.8), 1.4)
    # thumb, lying along under the index finger
    th_pts = [(958, 372), (900, 382), (850, 386), (812, 386)]
    th_out, _, _ = tube_outline(th_pts, [26, 22, 20, 18], cap_end=True)
    shaded(c, th_out, skin, off=(1.5, 2.5))
    c.arc(850, 380, 4, 9, 200, 330, crease, 1.8)
    rot_ellipse(c, 806, 378, 11, 6, 0.0, (250, 224, 216))
    if old:                                                                     # veins, spots and wrinkles
        for pts_ in ([(1000, 330), (950, 324), (900, 330), (860, 326)], [(990, 352), (930, 350), (880, 356)]):
            c.line(smooth(pts_, n=4), 2.4, (150, 160, 200, 120), caps=False)
        for (x, y) in [(960, 304), (1000, 342), (912, 340), (880, 300), (936, 362)]:
            c.circle(x, y, 4 + (x % 3), (196, 150, 120, 150))
        for xj in (700, 636):
            for dx in (-4, 4):
                c.arc(xj + dx, 300 - (xj - 636) * 0.065, 3, 8, 200, 330, crease, 1.2)
    # sleeve over the wrist
    c.rect(1000, 246, 1320, 428, sleeve, r=36)
    c.rect(990, 254, 1028, 420, tone(sleeve, 0.9), r=16)
    c.line([(1004, 262), (1004, 410)], 2, tone(sleeve, 0.8), caps=False)

def finger_top_y(x):
    return 322 + (806 - x) * 0.065 - (18 + (x - 584) / (806 - 584) * 6)

def gripping_hand(c, old, lt):
    """The small hand holding on, from below: a newborn's tiny fist, or (years later) her grown daughter's hand."""
    if old:
        skin, sleeve = SKIN, (124, 92, 130)
        fingers = [(646, 30, 0.0), (680, 31, 4.0), (713, 29, 2.0), (743, 25, -4.0)]    # x, width, length tweak
        palm_c, palm_w, palm_h = (694, 446), 74, 56
        wrist_y, base_y = 512, 396
        thumb = [(626, 440), (606, 410), (590 + 14 * math.sin(lt * 1.6), 384)]
        thumb_w = [16, 14, 12]
    else:
        skin, sleeve = BABY_SKIN, (250, 233, 184)
        fingers = [(666, 18, 0.0), (684, 19, 2.0), (701, 18, 1.0), (717, 15, -3.0)]
        palm_c, palm_w, palm_h = (692, 404), 44, 30
        wrist_y, base_y = 444, 372
        thumb = [(660, 410), (650, 394), (644, 382)]
        thumb_w = [9, 8, 7]
    crease = tone(skin, 0.74) + (170,)
    sq = 1.2 * math.sin(lt * 2.4)
    # forearm and sleeve
    fw = palm_w * (0.7 if old else 0.66)
    elbow = (palm_c[0] - (70 if old else 52), 800)
    arm_out, _, _ = tube_outline([(palm_c[0], wrist_y - 12), along((palm_c[0], wrist_y - 12), elbow, 0.5), elbow],
                                 [fw * (1.0 if old else 1.15), fw * (1.12 if old else 1.4), fw * (1.2 if old else 1.5)], cap_end=False)
    soft_shadow(c, arm_out, off=(5, 7), a=36)
    shaded(c, arm_out, skin, off=(2, 1))
    if not old:                                                                     # chubby wrist creases
        for k in range(2):
            c.arc(palm_c[0] - 2 - k * 4, wrist_y - 4 + k * 12, fw * 0.95, 5, 15, 165, crease, 2)
    # back of the hand
    back = smooth([(palm_c[0] - palm_w * 0.96, base_y + 2), (palm_c[0] - palm_w * 1.06, palm_c[1] - palm_h * 0.2),
                   (palm_c[0] - palm_w * 0.92, palm_c[1] + palm_h * 0.55), (palm_c[0] - fw, wrist_y),
                   (palm_c[0] + fw, wrist_y), (palm_c[0] + palm_w * 0.88, palm_c[1] + palm_h * 0.5),
                   (palm_c[0] + palm_w * 1.0, palm_c[1] - palm_h * 0.25), (palm_c[0] + palm_w * 0.86, base_y + 2)], n=6)
    soft_shadow(c, back, off=(4, 6), a=40)
    shaded(c, back, skin, off=(1.2, 2.4))
    if old:                                                                         # knuckles and tendons
        for (x, w, dl) in fingers:
            c.arc(x - w * 0.12, base_y + 14, w * 0.34, w * 0.18, 200, 340, tone(skin, 0.84), 1.6)
            c.line([(x - w * 0.12, base_y + 22), (palm_c[0] + (x - palm_c[0]) * 0.4, wrist_y - 8)], 1.6,
                   tone(skin, 1.06) + (150,), caps=False)
    # fingers: up across the near side of her finger and curling over the top
    for i, (x, w, dl) in enumerate(fingers):
        ft = finger_top_y(x) - dl - sq * (0.5 + 0.5 * (i % 2))
        pts = [(x - w * 0.15, base_y + 10), (x - w * 0.05, (base_y + ft) / 2 + 6), (x + w * 0.1, ft + w * 0.25),
               (x + w * 0.45, ft - w * 0.05)]
        hw_ = [w * 0.5, w * 0.48, w * 0.45, w * 0.42]
        out, _, _ = tube_outline(pts, hw_, cap_end=True)
        soft_shadow(c, out, off=(3, 5), a=44)
        shaded(c, out, skin, off=(1.2, 2.2))
        midy = (base_y + ft) / 2 + 6
        c.arc(x - w * 0.05, midy, w * 0.3, w * 0.14, 200, 340, crease, max(1, w * 0.07))     # middle knuckle
        nx_, ny_ = x + w * 0.42, ft - w * 0.02
        rot_ellipse(c, nx_, ny_, w * 0.26, w * 0.2, 0.6, (252, 232, 224))              # tiny nail on the curled tip
        if not old and i < 3:
            c.circle(x - w * 0.1, base_y + 16, w * 0.12, tone(skin, 0.86) + (160,))     # knuckle dimple
    # thumb, wrapping the near side from below
    t_out, _, _ = tube_outline(thumb, thumb_w, cap_end=True)
    shaded(c, t_out, skin, off=(1.0, 2.0))
    rot_ellipse(c, thumb[-1][0] - 1, thumb[-1][1] + 2, thumb_w[-1] * 0.6, thumb_w[-1] * 0.45, -1.0, (252, 232, 224))
    if old:
        c.rect(palm_c[0] - fw - 3, wrist_y - 12, palm_c[0] + fw + 3, wrist_y + 6, (196, 200, 210), r=6)   # watch strap
        c.rect(palm_c[0] - 13, wrist_y - 16, palm_c[0] + 13, wrist_y + 10, (170, 174, 186), r=4)
    # sleeve
    cuff0 = along((palm_c[0], wrist_y - 12), elbow, 0.22 if old else 0.1)
    sl_out, _, _ = tube_outline([cuff0, along(cuff0, elbow, 0.5), (elbow[0] - 8, elbow[1] + 40)],
                                [fw * 1.3, fw * 1.45, fw * 1.6], cap_end=False, cap_start=True)
    shaded(c, sl_out, sleeve, off=(2, 1))
    for k in range(5 if not old else 0):                                            # knitted cuff ribbing
        p = along(cuff0, elbow, 0.02 + k * 0.03)
        c.line([(p[0] - fw * 1.2, p[1] - 2), (p[0] + fw * 1.25, p[1] - 6)], 1.6, tone(sleeve, 0.88), caps=False)
    if old:
        c.line([(cuff0[0] - fw * 1.3, cuff0[1] + 4), (cuff0[0] + fw * 1.3, cuff0[1] - 2)], 3, tone(sleeve, 0.8), caps=False)

def hands_closeup(lt, old):
    """Close-up: a small hand wrapped around her finger (baby and mother; years later, daughter and mother)."""
    if old:
        arr = vgrad((208, 132, 112), (176, 104, 92)).astype(np.float32)
    else:
        arr = vgrad((208, 224, 244), (184, 206, 234)).astype(np.float32)
    img = to_img(arr)
    c = Canvas(img)
    rng = random.Random(6)
    if old:
        for y in range(40, 720, 90):
            c.line([(0, y), (1280, y + 30)], 2, (226, 156, 136, 140), caps=False)
        for x in range(60, 1280, 160):
            c.line([(x, 0), (x - 40, 720)], 2, (226, 156, 136, 120), caps=False)
    else:
        for _ in range(260):
            c.circle(rng.uniform(0, 1280), rng.uniform(0, 720), rng.uniform(2, 4), (255, 255, 255, 70))
    mother_hand(c, old, lt)
    gripping_hand(c, old, lt)
    out = finish(img, zoom=1.12 + 0.05 * ease(lt / 3), cx=760, cy=410)
    v = cached('vign_strong', _vignette_strong)
    return to_img(np.asarray(out, dtype=np.float32) * v)


def _vignette_strong():
    yy, xx = np.ogrid[0:OUT_H, 0:OUT_W]
    r2 = ((xx / OS - 680) / 680.0) ** 2 + ((yy / OS - 380) / 400.0) ** 2
    return (1 - 0.32 * np.clip(r2, 0, 2) / 2)[..., None].astype(np.float32)

def scene_hands_young(t):
    return hands_closeup(t - 42.0, False)

def scene_hands_old(t):
    return hands_closeup(t - 75.5, True)


# ---------------------------------------------------------------- her 70th birthday
def birthday_bg():
    arr = vgrad((240, 210, 182), (214, 178, 152)).astype(np.float32)
    img = to_img(arr)
    c = Canvas(img)
    c.rect(0, 600, W, H, (170, 122, 96))
    c.rect(0, 590, W, 602, (232, 214, 196))
    cols = [(232, 110, 100), (250, 200, 80), (110, 170, 220), (140, 196, 130), (236, 150, 190)]
    for row, (y0, sag) in enumerate([(70, 46), (120, 40)]):
        xs = np.linspace(-20, 1300, 24)
        pts = [(x, y0 + sag * (1 - ((x - 640) / 660) ** 2)) for x in xs]
        c.line(pts, 1.6, (120, 96, 90), caps=False)
        for k, (x, y) in enumerate(pts[:-1]):
            c.poly([(x + 6, y + 2), (x + 46, y + 2), (x + 26, y + 34)], cols[(k + row * 2) % 5])
    return img

def scene_birthday(t):
    lt = t - 56.5
    img = cached('bday_bg', birthday_bg).copy()
    c = Canvas(img)
    blew = ease((lt - 1.7) / 0.4)
    cheer = ease((lt - 2.3) / 0.3)
    # daughter (40) standing behind; husband (72) and her (70) seated; granddaughter (8) beside
    clap = math.sin(lt * 18) * cheer
    dh = {-1: (452 + 10 * clap, 420), 1: (488 - 10 * clap, 420)} if cheer > 0 else None
    draw_person(c, 470, 600, 40, hands=dh, style=dict(daughter_style(40), look=0.5, hat=(110, 170, 220), mouth='big' if cheer > 0.5 else 'smile',
                           eyes='happy' if cheer > 0.5 else 'open'))
    draw_person(c, 730, 660, 72, style=dict(father_style(72), nolegs=True, look=-0.65, hat=(250, 200, 80), mouth='big' if cheer > 0.5 else 'smile'))
    mouth = 'blow' if 1.5 < lt < 2.2 else ('big' if cheer > 0.5 else 'smile')
    draw_person(c, 600, 668, 70, mouth=mouth, style=dict(nolegs=True, lookdown=0.35 * (1 - cheer), hat=(232, 110, 100)))
    c.poly([(420, 470), (860, 470), (900, 600), (380, 600)], (244, 236, 226))
    c.ellipse(640, 470, 236, 26, (252, 248, 242))
    # cake
    c.rect(548, 424, 652, 470, (246, 196, 206), r=8)
    c.rect(568, 398, 632, 428, (252, 246, 240), r=6)
    for x in range(554, 650, 14):
        c.ellipse(x, 426, 7, 5, (252, 246, 240))
    for k, x in enumerate((582, 600, 618)):
        c.rect(x - 3, 378, x + 3, 400, [(110, 170, 220), (250, 200, 80), (140, 196, 130)][k])
        if blew < 1:
            lean = -6 * ease((lt - 1.5) / 0.3)
            s = 1 - blew
            c.circle(x + lean, 370, 9 * s, (255, 210, 120, 50))
            c.ellipse(x + lean, 370, 4 * s, 8 * s, (255, 200, 90))
        else:
            for j in range(3):
                u = (lt - 2.1 - j * 0.15) / 1.2
                if 0 < u < 1:
                    c.circle(x + 6 * math.sin(u * 7 + k), 372 - u * 50, 3 + 4 * u, (200, 200, 200, int(120 * (1 - u))))
    # a present
    c.rect(740, 430, 790, 468, (110, 170, 220), r=3)
    c.rect(762, 430, 768, 468, (250, 200, 80)); c.rect(740, 446, 790, 452, (250, 200, 80))
    # granddaughter with a balloon, cheering
    gst = dict(hair=(196, 150, 90), hairlen=0.5, shirt=(120, 170, 220), bun=0, look=-0.6, hat=(140, 196, 130), girl=True)
    if cheer > 0:
        gst['arms_up'] = True
    if cheer > 0.5:
        gst.update(mouth='big', eyes='happy')
    jg = draw_person(c, 910, 600, 8, style=gst)
    hx, hy = jg[1][2]
    bx, by = hx + 30 + 6 * math.sin(lt * 1.3), hy - 150 - 6 * math.cos(lt * 1.1)
    c.line([(hx, hy), (bx, by + 40)], 1.2, (120, 110, 110), caps=False)
    c.ellipse(bx, by, 30, 38, (236, 96, 110)); c.ellipse(bx - 10, by - 14, 7, 10, (255, 200, 210, 160))
    # confetti
    if cheer > 0:
        rng = random.Random(77)
        for k in range(90):
            ang = rng.uniform(-math.pi * 0.9, -math.pi * 0.1)
            sp = rng.uniform(180, 420)
            u = lt - 2.3
            x = 600 + math.cos(ang) * sp * u + 10 * math.sin(u * 6 + k)
            y = 380 + math.sin(ang) * sp * u + 160 * u * u
            col = [(232, 110, 100), (250, 200, 80), (110, 170, 220), (140, 196, 130), (236, 150, 190)][k % 5]
            c.ellipse(x, y, 4, 2 + 2 * abs(math.sin(u * 9 + k)), col)
    return finish(img, zoom=1.0 + 0.06 * ease(lt / 4), cx=640, cy=380)


# ---------------------------------------------------------------- the deathbed, with family moving
FLOWER_BED = (904, 449)

def scene_bed(t):
    lt = t - 68.0
    img = cached('bed_bg', bed_bg).copy()
    c = Canvas(img)
    sway = lambda i: math.radians(0.7) * math.sin(lt * 0.8 + i * 1.7)
    # daughter, seated, holding her mother's hand
    jd = draw_person(c, 520, 668, 58, eyes='closed', mouth='neutral', hands={1: (590, 457)},
                     style={'shirt': (124, 92, 130), 'nolegs': True, 'look': -0.6, 'wobble': sway(0)})
    # her husband, a hand on her shoulder
    draw_person(c, 648, 590, 61, eyes='closed', mouth='neutral', hands={-1: (562, 346, 'down')},
                style={'hairlen': 0, 'hair': (150, 145, 140), 'shirt': (70, 92, 124), 'pants': (60, 60, 70), 'male': True,
                       'sw': 0.92, 'glasses': 1, 'H': height_px(61) * 1.07, 'look': -0.5, 'bun': 0, 'wobble': sway(1)})
    # granddaughter, wiping her eye
    w = ease((lt - 4.4) / 0.5) * (1 - ease((lt - 6.0) / 0.6))
    gh = {1: (lerp(819, 783, w), lerp(412, 236, w))} if w > 0 else None
    draw_person(c, 770, 590, 27, eyes='closed', mouth='neutral', hands=gh,
                style={'hair': (196, 150, 90), 'hairlen': 0.78, 'shirt': (206, 146, 96), 'look': -0.4, 'wobble': sway(2)})
    # grandson, a hand on his little girl's shoulder
    draw_person(c, 880, 590, 31, eyes='closed', mouth='neutral', hands={1: (962, 381, 'down')},
                style={'hairlen': 0, 'hair': (70, 48, 34), 'shirt': (56, 128, 118), 'sw': 0.92, 'hw': 0.68, 'male': True,
                       'H': height_px(31) * 1.07, 'look': -0.3, 'bun': 0, 'wobble': sway(3), 'part': -1})
    # great-granddaughter places the yellow flower on the blanket
    reach = ease((lt - 1.2) / 1.4)
    back = ease((lt - 2.7) / 0.8)
    hand = (lerp(lerp(962, 918, reach), 956, back), lerp(lerp(452, 452, reach), 483, back))
    jc = draw_person(c, 994, 600, 7, eyes='open' if lt < 5 else 'closed', mouth='neutral', hands={-1: hand},
                     style=dict(GGC, look=-0.8, brow='worried', wobble=-math.radians(3.5) * reach * (1 - back) + sway(4)))
    draw_bed_and_elder(c, lt)
    if lt < 2.65:
        flower(c, hand[0] - 2, hand[1] - 24, 1.1, stem=22, lean=2)
        redraw_hand(c, jc, -1)
    else:
        c.line([(FLOWER_BED[0] + 4, FLOWER_BED[1] + 2), (FLOWER_BED[0] + 30, FLOWER_BED[1] + 6)], 2.6, (96, 150, 86))
        flower(c, FLOWER_BED[0], FLOWER_BED[1], 1.1)
    rng = random.Random(40)
    for k in range(40):
        x0, y0 = rng.uniform(1074, 1226), rng.uniform(150, 372)
        x = 1074 + (x0 - 1074 + 6 * math.sin(t * 0.8 + k)) % 152
        y = 150 + (y0 - 150 + 22 * t) % 214
        c.circle(x, y, rng.uniform(1.0, 2.2), (255, 255, 255, 220))
    redraw_forearm(c, jd, 1)

    mult, add, g = cached('bed_grade', bed_grade)
    arr = np.asarray(img, dtype=np.float32) * mult + add
    z = 1.0 + 0.06 * ease(lt / 9)
    out = finish(to_img(arr), zoom=z, cx=620, cy=380)
    # colour drains away, except the yellow flower; then the light goes, the flower last
    desat = ease((t - 79.6) / 1.8)
    f = ease((t - 81.3) / 1.6)
    f2 = ease((t - 82.6) / 0.8)
    if desat > 0 or f > 0:
        o = np.asarray(out, dtype=np.float32)
        w_, h_ = W / z, H / z
        cx = clamp(620, w_ / 2, W - w_ / 2); cy = clamp(380, h_ / 2, H - h_ / 2)
        fx = (FLOWER_BED[0] - (cx - w_ / 2)) * z
        fy = (FLOWER_BED[1] - (cy - h_ / 2)) * z
        yy, xx = np.ogrid[0:OUT_H, 0:OUT_W]
        mask = np.clip(1 - np.sqrt((xx - fx * OS) ** 2 + (yy - fy * OS) ** 2) / (34.0 * OS), 0, 1)[..., None] ** 0.6
        gray = (o @ np.array([0.299, 0.587, 0.114], np.float32))[..., None] * np.array([1.0, 1.0, 1.0], np.float32)
        o = o * mask + (o * (1 - desat) + gray * desat * 0.9) * (1 - mask)
        m = (1 - f) + f * mask * (1 - f2)
        o = o * m
        out = to_img(o)
    return out


# ---------------------------------------------------------------- title and end cards
PAGELLA = os.path.join(BASE, 'assets', 'fonts', 'texgyrepagella-italic.otf')
PAGELLA_R = os.path.join(BASE, 'assets', 'fonts', 'texgyrepagella-regular.otf')

def _font(key, path, size):
    return cached(key, lambda: ImageFont.truetype(path, int(round(size * OS))))

def scene_title(t):
    img = Image.new('RGB', (OUT_W, OUT_H), (14, 10, 14))
    d = ImageDraw.Draw(img, 'RGBA')
    a = ease((t - 0.4) / 1.0) * (1 - ease((t - 2.6) / 0.8))
    if a > 0:
        d.text((640 * OS, 352 * OS), 'A Life', font=_font('title_font', PAGELLA, 96),
               fill=(244, 232, 220, int(255 * a)), anchor='ms')
        wl = 120 * ease((t - 0.8) / 1.2)
        d.line([((640 - wl) * OS, 380 * OS), ((640 + wl) * OS, 380 * OS)], fill=(244, 232, 220, int(140 * a)),
               width=max(1, int(OS)))
    return img

def scene_end(t):
    img = Image.new('RGB', (OUT_W, OUT_H), (14, 10, 14))
    d = ImageDraw.Draw(img, 'RGBA')
    cream, muted = (244, 232, 220), (196, 182, 172)
    P = lambda x, y: (x * OS, y * OS)
    lw = max(1, int(OS))
    a = ease((t - 89.9) / 0.9) * (1 - ease((t - 92.4) / 0.7))
    if a > 0:
        d.text(P(640, 352), 'A Life', font=_font('end_font', PAGELLA, 60), fill=cream + (int(255 * a),), anchor='ms')
        d.line([P(580, 374), P(700, 374)], fill=cream + (int(120 * a),), width=lw)
    a = ease((t - 93.3) / 0.8) * (1 - ease((t - 97.0) / 0.8))
    if a > 0:
        lab = _font('cred_label', PAGELLA_R, 17)
        name = _font('cred_name', PAGELLA, 38)
        small = _font('end_font2', PAGELLA_R, 19)
        def spaced(y, txt):
            d.text(P(640, y), ' '.join(txt.upper()), font=lab, fill=muted + (int(230 * a),), anchor='ms')
        spaced(262, 'Director')
        d.text(P(640, 306), "Kevin O\u2019Shaughnessy", font=name, fill=cream + (int(255 * a),), anchor='ms')
        spaced(382, 'Producer')
        d.text(P(640, 426), 'Claude Fable 5.2', font=name, fill=cream + (int(255 * a),), anchor='ms')
        d.line([P(600, 470), P(680, 470)], fill=cream + (int(90 * a),), width=lw)
        d.text(P(640, 506), 'Music: Antonio Vivaldi, The Four Seasons', font=small,
               fill=muted + (int(220 * a),), anchor='ms')
        tiny = _font('end_font3', PAGELLA_R, 15)
        d.text(P(640, 536), 'John Harrison, violin, with the Wichita State University Chamber Players', font=tiny,
               fill=muted + (int(200 * a),), anchor='ms')
        d.text(P(640, 560), 'Recording licensed under Creative Commons BY-SA 3.0', font=tiny,
               fill=muted + (int(170 * a),), anchor='ms')
    return img


# ---------------------------------------------------------------- storybook finish
TINTS = {'neutral': (1.0, 1.0, 1.0), 'spring': (1.02, 1.0, 1.0), 'summer': (1.05, 1.01, 0.94),
         'autumn': (1.06, 0.97, 0.88), 'winter': (0.98, 0.99, 1.03)}
TINT_EDGES = [(8.6, 'spring'), (25.75, 'summer'), (44.75, 'autumn'), (61.6, 'winter'), (83.0, 'spring'),
              (89.6, 'neutral')]

def paper_texture():
    rng = np.random.default_rng(11)
    fine = rng.normal(0, 1, (OUT_H, OUT_W)).astype(np.float32)
    coarse = Image.fromarray(((rng.normal(0, 1, (H // 8, W // 8)) * 40 + 128).clip(0, 255)).astype(np.uint8))
    coarse = np.asarray(coarse.resize((OUT_W, OUT_H), Image.BICUBIC), dtype=np.float32) / 40 - 3.2
    fibre = Image.fromarray(((fine * 30 + 128).clip(0, 255)).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2 * OS))
    fibre = (np.asarray(fibre, dtype=np.float32) - 128) / 12 * 1.6
    yy, xx = np.ogrid[0:OUT_H, 0:OUT_W]
    vign = 1 - 0.17 * (((xx / OS - 640) / 640.0) ** 2 + ((yy / OS - 360) / 360.0) ** 2)
    return (1 + 0.009 * coarse + 0.013 * fibre + 0.005 * fine) * vign

def storybook(img, t):
    tint = np.array(TINTS['neutral'])
    for (edge, name) in TINT_EDGES:
        tint = tint + (np.array(TINTS[name]) - tint) * ease((t - (edge - 0.6)) / 1.2)
    tex = cached('paper', paper_texture)
    arr = np.asarray(img, dtype=np.float32) * tex[..., None] * tint.astype(np.float32)[None, None, :]
    return to_img(arr)



# ---------------------------------------------------------------- the embryo, curled in the womb
EMB0, EMB1, SHIFT, OLD_DUR = 9.0, 13.0, 4.0, 98.0

def embryo_bg():
    arr = vgrad((112, 28, 42), (64, 14, 28)).astype(np.float32)
    g = radial(640, 380, 620, 1.6)
    arr = arr + g[..., None] * np.array([130, 56, 52], np.float32)
    img = to_img(arr)
    c = Canvas(img)
    rng = random.Random(17)
    for _ in range(9):                                       # faint vessels in the uterine wall
        x0, y0 = rng.uniform(-100, 1380), rng.uniform(-100, 820)
        pts = [(x0, y0)]
        a = rng.uniform(0, 2 * math.pi)
        for _ in range(7):
            a += rng.uniform(-0.6, 0.6)
            pts.append((pts[-1][0] + math.cos(a) * 70, pts[-1][1] + math.sin(a) * 70))
        c.line(smooth(pts, n=5), rng.uniform(3, 7), (150, 40, 56, 90), caps=False)
    return img

def scene_embryo(T):
    le = T - EMB0
    img = cached('embryo_bg', embryo_bg).copy()
    c = Canvas(img)
    CX, CY = 640, 410
    g = 0.86 + 0.14 * ease(le / 4.5)                         # she grows a little as we watch
    rot = 0.05 * math.sin(le * 0.8)
    dy = 6 * math.sin(le * 1.2)
    cr, sr_ = math.cos(rot), math.sin(rot)
    def F(px, py):
        px, py = px * g, py * g
        return (CX + px * cr - py * sr_, CY + px * sr_ + py * cr + dy)
    def FL(pts):
        return [F(*p) for p in pts]
    skin = (246, 198, 188)
    sk_d = (226, 158, 152)
    # the amniotic sac and the placenta on its wall
    c.circle(CX, CY - 10, 262, (255, 196, 190, 40))
    rot_ellipse(c, 858, 572, 118, 62, -0.6, (150, 40, 56))
    rng = random.Random(3)
    for _ in range(40):
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, 1)
        c.circle(858 + math.cos(a) * 100 * r * math.cos(0.6) - math.sin(a) * 50 * r * math.sin(-0.6),
                 572 + math.cos(a) * 100 * r * math.sin(-0.6) + math.sin(a) * 50 * r * math.cos(0.6),
                 rng.uniform(3, 8), (176, 64, 76, 150))
    # umbilical cord, coiled, from her belly to the placenta
    cord = smooth(FL([(56, 66), (110, 120), (150, 112), (188, 150)]) + [(830, 552)], n=8)
    c.line(cord, 13 * g, (226, 146, 158))
    for i in range(1, len(cord) - 1, 3):
        p, q = cord[i - 1], cord[i + 1]
        n_ = unit(p, q)
        c.line([(cord[i][0] - n_[1] * 6 * g, cord[i][1] + n_[0] * 6 * g), (cord[i][0] + n_[1] * 6 * g + n_[0] * 4, cord[i][1] - n_[0] * 6 * g + n_[1] * 4)],
               2.2, (246, 196, 204, 200), caps=False)
    # far leg, curled body, near leg
    seg(c, F(-12, 104), F(52, 64), 22 * g, 17 * g, sk_d)
    seg(c, F(52, 64), F(30, 120), 17 * g, 12 * g, sk_d)
    spine = smooth([(-24, -24), (-58, 6), (-72, 52), (-60, 96), (-22, 124)], n=5)
    wid = [36 + 20 * math.sin(math.pi * min(1, k / (len(spine) - 1) * 1.15)) for k in range(len(spine))]
    body, _, _ = tube_outline(FL(spine), [w_ * g for w_ in wid], cap_end=True, cap_start=True)
    c.poly([(x + 2, y + 4) for x, y in body], sk_d)
    c.poly(body, skin)
    seg(c, F(-4, 108), F(64, 74), 26 * g, 20 * g, skin)               # thigh, knee drawn up
    seg(c, F(64, 74), F(42, 128), 20 * g, 14 * g, skin)               # shin
    fx, fy = F(56, 134)
    rot_ellipse(c, fx, fy, 17 * g, 9 * g, rot + 0.2, skin)
    # head, large at this stage
    hx, hy = F(0, -80)
    c.ellipse(hx + 2, hy + 4, 76 * g, 71 * g, sk_d)
    c.ellipse(hx, hy, 76 * g, 71 * g, skin)
    c.arc(hx - 6 * g, hy - 6 * g, 64 * g, 58 * g, 200, 290, (255, 228, 220, 150), 5 * g)   # light through the skin
    ex, ey = F(40, -76)
    c.circle(ex, ey + 2, 13 * g, (214, 140, 140, 90))                  # eyelid, closed
    c.arc(ex, ey, 12 * g, 7 * g, 10, 170, (150, 80, 84), 2.6 * g)
    nx_, ny_ = F(70, -54)
    c.circle(nx_, ny_, 8 * g, skin)
    c.arc(*F(56, -36), 9 * g, 5 * g, 20, 150, (176, 96, 100), 2 * g)
    c.arc(*F(-14, -70), 12 * g, 16 * g, 100, 280, sk_d, 3 * g)         # ear
    # arm, with the hand curled up by her face
    seg(c, F(-6, 4), F(38, 40), 16 * g, 13 * g, skin)
    seg(c, F(38, 40), F(50, -10), 13 * g, 10 * g, skin)
    hx2, hy2 = F(52, -20)
    c.circle(hx2, hy2, 11 * g, skin)
    for k in range(3):
        c.circle(*F(46 + k * 6, -32 + k * 2), 4 * g, skin)
    # her heartbeat, glowing in her chest
    ph = (le * 2.3) % 1.0
    p = math.exp(-ph * 7) + 0.6 * math.exp(-((ph - 0.18) % 1.0) * 9)
    cx_, cy_ = F(8, 36)
    c.circle(cx_, cy_, 26 * g, (255, 90, 90, int(30 + 70 * min(1, p))))
    c.circle(cx_, cy_, 13 * g, (255, 120, 110, int(40 + 90 * min(1, p))))
    # the sac's membrane, catching the light
    c.circle(CX, CY - 10, 262, None, outline=(255, 214, 208, 120), width=3)
    c.arc(CX, CY - 10, 246, 246, 200, 260, (255, 236, 230, 90), 6)
    rng = random.Random(9)
    for k in range(40):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        c.circle((x0 + 5 * le) % W, (y0 - 8 * le) % H, rng.uniform(1, 2.6), (255, 210, 200, int(50 + 40 * math.sin(le * 2 + k))))
    return finish(img, zoom=1.0 + 0.08 * ease(le / 4.5), cx=640, cy=400)


# ---------------------------------------------------------------- pregnancy: months later, in the room they painted
ROOM_MODE = None
P0, PDUR = 42.5, 4.5                     # film time where the scene starts, and its length
P1 = P0 + PDUR

def pregnancy_people(img, c, lt):
    her_x, his_x = 578, 698
    st_her = {'shirt': (238, 168, 150), 'pants': (70, 86, 120), 'hairlen': 0.62, 'bun': 0}
    st_him = dict(father_style(31))
    ph = 0.05 * 2 * math.pi                              # standing: both feet planted
    expr = 'o' if 2.15 < lt < 2.75 else ('big' if lt >= 2.75 else 'smile')
    st_her.update(mouth=expr, idle=1.0)
    st_him.update(mouth=expr, idle=1.0)
    kick = math.exp(-((lt - 2.2) / 0.12) ** 2) + 0.7 * math.exp(-((lt - 2.55) / 0.1) ** 2)
    reach = ease((lt - 0.5) / 0.9)
    def on_belly(g, t_, out=1.0):
        b = g['belly']
        return (b['bcx'] + b['f'] * b['brx'] * out * math.cos(t_), b['bcy'] + b['bry'] * out * math.sin(t_))
    hp_her = height_px(29.5)
    for xx, hp_ in ((her_x, hp_her), (his_x, st_him['H'])):
        c.ellipse(xx, GROUND + 2, 26 + hp_ * 0.12, 7, (0, 0, 0, 34))
    gh = {}
    draw_walker(c, her_x, GROUND, 29.5, ph, 1, style=st_her, stride=36, bump=1.0, kick=kick, out_geom=gh,
                near_hand=lambda g: on_belly(g, -0.7, 0.8))
    target = on_belly(gh, 0.32, 0.93)
    rest_him = (his_x - 14, gh['hip_y'] + 30)
    his_hand = (lerp(rest_him[0], target[0], reach), lerp(rest_him[1], target[1], reach))
    draw_walker(c, his_x, GROUND, 31.5, ph, -1, style=st_him, stride=40, near_hand=his_hand)
    kp = on_belly(gh, -0.55, 1.0)
    # a heart rises after the kick
    u = (lt - 2.5) / 1.6
    if 0 < u < 1:
        heart(c, 640 + 10 * math.sin(u * 5), 250 - 70 * u, 10 + 6 * u, (240, 110, 130, int(230 * math.sin(math.pi * u))))
    if 2.1 < lt < 2.9:                                   # little motion marks for the kick
        a = int(200 * math.sin(math.pi * (lt - 2.1) / 0.8))
        for k in (-1, 0, 1):
            ang = k * 0.5
            x0, y0 = kp[0] + 6, kp[1] - 4
            c.line([(x0 + 10 * math.cos(ang), y0 + 10 * math.sin(ang)), (x0 + 22 * math.cos(ang), y0 + 22 * math.sin(ang))],
                   2.5, (120, 90, 90, a))
    e_ = ease(lt / PDUR)
    return finish(img, zoom=1.04 + 0.12 * e_, cx=640 - 6 * e_, cy=360 + 14 * e_)

def scene_pregnancy(T):
    global ROOM_MODE
    lt = T - P0
    ROOM_MODE = lt
    try:
        img = scene_room(40.0 + 0.2 * lt)               # the finished, freshly painted room
    finally:
        ROOM_MODE = None
    return img

# ---------------------------------------------------------------- timeline
SEGS = [(0.0, 3.6, 'title'), (3.0, 9.5, 'cells'), (9.0, 13.0, 'baby'), (12.5, 16.5, 'crawl'),
        (16.0, 26.0, 'room'), (25.5, 30.5, 'date'), (30.0, 35.0, 'wedding'), (34.5, 39.0, 'room'),
        (38.5, 42.5, 'mother'), (42.0, 45.0, 'hands_young'), (44.5, 57.0, 'room'), (56.5, 60.5, 'birthday'),
        (60.0, 68.5, 'room'), (68.0, 76.0, 'bed'), (75.5, 78.5, 'hands_old'), (78.0, 83.5, 'bed'),
        (83.0, 90.0, 'room'), (89.5, 98.0, 'end')]

def compose_old(t):
    fns = dict(title=scene_title, cells=lambda t: scene_cells(t - 3.0), baby=scene_baby, crawl=scene_crawl,
               room=scene_room, date=scene_date, wedding=scene_wedding, mother=scene_mother,
               hands_young=scene_hands_young, hands_old=scene_hands_old, birthday=scene_birthday,
               bed=scene_bed, end=scene_end)
    act = [sg for sg in SEGS if sg[0] <= t < sg[1]] or [SEGS[-1]]
    if len(act) == 1:
        img = fns[act[0][2]](t)
    else:
        a, b = act[0], act[1]
        u = (t - b[0]) / (a[1] - b[0])
        img = Image.blend(fns[a[2]](t), fns[b[2]](t), ease(u))
    if 3.0 < t < 90.0:
        img = storybook(img, t)
    img = draw_caption(img, t)
    return img

def compose_v2(T):
    if T < EMB0:
        img = compose_old(T)
    elif T < EMB1 + 0.5:
        img = storybook(scene_embryo(T), 9.0)
        if T < EMB0 + 0.5:
            img = Image.blend(storybook(scene_cells(T - 3.0), T), img, ease((T - EMB0) / 0.5))
        if T >= EMB1:
            tb = T - SHIFT
            baby = draw_caption(storybook(scene_baby(tb), tb), tb)
            img = Image.blend(img, baby, ease((T - EMB1) / 0.5))
    else:
        img = compose_old(T - SHIFT)
    return img

# The ending, lengthened: the last fade of the deathbed plays at about half speed, the screen
# stays dark for a moment, and the new spring fades in from black, its colour returning with the music.
D0, D_FADE, D_HOLD, D_ADD = 88.1, 7.0, 1.4, 5.0
D_IN = 2.6                                # the new spring: seconds for light and colour to return
D_OLD0, D_OLD_LEN = 79.6, 3.85            # the stretch of the original deathbed timeline being slowed

def render(fi):
    global NOW
    T = fi / FPS
    NOW = T
    if T >= D0:
        if T < D0 + D_FADE:
            old = D_OLD0 + (T - D0) * D_OLD_LEN / D_FADE
            return storybook(scene_bed(old), old).tobytes()
        if T < D0 + D_FADE + D_HOLD:
            return Image.new('RGB', (OUT_W, OUT_H)).tobytes()
        T -= D_ADD
        old = T - SHIFT - PDUR
        if old < 83.0 + D_IN:                            # light first, then the colour comes back
            arr = np.asarray(storybook(scene_room(old), old), dtype=np.float32)
            u = (old - 83.0) / D_IN
            gray = (arr @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
            arr = gray + (arr - gray) * ease((u - 0.2) / 0.8)
            return draw_caption(to_img(arr * ease(u / 0.85)), old).tobytes()
    if T < P0:
        img = compose_v2(T)
    elif T < P1 + 0.5:
        img = storybook(scene_pregnancy(T), 38.6)
        if T < P0 + 0.5:                                 # from the painting
            to = T - SHIFT
            img = Image.blend(storybook(scene_room(to), to), img, ease((T - P0) / 0.5))
        if T >= P1:                                      # to the new baby
            to = T - SHIFT - PDUR
            img = Image.blend(img, storybook(scene_mother(to), to), ease((T - P1) / 0.5))
    else:
        img = compose_v2(T - PDUR)
    return img.tobytes()


# ---------------------------------------------------------------- soundtrack
# Recording: Vivaldi, The Four Seasons, John Harrison (violin) with the Wichita State University
# Chamber Players, CC BY-SA 3.0. Cues: (file, film time, offset into the recording, length, gain,
# fade in, fade out).
REAL_MUSIC = True
REC_DIR = os.path.join(BASE, 'assets', 'music')
BUILD = os.path.join(BASE, 'build')
CUES = [
    ('spring1', 13.0, 0.55, 7.83, 1.3, 0.25, 0.06),      # Spring I, first phrase: the newborn, crawling
    ('spring1', 20.8, 15.93, 9.3, 1.3, 0.03, 0.9),       # ... second phrase: first steps, growing up
    ('summer1', 29.5, 2.3, 4.9, 3.0, 0.3, 0.6),          # Summer I, the languid opening: the first date
    ('spring3', 34.1, 1.9, 15.8, 1.5, 0.3, 0.4),         # Spring III, pastoral dance: wedding, painting, pregnancy
    ('spring3', 49.5, 17.3, 3.9, 0.8, 0.4, 0.9),         # ... continuing more softly over the newborn's hand
    ('autumn1', 53.2, 0.4, 17.2, 1.5, 0.3, 0.8),         # Autumn I: the family years, falling silent as he goes
    ('winter2', 73.1, 1.3, 23.4, 2.6, 0.5, 7.0),         # Winter II, Largo: alone, and the deathbed, fading into the dark
    ('spring1', 96.5, 0.55, 14.65, 1.2, 2.6, 2.4),        # Spring I returns: the new child, and the credits
]

def load_rec(name):
    wav = os.path.join(BUILD, name + '.wav')
    if not os.path.exists(wav):               # decode the MP3 once
        os.makedirs(BUILD, exist_ok=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(REC_DIR, name + '.mp3'),
                        '-ar', '44100', '-ac', '2', wav], check=True)
    with wave.open(wav) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float64) / 32767

# Spring: the opening ritornello of Vivaldi's La Primavera (RV 269, public domain), arranged.
# Summer, Autumn and Winter: original themes written in the style, keys and textures of the other
# three concertos (G minor sighs and a driving tutti; an F major harvest dance; pizzicato "rain"
# under a slow violin line in E-flat).
CHORDS = {'E': (4, [4, 8, 11]), 'B7': (11, [11, 3, 6, 9]), 'B': (11, [11, 3, 6]), 'Em': (4, [4, 7, 11]),
          'Gm': (7, [7, 10, 2]), 'D': (2, [2, 6, 9]), 'D7': (2, [2, 6, 9, 0]), 'Bb': (10, [10, 2, 5]),
          'F': (5, [5, 9, 0]), 'Eb': (3, [3, 7, 10]), 'Cm': (0, [0, 3, 7]), 'G': (7, [7, 11, 2]),
          'C': (0, [0, 4, 7]), 'Ab': (8, [8, 0, 3]), 'Bb7': (10, [10, 2, 5, 8])}
TA = ([('E', 4)], [(80, 1), (80, 1), (80, 1), (78, .5), (76, .5)])
TB = ([('B7', 4)], [(83, 3), (83, .5), (81, .5)])
FIN = ([('E', 4)], [(76, 4)])
SL1 = ([('Gm', 4)], [(74, 1.5), (72, .5), (70, 1.5), (69, .5)])
SL2 = ([('D7', 4)], [(72, 1), (70, .5), (69, .5), (66, 2)])
ST1 = ([('Bb', 4)], [(82, .5)] * 4 + [(77, .25), (75, .25), (74, .25), (72, .25), (70, .5), (74, .5)])
ST2 = ([('F', 4)], [(81, .5)] * 4 + [(77, .5), (72, .5), (69, 1)])
ST3 = ([('Gm', 4)], [(79, .5)] * 4 + [(74, .25), (72, .25), (70, .25), (69, .25), (67, .5), (70, .5)])
ST4 = ([('Eb', 2), ('F', 2)], [(75, .5), (79, .5), (77, .5), (75, .5), (74, .5), (72, .5), (70, 1)])
ST5 = ([('Bb', 4)], [(82, .25), (81, .25), (79, .25), (77, .25), (75, .25), (74, .25), (72, .25), (70, .25), (77, 1), (70, 1)])
SA1 = ([('Gm', 4)], [(74, 2), (75, 1), (74, 1)])
SA2 = ([('Cm', 2), ('D', 2)], [(72, 1.5), (70, .5), (69, 2)])
SA3 = ([('G', 4)], [(67, 4)])
A1 = ([('F', 4)], [(72, .5), (72, .5), (77, 1), (77, .5), (76, .5), (77, .5), (79, .5)])
A2 = ([('C', 4)], [(81, 1), (79, 1), (76, 1), (72, 1)])
A3 = ([('F', 4)], [(72, .5), (72, .5), (77, 1), (77, .5), (76, .5), (77, .5), (81, .5)])
A4 = ([('C', 2), ('F', 2)], [(79, 1), (76, 1), (77, 2)])
B1 = ([('Bb', 4)], [(82, 1), (81, .5), (79, .5), (77, 1), (74, 1)])
B2 = ([('F', 4)], [(81, 1), (79, .5), (77, .5), (76, 1), (72, 1)])
B3 = ([('Gm', 2), ('C', 2)], [(79, .5), (77, .5), (76, .5), (74, .5), (72, 1), (76, 1)])
B4 = ([('F', 4)], [(77, 3), (72, 1)])
W1 = ([('Eb', 4)], [(79, 2), (77, 1), (75, 1)])
W2 = ([('Ab', 2), ('Bb', 2)], [(77, 1.5), (75, .5), (74, 2)])
W3 = ([('Cm', 4)], [(75, 1), (79, 1), (84, 1.5), (82, .5)])
W4 = ([('Ab', 2), ('Bb7', 2)], [(80, 1), (79, 1), (77, 2)])
W5 = ([('Eb', 2), ('Cm', 2)], [(79, 1.5), (77, .5), (75, 1), (72, 1)])
W6 = ([('Bb', 2), ('Eb', 2)], [(74, 2), (75, 2)])
SECTIONS = [  # start, end, bars, velocity per bar, style
    (9.0, 25.75, [TA, TB, TA, TB, TA, TB, TA, TB], [1, 1, 1, 1, .45, .45, .45, .45], 'spring'),
    (25.75, 30.25, [SL1, SL2], [.7, .7], 'languid'),
    (30.25, 38.75, [ST1, ST2, ST3, ST4, ST5], [1] * 5, 'tutti'),
    (38.75, 44.75, [SA1, SA2, SA3], [.6, .6, .55], 'adagio'),
    (44.75, 61.6, [A1, A2, A3, A4, B1, B2, B3, B4, A4], [.8, .8, .8, .8, .85, .85, 1, 1, 1], 'dance'),
    (64.6, 83.2, [W1, W2, W3, W4, W5, W6], [.6, .6, .6, .55, .5, .45], 'largo'),
    (83.2, 89.7, [TA, TB, FIN], [.45, .45, .4], 'epilogue'),
]

def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def make_audio_old(path):
    sr = 44100
    n = int(DUR * sr) + sr * 4
    ML, MR = np.zeros(n), np.zeros(n)      # music bus
    AL, AR = np.zeros(n), np.zeros(n)      # ambience bus
    rng = np.random.default_rng(3)

    def add(start, sig, pan, amb=False):
        s0 = int((start + rng.normal(0, 0.004)) * sr)
        if s0 < 0 or s0 >= n:
            return
        e = min(n, s0 + len(sig))
        Lb, Rb = (AL, AR) if amb else (ML, MR)
        Lb[s0:e] += sig[:e - s0] * (1 - pan) * 0.5
        Rb[s0:e] += sig[:e - s0] * (1 + pan) * 0.5

    def strings(start, midi, dur, vel, pan=0.0, attack=0.06, voices=3, bright=1.0):
        f = mtof(midi)
        ln = int((dur + 0.35) * sr)
        tt = np.arange(ln) / sr
        vib = 1 + 0.0035 * np.sin(2 * np.pi * 5.4 * tt) * np.clip((tt - 0.15) / 0.3, 0, 1)
        sig = np.zeros(ln)
        nh = max(1, min(12, int(5000 / f)))
        for v in range(voices):
            det = 1 + (v - (voices - 1) / 2) * 0.0028
            ph = 2 * np.pi * f * det * np.cumsum(vib) / sr
            for k in range(1, nh + 1):
                sig += (1 / k) * np.exp(-k / (4.5 * bright)) * np.sin(k * ph + v * 1.7 + k)
        env = np.clip(tt / attack, 0, 1) * np.where(tt > dur, np.exp(-(tt - dur) * 9), 1.0)
        add(start, sig * env * vel / voices, pan)

    def harpsichord(start, midi, vel, pan=0.0):
        f = mtof(midi)
        tt = np.arange(int(2.2 * sr)) / sr
        sig = sum((1 / k ** 0.75) * np.exp(-tt * (2.2 + 0.7 * k)) * np.sin(2 * np.pi * f * k * (1 + 0.0006 * k) * tt)
                  for k in range(1, 13))
        add(start, sig * np.clip(tt / 0.002, 0, 1) * vel, pan)

    def pizz(start, midi, vel, pan=0.0):
        f = mtof(midi)
        tt = np.arange(int(1.2 * sr)) / sr
        sig = sum((1 / k) * np.exp(-tt * (5 + 2.5 * k)) * np.sin(2 * np.pi * f * k * tt) for k in range(1, 8))
        add(start, sig * np.clip(tt / 0.003, 0, 1) * vel, pan)

    def bass_of(root):
        b = 40 + ((root - 4) % 12)
        return b - 12 if b > 46 else b

    # ---- prologue: swelling E major chord; a harpsichord note at each cell division
    for m in (40, 52, 56, 59, 64):
        strings(0.4, m, 8.4, 0.05, -0.1, attack=4.0, voices=3, bright=0.6)
    for k, m in enumerate([76, 80, 83, 88, 92]):
        harpsichord(DIV_T[k] + 3.4, m, 0.18, 0.2)

    for (s0, s1, bars, vels, style) in SECTIONS:
        bar = (s1 - s0) / len(bars)
        beat = bar / 4
        if style == 'dance':                        # a folk drone under the harvest dance
            strings(s0, 41, s1 - s0, 0.06, -0.2, attack=0.6, bright=0.6)
            strings(s0, 48, s1 - s0, 0.04, 0.2, attack=0.6, bright=0.6)
        for bi, (chs, mel) in enumerate(bars):
            bt = s0 + bi * bar
            v = vels[bi]
            ct = bt
            for (ch, nb) in chs:
                root, pcs = CHORDS[ch]
                bass = bass_of(root)
                inner = sorted(59 + ((p - 59) % 12) for p in pcs)
                if style in ('spring', 'tutti'):
                    for e in range(int(nb * 2)):
                        strings(ct + e * beat / 2, bass, beat * 0.38, 0.30 * v, -0.35, attack=0.012, voices=2)
                    if style == 'tutti':
                        for e in range(int(nb * 4)):
                            for m in inner[:2]:
                                strings(ct + e * beat / 4, m, beat * 0.2, 0.035 * v, 0.25, attack=0.01, voices=1)
                    strings(ct, inner[0], nb * beat * 0.95, 0.08 * v, 0.25, attack=0.08)
                    strings(ct, inner[-1], nb * beat * 0.95, 0.07 * v, 0.3, attack=0.08)
                    for hb in range(0, int(nb), 1 if style == 'tutti' else 2):
                        for i, m in enumerate(inner):
                            harpsichord(ct + hb * beat + 0.012 * i, m, 0.10 * v, -0.15)
                elif style == 'languid':
                    strings(ct, bass, nb * beat, 0.22 * v, -0.3, attack=0.4)
                    for m in inner:
                        strings(ct, m, nb * beat, 0.06 * v, 0.2, attack=0.5, bright=0.7)
                    for hb in (0, 2):
                        for i, m in enumerate(inner + [inner[0] + 12]):
                            harpsichord(ct + hb * beat + 0.09 * i, m, 0.07 * v, -0.1)
                elif style == 'adagio':
                    for q in range(int(nb)):
                        pizz(ct + q * beat, bass, 0.35 * v, -0.3)
                    for m in inner:
                        strings(ct, m, nb * beat, 0.06 * v, 0.2, attack=0.4, bright=0.7)
                elif style == 'dance':
                    for q in range(int(nb)):
                        if q % 2 == 0:
                            strings(ct + q * beat, bass, beat * 0.55, 0.32 * v, -0.3, attack=0.012, voices=2)
                        else:
                            for i, m in enumerate(inner):
                                strings(ct + q * beat, m, beat * 0.35, 0.06 * v, 0.2, attack=0.01, voices=2)
                                harpsichord(ct + q * beat + 0.01 * i, m, 0.08 * v, -0.15)
                elif style == 'largo':
                    strings(ct, bass, nb * beat, 0.16 * v, -0.3, attack=0.5, bright=0.6)
                    strings(ct, inner[1], nb * beat, 0.07 * v, 0.1, attack=0.6, bright=0.7)
                    cyc = inner + [inner[1] + 12]
                    for e in range(int(nb * 2)):
                        pizz(ct + e * beat / 2, cyc[e % len(cyc)], 0.16 * v, 0.3 if e % 2 else -0.1)
                else:  # epilogue
                    for q in range(int(nb)):
                        strings(ct + q * beat, bass, beat * 0.7, 0.18 * v, -0.3, attack=0.03, voices=2)
                    for i, m in enumerate(inner):
                        harpsichord(ct + 0.012 * i, m, 0.09 * v, -0.15)
                        strings(ct, m, nb * beat * 0.95, 0.05 * v, 0.2, attack=0.2)
                ct += nb * beat
            mt = bt
            for (m, nb) in mel:
                legato = 0.95 if style in ('languid', 'adagio', 'largo') else 0.8 if style != 'dance' else 0.7
                atk = 0.15 if style in ('languid', 'adagio', 'largo') else 0.03
                strings(mt, m, nb * beat * legato, 0.24 * v, 0.1, attack=atk)
                if style == 'tutti':
                    strings(mt, m - 12, nb * beat * legato, 0.12 * v, -0.1, attack=atk)
                if style in ('spring', 'epilogue') and nb >= 3:
                    for k in range(8):
                        harpsichord(mt + beat * 1.5 + k * 0.07, m + 12 + (2 if k % 2 else 0), 0.05 * v, 0.4)
                mt += nb * beat
    for i, m in enumerate([40, 52, 59, 64, 68, 71, 76]):     # a last E major chord
        strings(89.3 + 0.12 * i, m, 6.8, 0.07, 0.0, attack=0.8, bright=0.7)

    # ---- ambience
    def bandnoise(dur, lo, hi):
        m = int(dur * sr)
        spec = np.fft.rfft(rng.normal(0, 1, m))
        fr = np.fft.rfftfreq(m, 1 / sr)
        spec[(fr < lo) | (fr > hi)] = 0
        x = np.fft.irfft(spec, m)
        return x / (np.abs(x).max() + 1e-9)

    def birds(t0, t1, density=1.0, vel=0.05):
        tt = t0 + rng.uniform(0.1, 0.6)
        while tt < t1:
            nc = rng.integers(2, 6)
            f0 = rng.uniform(2800, 4600)
            pan = rng.uniform(-0.7, 0.7)
            ct = tt
            for _ in range(nc):
                d = rng.uniform(0.05, 0.12)
                u = np.arange(int(d * sr)) / sr
                f = f0 * (1 + rng.uniform(-0.3, 0.3) * u / d) + 180 * np.sin(2 * np.pi * 32 * u)
                sig = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.sin(np.pi * u / d)
                add(ct, sig * vel, pan, True)
                ct += d + rng.uniform(0.03, 0.09)
            tt += rng.uniform(0.6, 1.6) / density

    def crickets(t0, t1, vel=0.025):
        for (rate, f, pan) in ((0.52, 4700, -0.5), (0.61, 4300, 0.5)):
            tt = t0
            while tt < t1:
                for p in range(3):
                    u = np.arange(int(0.022 * sr)) / sr
                    add(tt + p * 0.045, np.sin(2 * np.pi * f * u) * np.sin(np.pi * u / 0.022) * vel, pan, True)
                tt += rate * rng.uniform(0.9, 1.1)

    def bell(t0, f0, vel=0.1, pan=0.0):
        u = np.arange(int(5 * sr)) / sr
        sig = sum(a * np.exp(-u * d) * np.sin(2 * np.pi * f0 * r * u)
                  for r, a, d in [(0.5, .6, .6), (1, 1, .9), (1.2, .6, 1.2), (1.5, .4, 1.5), (2, .35, 1.8),
                                  (2.5, .2, 2.4), (3, .15, 3)])
        add(t0, sig / 2.5 * vel * np.clip(u / 0.003, 0, 1), pan, True)

    def wind(t0, t1, lo, hi, vel):
        x = bandnoise(t1 - t0, lo, hi)
        u = np.arange(len(x)) / sr
        env = (0.55 + 0.45 * np.sin(2 * np.pi * 0.11 * u + 1.3) * np.sin(2 * np.pi * 0.07 * u))
        env *= np.clip(u / 1.5, 0, 1) * np.clip((t1 - t0 - u) / 1.5, 0, 1)
        add(t0, x * env * vel, 0.0, True)

    def bursts(t0, t1, lo, hi, vel, every=(0.8, 2.0), dur=(0.2, 0.5)):
        tt = t0 + rng.uniform(*every)
        while tt < t1:
            d = rng.uniform(*dur)
            x = bandnoise(d, lo, hi)
            u = np.arange(len(x)) / sr
            add(tt, x * np.sin(np.pi * u / d) ** 2 * vel, rng.uniform(-0.6, 0.6), True)
            tt += rng.uniform(*every)

    def applause(t0, t1, vel=0.05):
        for person in range(6):
            tt = t0 + rng.uniform(0, 0.15)
            pan = rng.uniform(-0.6, 0.6)
            while tt < t1:
                x = bandnoise(0.025, 700, 4500)
                u = np.arange(len(x)) / sr
                fade = 1 - (tt - t0) / (t1 - t0) * 0.6
                add(tt, x * np.exp(-u * 160) * vel * fade, pan, True)
                tt += rng.uniform(0.17, 0.26)

    def clock(t0, t1, vel=0.05):
        k = 0
        tt = t0
        while tt < t1:
            u = np.arange(int(0.03 * sr)) / sr
            f = 2600 if k % 2 == 0 else 1900
            sig = (np.sin(2 * np.pi * f * u) + 0.5 * bandnoise(0.03, 3000, 9000)[:len(u)]) * np.exp(-u * 220)
            add(tt, sig * vel, 0.35, True)
            tt += 1.0
            k += 1

    birds(9.0, 25.75, 1.0, 0.045)
    birds(34.5, 42.5, 0.6, 0.03)
    crickets(25.5, 30.4)
    for i, (tt, f) in enumerate([(30.3, 415.3), (31.1, 329.6), (31.9, 246.9), (32.7, 415.3), (33.5, 329.6)]):
        bell(tt, f, 0.07 if REAL_MUSIC else 0.11, -0.4 + 0.2 * i)
    wind(44.5, 61.8, 300, 1400, 0.035)
    bursts(44.8, 56.4, 2200, 7000, 0.02)
    x = bandnoise(0.06, 200, 6000)
    add(58.8, x * np.exp(-np.arange(len(x)) / sr * 60) * 0.18, 0.2, True)        # party popper
    applause(58.9, 60.4)
    wind(64.0, 83.2, 600, 2600, 0.03)
    clock(61.6, 64.6, 0.16)           # only the clock, in the silence after he is gone
    clock(64.6, 81.6)                 # it stops when she does
    for k in range(3):                                                               # pencil mark
        x = bandnoise(0.12, 3000, 8000)
        u = np.arange(len(x)) / sr
        add(86.25 + k * 0.14, x * np.sin(np.pi * u / 0.12) * 0.035, -0.5, True)
    birds(83.0, 89.5, 1.1, 0.045)

    # ---- mix: hall reverb, buses, fades
    tir = np.arange(int(2.2 * sr)) / sr
    ir = rng.normal(0, 1, len(tir)) * np.exp(-tir / 0.55)
    ir /= np.sqrt((ir ** 2).sum())
    nfft = 1 << int(np.ceil(np.log2(n + len(ir))))
    IR = np.fft.rfft(ir, nfft)
    def verb(ch, wetk):
        return ch + wetk * np.fft.irfft(np.fft.rfft(ch, nfft) * IR, nfft)[:n]
    ML, MR = verb(ML, 0.3), verb(MR, 0.3)
    AL, AR = verb(AL, 0.15), verb(AR, 0.15)
    tm = np.arange(n) / sr                # the music falls silent as he fades away
    hush = np.where(tm < 63.0, 1 - np.clip((tm - 61.2) / 0.7, 0, 1), np.clip((tm - 64.5) / 0.15, 0, 1))
    ML, MR = ML * hush, MR * hush
    mpk = max(np.abs(ML).max(), np.abs(MR).max())
    if REAL_MUSIC:                         # keep only the synthesized prologue; the recording takes over
        keep = 1 - np.clip((tm - 8.5) / 0.5, 0, 1)
        ML, MR = ML * keep, MR * keep
    L = ML / mpk * 0.68 + AL
    R = MR / mpk * 0.68 + AR
    t = np.arange(n) / sr
    master = np.clip(t / 0.4, 0, 1) * (1 - np.clip((t - 95.4) / 2.4, 0, 1))
    L, R = (L * master)[:int(DUR * sr)], (R * master)[:int(DUR * sr)]
    pk = max(np.abs(L).max(), np.abs(R).max())
    if pk > 0.95:
        L, R = L / pk * 0.95, R / pk * 0.95
    data = (np.stack([L, R], 1) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(data.tobytes())



def make_audio_v2(path):
    global DUR
    sr = 44100
    saved, DUR = DUR, OLD_DUR
    tmp = path + '.old.wav'
    make_audio_old(tmp)
    DUR = saved
    with wave.open(tmp) as w:
        old = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float64) / 32767
    n = int(DUR * sr)
    out = np.zeros((n, 2))
    i9, i13 = int(EMB0 * sr), int(EMB1 * sr)
    pre = old[:i9].copy()
    fo = int(0.35 * sr)
    pre[-fo:] *= np.linspace(1, 0, fo)[:, None]
    out[:i9] = pre
    post = old[i9:].copy()
    fi_ = int(0.015 * sr)
    post[:fi_] *= np.linspace(0, 1, fi_)[:, None]
    m = min(len(post), n - i13)
    out[i13:i13 + m] = post[:m]
    # the womb: a held E major chord, a heartbeat, and a soft whoosh of blood
    t0, t1 = EMB0 - 0.4, EMB1 + 0.3
    a, b = int(t0 * sr), int(t1 * sr)
    tt = np.arange(b - a) / sr + t0
    env = np.clip((tt - t0) / 0.9, 0, 1) * np.clip((t1 - tt) / 0.7, 0, 1)
    pad = sum(np.sin(2 * np.pi * f * tt + k) * amp for k, (f, amp) in
              enumerate([(82.41, 1.0), (123.47, 0.7), (164.81, 0.55), (207.65, 0.35), (246.94, 0.25)]))
    pad *= env * (1 + 0.08 * np.sin(2 * np.pi * 0.4 * tt))
    beat = np.zeros_like(tt)
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 1, len(tt))
    spec = np.fft.rfft(noise)
    fr = np.fft.rfftfreq(len(noise), 1 / sr)
    spec[fr > 380] = 0
    whoosh = np.fft.irfft(spec, len(noise))
    whoosh /= np.abs(whoosh).max() + 1e-9
    wenv = np.zeros_like(tt)
    bt = EMB0 + 0.15
    while bt < EMB1 - 0.1:
        for off, f, amp in ((0.0, 46, 1.0), (0.16, 62, 0.7)):
            u = tt - (bt + off)
            m_ = u > 0
            beat[m_] += amp * (np.sin(2 * np.pi * f * u[m_]) * np.exp(-u[m_] * 16) +
                               0.5 * np.sin(2 * np.pi * f * 1.5 * u[m_]) * np.exp(-u[m_] * 24))
        u = tt - bt
        wenv += np.where(u > 0, np.exp(-np.clip(u, 0, None) * 5) * np.clip(u * 30, 0, 1), 0)
        bt += 1 / 2.3
    beat *= np.clip((tt - EMB0) / 0.4, 0, 1) * np.clip((EMB1 + 0.1 - tt) / 0.4, 0, 1)
    womb = pad / np.abs(pad).max() * 0.16 + beat / (np.abs(beat).max() + 1e-9) * 0.42 + whoosh * wenv * 0.05 * env
    out[a:b, 0] += womb
    out[a:b, 1] += womb
    pk = np.abs(out).max()
    if pk > 0.97:
        out *= 0.97 / pk
    data = (out * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(data.tobytes())


AGAP = 42.75                               # splice point: the boundary between two Summer sections

def make_audio(path):
    """The 102 s mix with a 4.5 s gap opened for the pregnancy, filled by a soft string bridge,
    a quick fetal heartbeat (echoing the embryo scene) and a little thump for the kick."""
    global DUR
    sr = 44100
    saved, DUR = DUR, 102.0
    tmp = path + '.v2.wav'
    make_audio_v2(tmp)
    DUR = saved
    with wave.open(tmp) as w:
        v2 = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float64) / 32767
    n = int((DUR - D_ADD) * sr)
    out = np.zeros((n, 2))
    g0, g1 = int(AGAP * sr), int((AGAP + PDUR) * sr)
    pre = v2[:g0].copy()
    fo = int(0.4 * sr)
    pre[-fo:] *= np.linspace(1, 0, fo)[:, None]
    out[:g0] = pre
    post = v2[g0:].copy()
    fi_ = int(0.25 * sr)
    post[:fi_] *= np.linspace(0, 1, fi_)[:, None]
    m = min(len(post), n - g1)
    out[g1:g1 + m] += post[:m]
    # string bridge: B-flat major settling towards G minor, which the next section begins in
    t0, t1 = AGAP - 0.45, AGAP + PDUR + 0.3
    a, b = int(t0 * sr), int(t1 * sr)
    tt = np.arange(b - a) / sr
    L_, R_ = np.zeros(b - a), np.zeros(b - a)
    def string(start, f, dur, vel, pan):
        u = tt - start
        env = np.clip(u / 0.6, 0, 1) * np.clip((dur - u) / 0.8, 0, 1) * (u > 0)
        vib = 1 + 0.003 * np.sin(2 * np.pi * 5.2 * u)
        ph = 2 * np.pi * f * np.cumsum(vib) / sr
        sig = sum((1 / k) * np.exp(-k / 4.0) * np.sin(k * ph + k) for k in range(1, 9))
        L_[:] += sig * env * vel * (1 - pan) / 2
        R_[:] += sig * env * vel * (1 + pan) / 2
    def pluck(start, f, vel, pan):
        u = tt - start
        m_ = u > 0
        sig = np.zeros_like(tt)
        sig[m_] = sum((1 / k) * np.exp(-u[m_] * (2.4 + 0.8 * k)) * np.sin(2 * np.pi * f * k * u[m_]) for k in range(1, 9))
        L_[:] += sig * vel * (1 - pan) / 2
        R_[:] += sig * vel * (1 + pan) / 2
    mt = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    for m, vel, pan in ((46, 0.22, -0.3), (53, 0.08, -0.1), (62, 0.07, 0.2), (65, 0.06, 0.3)):   # Bb chord
        string(0.0, mt(m), 2.9, vel, pan)
    for m, vel, pan in ((43, 0.22, -0.3), (50, 0.08, -0.1), (58, 0.07, 0.2), (62, 0.06, 0.3)):   # G minor
        string(2.6, mt(m), PDUR + 0.6 - 2.6, vel, pan)
    for k, m in enumerate([70, 74, 77, 82, 77, 74, 70, 74, 79, 82, 79, 74]):                      # harpsichord, gently
        pluck(0.45 + k * 0.36, mt(m), 0.07, 0.25 if k % 2 else -0.1)
    melody = [(1.2, 77, 0.9), (2.1, 79, 0.5), (2.6, 74, 1.4)]                                     # a small violin line
    for st, m, d in melody:
        string(st, mt(m), d + 0.4, 0.09, 0.1)
    # the baby's heartbeat, quick and soft, as he lays his hand on her
    hb = np.zeros_like(tt)
    bt = AGAP + 0.9 - t0
    while bt < PDUR + 0.2:
        for off, f, amp in ((0.0, 52, 1.0), (0.13, 66, 0.6)):
            u = tt - (bt + off)
            m_ = u > 0
            hb[m_] += amp * np.sin(2 * np.pi * f * u[m_]) * np.exp(-u[m_] * 18)
        bt += 1 / 2.35
    hb *= np.clip((tt - (AGAP + 0.9 - t0)) / 0.6, 0, 1) * np.clip((PDUR + 0.1 - tt) / 0.6, 0, 1)
    # the kick
    for kt, amp in ((P0 + 2.2 - t0 - (AGAP - P0) + (AGAP - P0), 1.0), (P0 + 2.55 - t0, 0.7)):
        u = tt - kt
        m_ = u > 0
        hb[m_] += amp * 1.6 * np.sin(2 * np.pi * 40 * u[m_]) * np.exp(-u[m_] * 22)
    env_all = np.clip(tt / 0.45, 0, 1) * np.clip((t1 - t0 - tt) / 0.3, 0, 1)
    pk = max(np.abs(L_).max(), np.abs(R_).max()) + 1e-9
    bg_ = 0.0 if REAL_MUSIC else 0.26
    L_ = L_ / pk * bg_ * env_all + hb / (np.abs(hb).max() + 1e-9) * 0.22
    R_ = R_ / pk * bg_ * env_all + hb / (np.abs(hb).max() + 1e-9) * 0.22
    out[a:b, 0] += L_
    out[a:b, 1] += R_
    # open five seconds for the longer ending, where the deathbed scene used to cut to spring
    cut = int((D0 + 3.4) * sr)
    tail = out[cut:].copy()
    k_ = int(0.05 * sr)
    out[cut - k_:cut] *= np.linspace(1, 0, k_)[:, None]
    tail[:k_] *= np.linspace(0, 1, k_)[:, None]
    out = np.concatenate([out[:cut], np.zeros((int(D_ADD * sr), 2)), tail])
    n = len(out)
    if REAL_MUSIC:
        for name, ft, off, dur, gain, fin, fout in CUES:
            rec = cached('rec_' + name, lambda: load_rec(name))
            x = rec[int(off * sr):int((off + dur) * sr)].copy() * gain
            u = np.arange(len(x)) / sr
            x *= (np.clip(u / fin, 0, 1) * np.clip((dur - u) / fout, 0, 1))[:, None]
            i0 = int(ft * sr)
            e = min(n, i0 + len(x))
            out[i0:e] += x[:e - i0]
    # her heartbeat at the end: slowing, softening, and then gone
    bt, gap = 84.2, 0.95
    for amp, dub in ((0.5, True), (0.8, True), (1.0, True), (0.9, True), (0.75, True), (0.55, True), (0.35, False)):
        i0 = int(bt * sr)
        u = np.arange(int(0.7 * sr)) / sr
        beat = np.zeros_like(u)
        for off, f, a_ in (((0.0, 50, 1.0), (0.17, 64, 0.65)) if dub else ((0.0, 50, 1.0),)):
            v = u - off
            m_ = v > 0
            beat[m_] += a_ * (np.sin(2 * np.pi * f * v[m_]) + 0.5 * np.sin(2 * np.pi * 2 * f * v[m_])
                              + 0.2 * np.sin(2 * np.pi * 3 * f * v[m_])) * np.exp(-v[m_] * 15)
        e = min(n, i0 + len(u))
        out[i0:e] += (beat[:e - i0] * amp * 0.3)[:, None]
        bt += gap
        gap *= 1.1
    pk = np.abs(out).max()
    if pk > 0.97:
        out *= 0.97 / pk
    data = (out * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(data.tobytes())


def main():
    usage = ('usage:\n  python3 life.py video [out.mp4]          render the whole film\n'
             '  python3 life.py stills 12.5,40,71 [dir]  render single frames at these times (seconds)\n'
             '  python3 life.py audio [out.wav]           build only the soundtrack')
    if len(sys.argv) < 2 or sys.argv[1] not in ('video', 'stills', 'audio'):
        sys.exit(usage)
    os.makedirs(BUILD, exist_ok=True)
    mode = sys.argv[1]
    if mode == 'stills':
        if len(sys.argv) < 3:
            sys.exit(usage)
        outdir = sys.argv[3] if len(sys.argv) > 3 else BUILD
        os.makedirs(outdir, exist_ok=True)
        for tt in [float(x) for x in sys.argv[2].split(',')]:
            path = os.path.join(outdir, f'still_{tt:05.1f}.png')
            Image.frombytes('RGB', (OUT_W, OUT_H), render(int(tt * FPS))).save(path)
            print(path)
    elif mode == 'audio':
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BUILD, 'score.wav')
        make_audio(out)
        print(out)
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BUILD, 'a_life_master.mp4')
        wav = os.path.join(BUILD, 'score.wav')
        make_audio(wav)
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                              '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', '-', '-i', wav,
                              '-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-tune', 'animation', '-pix_fmt', 'yuv420p',
                              '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out],
                             stdin=subprocess.PIPE)
        with Pool(int(os.environ.get('LIFE_WORKERS', os.cpu_count() or 2))) as pool:
            for i, b in enumerate(pool.imap(render, range(NF), chunksize=4)):
                p.stdin.write(b)
                if i % 150 == 0:
                    print(f'frame {i} of {NF}', flush=True)
        p.stdin.close(); p.wait()
        print(out)


if __name__ == '__main__':
    main()

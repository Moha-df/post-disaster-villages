"""
Rendering of the damage, all procedural (no texture taken from another dataset):
roofs, partial / total collapse, burnt houses, toppled trees, debris trails, flood water,
burn scars and smoke. Every function edits `img` (BGR uint8) and `mask` (RescueNet ids) in place.

Sizes are given in METRES and converted with the ground sample distance (gsd, m/px), so the
same code works on 0.25 m aerial photos and on 0.5 m satellite images.
"""

import cv2
import numpy as np

from . import classes as C
from .hazard import directional_blur, fbm


def disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def feather(m, px):
    k = max(1, int(px)) * 2 + 1
    return cv2.GaussianBlur(m.astype(np.float32), (k, k), 0)


def over(dst, layer, alpha):
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    dst[:] = (dst.astype(np.float32) * (1 - a) + layer.astype(np.float32) * a).clip(0, 255).astype(np.uint8)


class Scene:
    """Per-image context: the ORIGINAL image/mask, scale and local colour statistics."""

    def __init__(self, img, mask, gsd, rng):
        self.img0, self.mask0, self.gsd, self.rng = img.copy(), mask.copy(), gsd, rng
        self.H, self.W = mask.shape
        ground = np.isin(mask, C.GROUND + [C.ROAD])
        self.lab0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        sel = ground if ground.sum() > 500 else np.ones_like(ground)
        self.ground_bgr = np.median(img[sel], 0).astype(np.float32)

    def m(self, metres):
        """metres -> pixels"""
        return metres / self.gsd

    def local_ground(self, x0, y0, x1, y1):
        """Median ground colour around a box (dust/rubble take the colour of the local soil)."""
        p = int(self.m(6))
        X0, Y0, X1, Y1 = max(0, x0 - p), max(0, y0 - p), min(self.W, x1 + p), min(self.H, y1 + p)
        sel = np.isin(self.mask0[Y0:Y1, X0:X1], C.GROUND + [C.ROAD])
        if sel.sum() < 30:
            return self.ground_bgr
        return np.median(self.img0[Y0:Y1, X0:X1][sel], 0).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# rubble made of the building's own material
# ---------------------------------------------------------------------------------------------
def shatter(src, fp, rng, frag_px, ground, dust=0.4, dark=1.0):
    """Voronoi fragments: each cell copies a random piece of the roof with a random shade,
    some cells become dust coloured like the local soil, cracks between cells are dark."""
    h, w = fp.shape
    # sample real roof material: skip tree crowns hanging over the footprint
    f = src.astype(np.float32)
    roof = fp & ((2 * f[..., 1] - f[..., 0] - f[..., 2]) < 12)
    ys, xs = np.nonzero(roof if roof.sum() >= 0.3 * fp.sum() and roof.sum() > 10 else fp)
    if len(ys) == 0:
        return src.copy()
    if roof.sum() < 0.3 * fp.sum():
        dust = max(dust, 0.7)  # mostly hidden roof: rubble is dominated by dust / soil
    n_seeds = int(np.clip(h * w / max(frag_px, 1.5) ** 2, 8, 30000))
    seeds = np.ones((h, w), np.uint8)
    sy, sx = rng.integers(0, h, n_seeds), rng.integers(0, w, n_seeds)
    seeds[sy, sx] = 0
    _, lbl = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    n = int(lbl.max()) + 1
    cy, cx = np.zeros(n, np.float32), np.zeros(n, np.float32)
    cy[lbl[sy, sx]], cx[lbl[sy, sx]] = sy, sx
    pick = rng.integers(0, len(ys), n)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # each fragment also gets a small random rotation: broken pieces don't stay aligned
    th = rng.uniform(-0.6, 0.6, n).astype(np.float32)[lbl]
    dy, dx = yy - cy[lbl], xx - cx[lbl]
    my = np.clip(ys[pick][lbl] + dy * np.cos(th) - dx * np.sin(th), 0, h - 1)
    mx = np.clip(xs[pick][lbl] + dy * np.sin(th) + dx * np.cos(th), 0, w - 1)
    out = cv2.remap(src, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)
    out *= (rng.uniform(0.55, 1.15, n).astype(np.float32) * dark)[lbl][..., None]
    is_dust = (rng.random(n) < dust)[lbl]
    dust_col = ground[None, None] * rng.uniform(0.95, 1.3, n).astype(np.float32)[lbl][..., None]
    out[is_dust] = dust_col[is_dust]
    crack = np.zeros((h, w), bool)
    crack[:, 1:] |= lbl[:, 1:] != lbl[:, :-1]
    crack[1:] |= lbl[1:] != lbl[:-1]
    out[crack & (rng.random((h, w)) < 0.7)] *= 0.55
    out += rng.normal(0, 7, out.shape)
    return out.clip(0, 255).astype(np.uint8)


def debris_trail(sc, img, mask, origin, angle, length_m, density, rng):
    """Debris thrown from `origin` (bool HxW) in a direction (wind / tornado rotation):
    elongated streaks, denser close to the source. Labels: debris on ground and roads."""
    L = sc.m(length_m)
    if L < 2 or not origin.any():
        return
    ys, xs = np.nonzero(origin)
    x0, x1 = max(0, int(xs.min() - L - 4)), min(sc.W, int(xs.max() + L + 4))
    y0, y1 = max(0, int(ys.min() - L - 4)), min(sc.H, int(ys.max() + L + 4))
    o = origin[y0:y1, x0:x1].astype(np.float32)
    h, w = o.shape
    # push the source along the direction: average of shifted copies = a trail
    trail = np.zeros_like(o)
    for t in np.linspace(0, 1, 8):
        M = np.float32([[1, 0, np.cos(angle) * L * t], [0, 1, np.sin(angle) * L * t]])
        trail = np.maximum(trail, cv2.warpAffine(o, M, (w, h)) * (1 - 0.7 * t))
    trail = cv2.GaussianBlur(trail, (0, 0), max(1.0, L / 10))
    grain = directional_blur(fbm(h, w, rng, scale=max(6, int(w * sc.gsd / 2)), octaves=2), angle, sc.m(2))
    pieces = (trail * (0.35 + grain) > 1.1 - 0.45 * density) & ~origin[y0:y1, x0:x1]
    mc = mask[y0:y1, x0:x1]
    pieces &= np.isin(mc, C.GROUND + [C.ROAD])  # debris under tree crowns would not be visible
    if not pieces.any():
        return
    # the debris is made of the building's material (only its footprint is sampled)
    src_fp = origin[y0:y1, x0:x1] & np.isin(sc.mask0[y0:y1, x0:x1], C.BUILDINGS)
    if not src_fp.any():
        src_fp = origin[y0:y1, x0:x1]
    rub = shatter(sc.img0[y0:y1, x0:x1], src_fp, rng, sc.m(0.6), sc.local_ground(x0, y0, x1, y1), dust=0.5)
    over(img[y0:y1, x0:x1], rub, feather(pieces, 1) * 0.9)
    mc[pieces] = C.DEBRIS


# ---------------------------------------------------------------------------------------------
# buildings
# ---------------------------------------------------------------------------------------------
def _box(b, pad, sc):
    x, y, w, h = b["bbox"]
    return max(0, x - pad), max(0, y - pad), min(sc.W, x + w + pad), min(sc.H, y + h + pad)


def roof_damage(sc, crop, fp, rng, amount, wind=None):
    """Missing shingles, exposed decking, blue tarps and ridge damage."""
    ys, xs = np.nonzero(fp)
    if len(ys) == 0:
        return
    h, w = fp.shape
    layer, alpha = crop.copy(), np.zeros((h, w), np.float32)
    roof = np.median(crop[fp], 0).astype(np.float32)
    size = np.sqrt(len(ys))
    for _ in range(int(2 + amount * 10)):
        k = int(rng.integers(len(ys)))
        cx, cy = int(xs[k]), int(ys[k])
        r = float(np.clip(size * rng.uniform(0.05, 0.14), sc.m(0.4), sc.m(4)))
        pm = np.zeros((h, w), np.uint8)
        kind = rng.choice(["missing", "decking", "tarp"], p=[0.45, 0.3, 0.25])
        if kind == "tarp":
            rect = cv2.boxPoints(((cx, cy), (r * 2.4, r * 1.6), float(rng.uniform(0, 180))))
            cv2.fillPoly(pm, [rect.astype(np.int32)], 1)
            col = np.array([rng.integers(150, 215), rng.integers(95, 140), rng.integers(15, 55)], np.float32)
        else:
            ang = wind if wind is not None else rng.uniform(0, 2 * np.pi)
            # wind peels shingles in strips along its direction
            ax = (int(r * rng.uniform(1.0, 2.0)), max(1, int(r * rng.uniform(0.4, 0.8))))
            cv2.ellipse(pm, (cx, cy), ax, np.degrees(ang) + rng.normal(0, 15), 0, 360, 1, -1)
            col = roof * 0.42 if kind == "missing" else 0.4 * roof + 0.6 * np.array([105, 145, 180], np.float32)
        sel = (pm > 0) & fp
        layer[sel] = (col + rng.normal(0, 10, (int(sel.sum()), 3))).clip(0, 255)
        alpha[sel] = rng.uniform(0.75, 0.95)
    over(crop, layer, cv2.GaussianBlur(alpha, (3, 3), 0) * fp)


def remove_shadow(sc, crop, mcrop, fp):
    """A collapsed building no longer casts a shadow: brighten the dark area touching it."""
    ring = (cv2.dilate(fp.astype(np.uint8), disk(sc.m(5))) > 0) & ~fp & np.isin(mcrop, C.GROUND + [C.ROAD])
    if ring.sum() < 20:
        return
    L = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    med = float(np.median(L[ring]))
    dark = (ring & (L < 0.65 * med)).astype(np.uint8)
    n, lab = cv2.connectedComponents(dark)
    touch = set(np.unique(lab[(cv2.dilate(fp.astype(np.uint8), disk(2)) > 0) & (dark > 0)])) - {0}
    keep = np.isin(lab, list(touch))
    if keep.any():
        gain = np.clip(med / np.maximum(L, 1), 1, 3)[..., None]
        over(crop, (crop * gain).clip(0, 255), feather(keep, 2) * 0.9)


def collapse(sc, img, mask, b, b_lab, level, rng, wind=None, burnt=False):
    """level 2: partial collapse (a part of the roof falls in), level 3: total collapse."""
    size = np.sqrt(b["area"])
    pad = int(size * 0.6 + sc.m(3))
    x0, y0, x1, y1 = _box(b, pad, sc)
    crop, mcrop = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    fp = b_lab[y0:y1, x0:x1] == b["label"]
    src = crop.copy()
    h, w = fp.shape
    ground = sc.local_ground(x0, y0, x1, y1)
    ys, xs = np.nonzero(fp)

    if level == 2:
        # cut the footprint with a random line: one side caves in
        ang = rng.uniform(0, np.pi)
        cx, cy = xs.mean() + rng.normal(0, size * 0.15), ys.mean() + rng.normal(0, size * 0.15)
        yy, xx = np.mgrid[0:h, 0:w]
        side = ((xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)) > 0
        side = side if rng.random() < 0.5 else ~side
        jag = fbm(h, w, rng, 10, 2) > 0.3
        caved = fp & side & (jag | (cv2.erode(fp.astype(np.uint8), disk(2)) > 0))
        roof_damage(sc, crop, fp & ~caved, rng, 0.7, wind)
        rub = shatter(src, fp, rng, sc.m(0.7), ground, dust=0.25, dark=0.75)
        over(crop, rub, feather(caved, 1) * 0.95)
        region = fp
    else:
        remove_shadow(sc, crop, mcrop, fp)
        noise = fbm(h, w, rng, 8, 3)
        spill = cv2.dilate(fp.astype(np.uint8), disk(max(1.5, size * 0.12))) > 0
        region = (fp & (noise > 0.1)) | (spill & (noise > 0.5) & np.isin(mcrop, C.GROUND + [C.ROAD]))
        rub = shatter(src, fp, rng, sc.m(0.6), ground, dust=0.45)
        over(crop, rub, feather(region, 2) * 0.97)
        # remains of walls: broken pieces of the outline stay standing (bright lines)
        cnts, _ = cv2.findContours(fp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        walls = np.zeros((h, w), np.uint8)
        cv2.drawContours(walls, cnts, -1, 1, max(1, int(sc.m(0.3))))
        walls = (walls > 0) & (fbm(h, w, rng, 14, 2) > 0.55)
        over(crop, np.full_like(crop, 1) * (ground * 1.25 + 10).clip(0, 255), feather(walls, 0) * 0.6)

    if burnt:
        char = np.array([28, 30, 33], np.float32)
        ash = np.array([165, 170, 172], np.float32)
        n = fbm(h, w, rng, 12, 3)[..., None]
        burnt_layer = char * (1 - n * 0.6) + ash * n * 0.6 + rng.normal(0, 6, (h, w, 1))
        over(crop, burnt_layer.clip(0, 255), feather(region, 1) * (0.85 if level == 3 else 0.6))

    full = np.zeros(mask.shape, bool)
    full[y0:y1, x0:x1] = region
    if not burnt:
        debris_trail(sc, img, mask, full, wind if wind is not None else rng.uniform(0, 2 * np.pi),
                     length_m=2 + 4 * level + (6 if wind is not None else 0), density=0.5 + 0.2 * level, rng=rng)
    mcrop[region | fp] = C.BLD_BY_LEVEL[level]


def scorch(sc, img, mask, b, b_lab, rng):
    """Minor fire damage: blackened roof edge on the side facing the fire."""
    x0, y0, x1, y1 = _box(b, 0, sc)
    crop = img[y0:y1, x0:x1]
    fp = b_lab[y0:y1, x0:x1] == b["label"]
    d = cv2.distanceTransform(fp.astype(np.uint8), cv2.DIST_L2, 3)
    edge = fp & (d < max(2, np.sqrt(b["area"]) * 0.25)) & (fbm(*fp.shape, rng, 6, 2) > 0.45)
    over(crop, (crop * 0.35).astype(np.uint8), feather(edge, 1) * 0.8)
    mask[y0:y1, x0:x1][fp] = C.BLD_MINOR


def damage_building(sc, img, mask, b, b_lab, level, scenario, rng, wind=None):
    if level == 0:
        return
    if scenario == "wildfire":
        if level == 1:
            scorch(sc, img, mask, b, b_lab, rng)
        else:
            collapse(sc, img, mask, b, b_lab, level, rng, burnt=True)
        return
    if level == 1:
        x0, y0, x1, y1 = _box(b, 0, sc)
        fp = b_lab[y0:y1, x0:x1] == b["label"]
        roof_damage(sc, img[y0:y1, x0:x1], fp, rng, 0.45, wind)
        mask[y0:y1, x0:x1][fp] = C.BLD_MINOR
    else:
        collapse(sc, img, mask, b, b_lab, level, rng, wind=wind)


# ---------------------------------------------------------------------------------------------
# trees and vegetation
# ---------------------------------------------------------------------------------------------
def topple_trees(sc, img, mask, field, wind, rng, strength):
    """Single trees in strong wind fall DOWNWIND: the canopy moves by about its height and
    the ground where it stood is revealed (inpainted from the surroundings)."""
    n, lab, st, cen = cv2.connectedComponentsWithStats((sc.mask0 == C.TREE).astype(np.uint8))
    fallen = []
    max_area = sc.m(12) ** 2
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < sc.m(2) ** 2 or area > max_area:
            continue
        cx, cy = cen[i]
        p = field[int(cy), int(cx)] * strength
        if rng.random() > p * 0.9:
            continue
        pad = int(max(w, h) * 1.6)
        X0, Y0, X1, Y1 = max(0, x - pad), max(0, y - pad), min(sc.W, x + w + pad), min(sc.H, y + h + pad)
        canopy = (lab[Y0:Y1, X0:X1] == i).astype(np.uint8)
        patch = sc.img0[Y0:Y1, X0:X1].copy()
        a = np.clip(wind + rng.normal(0, 0.25), -10, 10)
        shift = max(w, h) * rng.uniform(0.7, 1.2)
        # a fallen crown looks elongated and seen from the side: stretch along the fall direction
        M = cv2.getRotationMatrix2D((float(cx - X0), float(cy - Y0)), 0, 1.0)
        M[:, 2] += [np.cos(a) * shift, np.sin(a) * shift]
        S = np.float32([[1 + 0.35 * np.cos(a) ** 2, 0.35 * np.cos(a) * np.sin(a), 0],
                        [0.35 * np.cos(a) * np.sin(a), 1 + 0.35 * np.sin(a) ** 2, 0]])
        M2 = np.vstack([M, [0, 0, 1]]) @ np.vstack([S, [0, 0, 1]])
        hh, ww = canopy.shape
        moved = cv2.warpAffine(patch, M2[:2], (ww, hh), flags=cv2.INTER_LINEAR)
        moved_a = cv2.warpAffine(canopy.astype(np.float32), M2[:2], (ww, hh))
        region = img[Y0:Y1, X0:X1]
        mreg = mask[Y0:Y1, X0:X1]
        # 1) reveal the ground under the old crown
        hole = cv2.dilate(canopy, disk(1))
        region[:] = cv2.inpaint(region, hole, max(3, int(sc.m(1.5))), cv2.INPAINT_TELEA)
        mreg[(canopy > 0) & (mreg == C.TREE)] = C.BACKGROUND
        # 2) trunk + fallen crown + its shadow, but not on top of buildings
        free = ~np.isin(mreg, C.BUILDINGS)
        trunk = np.zeros((hh, ww), np.uint8)
        cv2.line(trunk, (int(cx - X0), int(cy - Y0)),
                 (int(cx - X0 + np.cos(a) * shift), int(cy - Y0 + np.sin(a) * shift)), 1, max(1, int(sc.m(0.35))))
        sh = np.roll(np.roll(moved_a, 2, 0), 2, 1) * (moved_a < 0.3) * free
        over(region, region * 0.5, sh * 0.5)
        over(region, np.full_like(region, 1) * np.float32([55, 75, 95]), (trunk > 0) * free * 0.9)
        over(region, moved, cv2.GaussianBlur(moved_a, (3, 3), 0) * free)
        mreg[(moved_a > 0.5) & free] = C.TREE
        fallen.append([int(cx), int(cy), round(float(np.degrees(a)) % 360, 1)])
    return fallen


def vegetation_stress(sc, img, mask, field, amount, rng, kind="wind"):
    """Wind strips leaves / flattens crops (brown-grey); flood leaves mud on fields."""
    f = sc.img0.astype(np.float32)
    green = ((2 * f[..., 1] - f[..., 0] - f[..., 2]) > 8) | (sc.mask0 == C.TREE)
    green &= ~np.isin(mask, C.BUILDINGS + [C.WATER, C.DEBRIS])
    s = np.clip(field * amount * (0.6 + 0.8 * fbm(sc.H, sc.W, rng, 5, 3)) - 0.15, 0, 1) * green
    s = cv2.GaussianBlur(s.astype(np.float32), (0, 0), 1.2)
    L = f.mean(2, keepdims=True)
    if kind == "wind":
        target = np.concatenate([L * 0.75 + 30, L * 0.85 + 38, L * 0.95 + 48], 2)
    else:
        target = np.concatenate([L * 0.4 + 55, L * 0.4 + 80, L * 0.4 + 98], 2)
    target += (fbm(sc.H, sc.W, rng, 60, 2)[..., None] - 0.5) * 25
    over(img, target.clip(0, 255), s * 0.85)


def burn_scar(sc, img, mask, field, rng):
    """Burnt ground: black core, grey/white ash, scorched brown margin. Burnt trees lose
    their crown (label -> background) where the fire was strong."""
    burnable = ~np.isin(mask, C.BUILDINGS + [C.WATER, C.ROAD])
    core = np.clip((field - 0.35) / 0.3, 0, 1) * burnable
    margin = np.clip((field - 0.15) / 0.25, 0, 1) * burnable
    f = sc.img0.astype(np.float32)
    L = f.mean(2, keepdims=True)
    # burnt land keeps the STRUCTURE of what was there (tree skeletons, field patterns), but
    # charcoal-dark; where grass burnt completely, white/grey ash patches
    tree0 = (sc.mask0 == C.TREE)[..., None]
    charcoal = np.where(tree0, L * 0.30 + 8, L * 0.38 + 14) * np.float32([1.0, 1.02, 1.06])
    ash_n = fbm(sc.H, sc.W, rng, 45, 3)
    ash = ((ash_n > 0.72) & (sc.mask0 != C.TREE))[..., None]
    texture = np.where(ash, L * 0.3 + np.float32([80, 84, 87]), charcoal)
    texture += (fbm(sc.H, sc.W, rng, 90, 2)[..., None] - 0.5) * 18
    scorched = f * np.float32([0.5, 0.6, 0.75])  # brown, dried margin
    over(img, scorched, feather(margin > 0, 2) * margin * 0.8)
    over(img, texture.clip(0, 255), cv2.GaussianBlur(core, (0, 0), 1.0) * 0.95)
    dead = (core > 0.6) & (mask == C.TREE)
    mask[dead] = C.BACKGROUND
    return float((core > 0.5).mean())


def smoke(sc, img, field, wind, rng, sev):
    """Smoke plumes rising from the active fire edge, drifting downwind."""
    edge = np.clip(field * (1 - field) * 4, 0, 1)
    plume = directional_blur(edge, wind, max(sc.H, sc.W) * 0.35)
    M = np.float32([[1, 0, np.cos(wind) * sc.W * 0.12], [0, 1, np.sin(wind) * sc.H * 0.12]])
    plume = cv2.warpAffine(plume, M, (sc.W, sc.H), borderMode=cv2.BORDER_REFLECT)
    plume = np.clip(plume / max(plume.max(), 1e-6) * fbm(sc.H, sc.W, rng, 4, 4) * 1.3, 0, 1)
    plume = np.clip(plume - 0.25, 0, 1) / 0.75  # thin, separate wisps rather than a blanket
    col = np.float32([185, 190, 196]) + (fbm(sc.H, sc.W, rng, 16, 2)[..., None] - 0.5) * 30
    over(img, np.broadcast_to(col, img.shape), plume * (0.12 + 0.2 * sev))


def haze(sc, img, rng, amount, color=(178, 182, 186)):
    """Patchy dust (earthquake) or storm haze: light, keeps the ground readable."""
    a = np.clip(fbm(sc.H, sc.W, rng, 3, 3) - 0.35, 0, 1) / 0.65 * amount
    over(img, np.broadcast_to(np.float32(color), img.shape), a)


# ---------------------------------------------------------------------------------------------
# flood water
# ---------------------------------------------------------------------------------------------
def flood_water(sc, img, mask, depth, rng, flow=None):
    """Depth-dependent muddy water: translucent in the shallows (ground still visible),
    opaque where deep; sediment streaks along the flow; floating debris line (wrack) at the
    shoreline; buildings stay above water, tree crowns stay visible."""
    wet = (depth > 0.02) & ~np.isin(mask, C.BUILDINGS)
    wet = cv2.morphologyEx(wet.astype(np.uint8), cv2.MORPH_OPEN, disk(sc.m(0.8))).astype(bool)
    flow = rng.uniform(0, 2 * np.pi) if flow is None else flow
    # sediment-laden water (BGR): grey-brown, darker and smoother than any field
    base = np.float32([rng.integers(55, 68), rng.integers(80, 92), rng.integers(104, 118)])
    streak = directional_blur(fbm(sc.H, sc.W, rng, 30, 3), flow, sc.m(25))
    col = base * (1.08 - 0.25 * depth[..., None]) + (streak[..., None] - 0.5) * 16
    sheen = np.clip(directional_blur(fbm(sc.H, sc.W, rng, 6, 2), flow, sc.m(40)) - 0.6, 0, 1) * 35
    col += sheen[..., None]  # sky reflection on calm water
    alpha = np.clip(0.9 + 0.4 * depth, 0, 0.97) * feather(wet, 2)
    tree = sc.mask0 == C.TREE
    alpha[tree] *= np.clip(1.1 - 2.2 * depth[tree], 0.15, 1)  # crowns above shallow water
    already = sc.mask0 == C.WATER
    alpha[already] *= 0.5  # rivers keep part of their look, now muddier
    over(img, col.clip(0, 255), alpha)
    # wrack line: thin band of floating debris / foam along the edge
    d_out = cv2.distanceTransform((~wet).astype(np.uint8), cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(wet.astype(np.uint8), cv2.DIST_L2, 3)
    band = (d_in > 0) & (d_in < sc.m(1.2)) & (fbm(sc.H, sc.W, rng, 50, 2) > 0.5)
    over(img, np.broadcast_to(np.float32([150, 165, 175]), img.shape), feather(band, 0) * 0.45)
    # wet, darker ground just outside the water
    damp = np.clip(1 - d_out / sc.m(4), 0, 1) * (~wet) * ~np.isin(mask, C.BUILDINGS)
    img[:] = (img.astype(np.float32) * (1 - 0.3 * damp[..., None])).astype(np.uint8)
    lab = wet & ~(tree & (depth < 0.45))
    mask[lab] = C.WATER
    return lab


# ---------------------------------------------------------------------------------------------
# match the look of the original sensor
# ---------------------------------------------------------------------------------------------
def match_sensor(before, after):
    """Painted areas are too clean: give them the blur and noise level of the real photo."""
    changed = cv2.absdiff(before, after).max(2) > 4
    if changed.mean() < 1e-4:
        return after
    g = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    sigma = 1.4826 * float(np.median(np.abs(hp[~changed]))) + 0.5  # robust noise level
    soft = cv2.GaussianBlur(after, (0, 0), 0.6).astype(np.float32)
    noise = np.random.default_rng(0).normal(0, sigma, after.shape[:2])[..., None]
    a = feather(changed, 2)[..., None]
    return (after * (1 - a) + (soft + noise) * a).clip(0, 255).astype(np.uint8)

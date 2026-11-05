"""
High-resolution renderer ("drone mode", a few cm per pixel).

At 0.3 m/px a collapsed house is a few hundred pixels and a 2D texture is enough. At 2 cm/px the eye
expects relief: pieces of roof, concrete, bricks and beams lying on top of each other, lit by the sun
and casting shadows. This module therefore renders the damage in 2.5D:

  * every piece of debris has an albedo (colour), a coverage and a HEIGHT (metres);
  * the height map is lit with the sun direction ESTIMATED FROM THE PHOTO (direction of the existing
    building shadows) and casts shadows on what lies around it;
  * crevices get ambient occlusion, fine grain is drawn at its real size (cm).

Same interface as effects.py: functions edit `img` (BGR uint8) and `mask` (RescueNet ids) in place.
"""

import cv2
import numpy as np

from . import classes as C
from .effects import disk, feather, over, remove_shadow
from .hazard import directional_blur, fbm

GROUND_ROAD = C.GROUND + [C.ROAD]


# ---------------------------------------------------------------------------------------------
# sun and lighting
# ---------------------------------------------------------------------------------------------
def estimate_sun(img, mask, gsd):
    """Sun direction from the photo: the side of the buildings where the ground is darkest is the
    shadow side. Returns (unit vector TOWARD the sun in image coords, elevation, shadow ratio)."""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    bld = np.isin(mask, C.BUILDINGS).astype(np.uint8)
    ground = np.isin(mask, GROUND_ROAD)
    k = max(2.0, 1.2 / gsd)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    means = []
    for a in angles:
        M = np.float32([[1, 0, np.cos(a) * k], [0, 1, np.sin(a) * k]])
        new = (cv2.warpAffine(bld, M, bld.shape[::-1]) > 0) & (bld == 0) & ground
        means.append(float(L[new].mean()) if new.sum() > 200 else np.nan)
    means = np.array(means)
    lit = float(np.median(L[ground])) if ground.sum() > 500 else 150.0
    if np.all(np.isnan(means)) or lit - np.nanmin(means) < 12:
        shadow_dir = -np.pi / 2 - 0.4  # no clear shadow: classic north-up midday look
        ratio = 0.5
    else:
        sm = np.where(np.isnan(means), np.nanmax(means), means)
        sm = np.convolve(np.r_[sm[-1], sm, sm[0]], [0.25, 0.5, 0.25], "valid")  # smooth over angle
        shadow_dir = float(angles[int(np.argmin(sm))])
        ratio = float(np.clip(np.nanmin(means) / max(lit, 1), 0.3, 0.75))
    sun = np.array([np.cos(shadow_dir + np.pi), np.sin(shadow_dir + np.pi)], np.float32)
    return sun, np.radians(42.0), ratio


class Light:
    def __init__(self, sc):
        self.sun, self.elev, self.amb = estimate_sun(sc.img0, sc.mask0, sc.gsd)
        self.gsd = sc.gsd

    def lambert(self, z):
        """Lighting of a height map z (metres), relative to flat ground (flat = 1)."""
        gy, gx = np.gradient(z.astype(np.float32))
        gx, gy = gx / self.gsd, gy / self.gsd
        lx, ly = self.sun * np.cos(self.elev)
        lz = np.sin(self.elev)
        dot = (-gx * lx - gy * ly + lz) / np.sqrt(gx * gx + gy * gy + 1)
        return np.clip(dot / lz, 0.0, 1.3)

    def shadow(self, z, max_m=None):
        """Cast shadow (0..1) of the height map z on itself (ray marching toward the sun)."""
        zmax = float(z.max())
        if zmax <= 0.005:
            return np.zeros(z.shape, np.float32)
        tan_e = np.tan(self.elev)
        reach = (max_m or zmax / tan_e) / self.gsd
        step = max(1.0, reach / 60)
        sh = np.zeros(z.shape, bool)
        h, w = z.shape
        d = step
        while d <= reach + step:
            M = np.float32([[1, 0, self.sun[0] * d], [0, 1, self.sun[1] * d]])
            ahead = cv2.warpAffine(z, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            sh |= ahead > z + d * self.gsd * tan_e + 0.004
            d += step
        return cv2.GaussianBlur(sh.astype(np.float32), (0, 0), 0.8)

    def factor(self, z, ao=True):
        """Multiplicative light factor for a height map: sun + cast shadow + ambient occlusion."""
        f = self.amb + (1 - self.amb) * self.lambert(z) * (1 - self.shadow(z))
        if ao:
            s = max(1.0, 0.25 / self.gsd)
            occl = np.clip((cv2.GaussianBlur(z, (0, 0), s) - z) / 0.25, 0, 1)
            f *= 1 - 0.45 * occl
        return f


def get_light(sc):
    if getattr(sc, "light", None) is None:
        sc.light = Light(sc)
    return sc.light


def compose(sc, crop, mcrop, col, cov, z):
    """Puts a layer (albedo `col`, coverage `cov`, height `z` in metres) on `crop`, lit by the sun;
    the layer also casts its shadow on the ground around it."""
    lt = get_light(sc)
    zt = (z * (cov > 0.5)).astype(np.float32)
    f_layer = lt.factor(zt)
    sh = lt.shadow(zt)
    # shadow on the existing image (not where it is already in shadow)
    L = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    lit = L > 0.75 * np.median(L)
    darken = 1 - (1 - lt.amb) * sh * lit * (1 - cov)
    base = crop.astype(np.float32) * darken[..., None]
    top = col.astype(np.float32) * f_layer[..., None]
    a = np.clip(cov, 0, 1)[..., None]
    crop[:] = (base * (1 - a) + top * a).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------------------------
def grain(h, w, rng, px, amp):
    """Fine grain of size `px` pixels (gravel, dust, sand)."""
    gh, gw = max(2, int(h / max(px, 1))), max(2, int(w / max(px, 1)))
    g = cv2.resize(rng.normal(0, 1, (gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return g * amp


class Palette:
    """Materials of a building: its own roof (texture is copied), concrete, plaster, brick, wood."""

    def __init__(self, sc, src, fp, rng, burnt=False):
        f = src.astype(np.float32)
        roof = fp & ((2 * f[..., 1] - f[..., 0] - f[..., 2]) < 12)
        roof = roof if roof.sum() > 0.3 * fp.sum() else fp
        self.src, self.roof_px = src, np.column_stack(np.nonzero(roof))
        self.roof_col = np.median(src[roof], 0).astype(np.float32) if roof.any() else np.float32([120] * 3)
        self.burnt = burnt
        warm = rng.uniform(-8, 8)
        self.cols = {
            "concrete": np.float32([150 - warm, 152, 150 + warm]),
            "plaster": np.float32([196, 200, 204]),
            "brick": np.float32([62, 88, 152]),
            "wood": np.float32([96, 116, 138]),
            "dark": np.float32([48, 50, 54]),
        }
        if burnt:
            self.weights = {"char": 0.45, "ash": 0.25, "concrete": 0.15, "metal": 0.15}
            self.cols.update(char=np.float32([30, 31, 34]), ash=np.float32([168, 172, 174]),
                             metal=np.float32([60, 85, 125]))
        else:
            self.weights = {"roof": 0.46, "concrete": 0.24, "plaster": 0.16, "brick": 0.1, "wood": 0.04}

    def pick(self, rng):
        k = list(self.weights)
        return k[int(rng.choice(len(k), p=np.array(list(self.weights.values())) / sum(self.weights.values())))]

    def texture(self, kind, h, w, rng, angle, gsd):
        """Albedo patch (h, w, 3) of a material; roof pieces are rotated copies of the real roof."""
        if kind == "roof" and len(self.roof_px):
            cy, cx = self.roof_px[int(rng.integers(len(self.roof_px)))]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), float(np.degrees(angle) + rng.normal(0, 25)), 1.0)
            M[:, 2] += [cx - w / 2, cy - h / 2]
            patch = cv2.warpAffine(self.src, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT).astype(np.float32)
            return patch * rng.uniform(0.85, 1.1)
        col = self.cols.get(kind, self.cols["concrete"]) * rng.uniform(0.82, 1.15)
        out = np.broadcast_to(col, (h, w, 3)).astype(np.float32).copy()
        if kind == "wood":  # grain along the plank
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            t = -xx * np.sin(angle) + yy * np.cos(angle)
            out *= (1 + 0.08 * np.sin(t / max(1, 0.012 / gsd)) + grain(h, w, rng, 2, 0.04))[..., None]
        else:
            out *= (1 + grain(h, w, rng, max(1, 0.03 / gsd), 0.06))[..., None]
        return out


# ---------------------------------------------------------------------------------------------
# rubble pile
# ---------------------------------------------------------------------------------------------
def _poly(rng, cx, cy, rx, ry, ang, n=None):
    n = n or int(rng.integers(4, 8))
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(0.65, 1.0, n)
    x, y = np.cos(t) * rx * r, np.sin(t) * ry * r
    c, s = np.cos(ang), np.sin(ang)
    return np.column_stack([cx + x * c - y * s, cy + x * s + y * c]).astype(np.int32)


def draw_pieces(sc, pal, rng, layer, where, n, size_m, height_m, kinds=None, base_z=None, planks=0.0, angle=None):
    """Draws `n` pieces (convex polygons or planks) inside `where` (bool mask) on the layer
    (col, cov, z). Each piece is a tilted plate lying on what is below it."""
    col, cov, z = layer
    h, w = where.shape
    ys, xs = np.nonzero(where)
    if len(ys) == 0 or n <= 0:
        return
    g = sc.gsd
    for _ in range(int(n)):
        k = int(rng.integers(len(ys)))
        cy, cx = float(ys[k]), float(xs[k])
        a = float(rng.uniform(0, np.pi)) if angle is None else float(angle + rng.normal(0, 0.5))
        if rng.random() < planks:
            kind = "wood"
            L, Wd = rng.uniform(0.6, 2.0) / g, rng.uniform(0.06, 0.14) / g
            pts = cv2.boxPoints(((cx, cy), (L, Wd), np.degrees(a))).astype(np.int32)
            thick = rng.uniform(0.05, 0.12)
        else:
            kind = (kinds[int(rng.integers(len(kinds)))] if kinds else pal.pick(rng))
            s = float(np.exp(rng.uniform(np.log(size_m[0]), np.log(size_m[1])))) / g
            pts = _poly(rng, cx, cy, s, s * rng.uniform(0.5, 1.0), a)
            thick = rng.uniform(0.04, 0.12) + (0.1 if kind in ("concrete", "brick") else 0)
        x0, y0 = np.maximum(pts.min(0) - 2, 0)
        x1, y1 = np.minimum(pts.max(0) + 3, [w, h])
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        pm = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(pm, [pts - [x0, y0]], 1)
        sel = pm > 0
        if not sel.any():
            continue
        hh, ww = pm.shape
        tex = pal.texture(kind, hh, ww, rng, a, g)
        # tilted plate on top of what is already there
        below = z[y0:y1, x0:x1][sel]
        zb = float(np.percentile(below, 70)) if base_z is None else base_z
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        sx, sy = rng.normal(0, 0.25, 2) * g  # slope in m per pixel
        plate = zb + thick + (xx - ww / 2) * sx + (yy - hh / 2) * sy
        plate = np.clip(plate, 0, None)
        # thin dark rim: broken edges
        rim = sel & ~(cv2.erode(pm, disk(1)) > 0)
        tex[rim] *= 0.85
        col[y0:y1, x0:x1][sel] = tex[sel]
        z[y0:y1, x0:x1][sel] = plate[sel]
        cov[y0:y1, x0:x1][sel] = 1.0


def rubble_layer(sc, src, region, pal, rng, pile_m, dust_col, dust=0.3, voids=0.0, planks=0.03):
    """Albedo / coverage / height of a rubble pile covering `region`."""
    h, w = region.shape
    g = sc.gsd
    d = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5) * g
    dm = max(float(d.max()), 1e-3)
    mound = pile_m * np.clip(d / min(dm, 3.0), 0, 1) ** 0.7
    mound *= 0.65 + 0.7 * fbm(h, w, rng, max(3, int(w * g / 2)), 3)
    z = (mound * region).astype(np.float32)
    # base: gravel / dust / small debris, coloured by a mix of the materials
    mix = fbm(h, w, rng, max(4, int(w * g / 0.6)), 2)
    base = np.where(mix[..., None] > 0.55, pal.cols["concrete"], dust_col)
    base = base * 0.6 + pal.roof_col * 0.4 * (mix[..., None] > 0.75) + base * 0.4 * (mix[..., None] <= 0.75)
    base = base * (1 + grain(h, w, rng, max(1, 0.025 / g), 0.12) + grain(h, w, rng, max(1, 0.08 / g), 0.08))[..., None]
    if voids > 0:  # dark holes into the building (partial collapse)
        holes = (fbm(h, w, rng, max(3, int(w * g / 1.2)), 2) > 1 - voids) & region
        base[holes] = pal.cols["dark"] * 0.7
        z[holes] *= 0.3
    col = base.astype(np.float32)
    cov = region.astype(np.float32)
    area_m2 = region.sum() * g * g
    layer = (col, cov, z)
    # big slabs, then medium chunks, then small pieces and beams on top
    draw_pieces(sc, pal, rng, layer, region, area_m2 * 0.5, (0.5, 1.5), pile_m, kinds=["roof"] * 3 + ["concrete"])
    draw_pieces(sc, pal, rng, layer, region, area_m2 * 1.2, (0.15, 0.45), pile_m, planks=planks)
    draw_pieces(sc, pal, rng, layer, region, area_m2 * 3.0, (0.04, 0.12), pile_m)
    # everything is covered by a little dust of the local soil
    dmask = (fbm(h, w, rng, max(3, int(w * g / 1.5)), 3) * dust)[..., None]
    col[:] = col * (1 - dmask) + dust_col * dmask
    # soft, irregular edge of the pile
    cov[:] = np.clip(feather(cov > 0.5, max(1, int(0.04 / g))) * 1.0, 0, 1) * (cov > 0)
    return col, cov, z


def roof_sections(sc, src, fp, layer, rng, frac, inward=0.5):
    """Large pieces of the REAL roof (2-4 m), broken along jagged lines, slid towards the centre,
    slightly rotated and tilted, lying on the rubble: they keep the structure of the house readable."""
    col, cov, z = layer
    g = sc.gsd
    h, w = fp.shape
    ys, xs = np.nonzero(fp)
    if len(ys) < 50:
        return
    n = max(2, int(len(ys) * g * g / rng.uniform(4.0, 8.0)))
    seeds = np.ones((h, w), np.uint8)
    k = rng.integers(0, len(ys), n)
    seeds[ys[k], xs[k]] = 0
    _, lbl = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    # jagged break lines: the cell borders are perturbed by noise
    jx = (fbm(h, w, rng, max(6, int(w * g / 0.4)), 2) - 0.5) * 0.5 / g
    jy = (fbm(h, w, rng, max(6, int(w * g / 0.4)), 2) - 0.5) * 0.5 / g
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lbl = cv2.remap(lbl.astype(np.float32), xx + jx, yy + jy, cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REPLICATE).astype(np.int32)
    ccx, ccy = float(xs.mean()), float(ys.mean())
    gap = disk(max(1, 0.06 / g))
    for i in np.unique(lbl[fp]):
        if rng.random() > frac:
            continue
        cell = ((lbl == i) & fp).astype(np.uint8)
        cell = cv2.erode(cell, gap)
        if cell.sum() < 30:
            continue
        cy, cx = [float(v.mean()) for v in np.nonzero(cell)]
        # slide towards the centre of the house and rotate a little
        sh = rng.uniform(0.1, inward) / g
        dx, dy = ccx - cx, ccy - cy
        dn = max(np.hypot(dx, dy), 1e-3)
        M = cv2.getRotationMatrix2D((cx, cy), float(rng.normal(0, 6)), 1.0)
        M[:, 2] += [dx / dn * sh, dy / dn * sh]
        piece = cv2.warpAffine(cell, M, (w, h), flags=cv2.INTER_NEAREST) > 0
        tex = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        if not piece.any():
            continue
        # tilted plate resting on the pile
        a = rng.uniform(0, 2 * np.pi)
        slope = rng.uniform(0.15, 0.55)
        plane = ((xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)) * g * slope
        base = float(np.percentile(z[piece], 60)) + 0.1
        zp = np.clip(base + plane, 0.05, None)
        rim = piece & ~(cv2.erode(piece.astype(np.uint8), disk(max(1, 0.05 / g))) > 0)
        tex[rim] *= 0.6  # broken edge, dark
        col[piece] = tex[piece] * rng.uniform(0.9, 1.05)
        z[piece] = zp[piece]
        cov[piece] = 1.0


# ---------------------------------------------------------------------------------------------
# roof details
# ---------------------------------------------------------------------------------------------
def roof_axis(crop, fp, cx, cy, r):
    """Direction of the tile rows near (cx, cy): perpendicular to the dominant gradient."""
    y0, y1 = max(0, int(cy - r)), int(cy + r)
    x0, x1 = max(0, int(cx - r)), int(cx + r)
    g = cv2.GaussianBlur(cv2.cvtColor(crop[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    m = fp[y0:y1, x0:x1]
    if m.sum() < 20:
        return 0.0
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (gx * gx)[m].sum(), (gy * gy)[m].sum(), (gx * gy)[m].sum()
    grad = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    return float(grad + np.pi / 2)


def missing_tiles(sc, crop, mcrop, fp, rng, n, size_m, decking=0.3, scatter=True):
    """Patches of missing tiles, cut row by row (stepped outline): dark underlay with battens (or bare
    decking), a few tiles still in place, recessed in the roof; the lost tiles lie around."""
    ys, xs = np.nonzero(fp)
    if len(ys) == 0:
        return
    h, w = fp.shape
    g = sc.gsd
    roof_med = np.median(crop[fp], 0).astype(np.float32)
    col = crop.astype(np.float32).copy()
    inner = cv2.erode(fp.astype(np.uint8), disk(max(1, 0.3 / g))) > 0
    iy, ix = np.nonzero(inner if inner.any() else fp)
    holes = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    row_m, tile_m = 0.33, 0.25
    for _ in range(int(n)):
        k = int(rng.integers(len(iy)))
        cx, cy = int(ix[k]), int(iy[k])
        ax = roof_axis(crop, fp, cx, cy, 1.5 / g)
        u = ((xx - cx) * np.cos(ax) + (yy - cy) * np.sin(ax)) * g      # along the rows (m)
        v = (-(xx - cx) * np.sin(ax) + (yy - cy) * np.cos(ax)) * g     # across the rows (m)
        half = rng.uniform(*size_m) / 2
        nrows = max(1, int(rng.uniform(0.3, 0.8) * 2 * half / row_m))
        row = np.floor(v / row_m).astype(np.int32)
        hole = np.zeros((h, w), bool)
        for j in range(-nrows // 2, nrows - nrows // 2):
            ext = half * (1 - 0.6 * abs(j) / max(1, nrows / 2)) * rng.uniform(0.6, 1.1)
            off = rng.normal(0, 0.15)
            # cut on whole tiles: the edge of the hole follows the tile joints
            lo = np.floor((off - ext) / tile_m) * tile_m
            hi = np.ceil((off + ext) / tile_m) * tile_m
            hole |= (row == j) & (u >= lo) & (u <= hi)
        cell = (np.floor(u / tile_m).astype(np.int32) * 7919 + row * 104729) % 1000
        keep = (cell < 120) & hole  # a few tiles still hang there
        hole &= fp & ~keep
        if not hole.any():
            continue
        holes |= hole
        if rng.random() < decking:
            panel = np.float32([108, 140, 172]) * rng.uniform(0.8, 1.0)
            seam = (np.abs((u % 1.22) - 0.61) > 0.6) | (np.abs((v % 2.44) - 1.22) > 1.21)
            tex = np.broadcast_to(panel, (h, w, 3)) * (1 + grain(h, w, rng, max(1, 0.02 / g), 0.05))[..., None]
            tex = tex * (1 - 0.35 * seam[..., None])
        else:
            under = np.float32([46, 48, 52])
            bat = np.abs((v % row_m) - row_m / 2) > row_m / 2 - 0.035
            wood = np.float32([90, 118, 145])
            tex = np.where(bat[..., None], wood, under) * (1 + grain(h, w, rng, max(1, 0.015 / g), 0.08))[..., None]
        col[hole] = tex[hole]
    if not holes.any():
        return
    zt = np.where(fp & ~holes, 0.05, 0.0).astype(np.float32)
    lt = get_light(sc)
    f = lt.amb + (1 - lt.amb) * (1 - lt.shadow(zt, 0.1))
    out = crop.astype(np.float32)
    out[holes] = col[holes] * f[holes][..., None]
    edge = holes & ~(cv2.erode(holes.astype(np.uint8), disk(1)) > 0)
    out[edge] = out[edge] * 0.7 + roof_med * 0.3
    crop[:] = out.clip(0, 255).astype(np.uint8)
    if scatter:
        scatter_tiles(sc, crop, mcrop, fp, holes, rng)


def scatter_tiles(sc, crop, mcrop, fp, holes, rng, count=None):
    g = sc.gsd
    h, w = fp.shape
    pal = Palette(sc, crop, fp, rng)
    ring = (cv2.dilate(fp.astype(np.uint8), disk(1.2 / g)) > 0) & ~fp & np.isin(mcrop, GROUND_ROAD)
    where = ring | (fp & ~holes & (rng.random((h, w)) < 0.05))
    area = holes.sum() * g * g
    col = crop.astype(np.float32).copy()
    cov = np.zeros((h, w), np.float32)
    z = np.zeros((h, w), np.float32)
    draw_pieces(sc, pal, rng, (col, cov, z), where, count or 3 + area * 8, (0.1, 0.2), 0.05,
                kinds=["roof", "roof", "roof", "concrete"], base_z=0.0)
    compose(sc, crop, mcrop, col, cov, z)
    mcrop[(cov > 0.5) & np.isin(mcrop, GROUND_ROAD)] = C.DEBRIS


def tarp(sc, crop, fp, rng):
    """Blue tarp with folds, held by planks / sandbags, following the roof axis."""
    ys, xs = np.nonzero(cv2.erode(fp.astype(np.uint8), disk(max(1, 0.5 / sc.gsd))) > 0)
    if len(ys) == 0:
        return
    g = sc.gsd
    h, w = fp.shape
    k = int(rng.integers(len(ys)))
    cx, cy = float(xs[k]), float(ys[k])
    ax = roof_axis(crop, fp, cx, cy, 1.5 / g)
    L, Wd = rng.uniform(2, 4) / g, rng.uniform(1.5, 3) / g
    pm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(pm, [cv2.boxPoints(((cx, cy), (L, Wd), np.degrees(ax))).astype(np.int32)], 1)
    sel = (pm > 0) & fp
    if not sel.any():
        return
    blue = np.float32([rng.integers(135, 165), rng.integers(100, 120), rng.integers(45, 70)])
    folds = directional_blur(fbm(h, w, rng, max(4, int(w * g / 0.8)), 3), ax + np.pi / 2, 0.6 / g)
    z = (folds * 0.12 + 0.02).astype(np.float32) * sel
    col = np.broadcast_to(blue, (h, w, 3)).astype(np.float32) * (1 + grain(h, w, rng, 2, 0.02))[..., None]
    # planks / sandbags holding the edges
    edge = sel & ~(cv2.erode(sel.astype(np.uint8), disk(max(1, 0.12 / g))) > 0)
    hold = edge & (fbm(h, w, rng, max(4, int(w * g / 1.0)), 1) > 0.6)
    col[hold] = np.float32([95, 130, 165])
    z[hold] = 0.1
    lt = get_light(sc)
    f = lt.amb + (1 - lt.amb) * lt.lambert(z * 3) * (1 - lt.shadow(z))
    a = feather(sel, 1)[..., None]
    crop[:] = (crop * (1 - a) + (col * f[..., None]).clip(0, 255) * a).astype(np.uint8)


def cracks(sc, crop, fp, rng, n):
    """Thin cracks on a roof / slab (earthquake)."""
    ys, xs = np.nonzero(fp)
    if len(ys) == 0:
        return
    g = sc.gsd
    m = np.zeros(fp.shape, np.uint8)
    for _ in range(int(n)):
        k = int(rng.integers(len(ys)))
        p = np.array([xs[k], ys[k]], np.float32)
        a = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(8, 20))):
            a += rng.normal(0, 0.5)
            q = p + np.array([np.cos(a), np.sin(a)]) * 0.25 / g
            cv2.line(m, tuple(p.astype(int)), tuple(q.astype(int)), 1, max(1, int(0.02 / g)))
            p = q
    sel = (m > 0) & fp
    crop[sel] = (crop[sel] * 0.45).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# buildings
# ---------------------------------------------------------------------------------------------
def collapse(sc, img, mask, b, b_lab, level, rng, wind=None, burnt=False, scenario="earthquake"):
    g = sc.gsd
    size = np.sqrt(b["area"])
    pad = int(size * 0.5 + 4.0 / g)
    x, y, w0, h0 = b["bbox"]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(sc.W, x + w0 + pad), min(sc.H, y + h0 + pad)
    crop, mcrop = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    fp = b_lab[y0:y1, x0:x1] == b["label"]
    src = sc.img0[y0:y1, x0:x1]
    h, w = fp.shape
    ground = 0.5 * sc.local_ground(x0, y0, x1, y1) + 0.5 * np.float32([158, 162, 166])
    pal = Palette(sc, src, fp, rng, burnt=burnt)
    windy = scenario in ("hurricane", "tornado")
    if windy:
        pal.weights = {"roof": 0.45, "wood": 0.25, "plaster": 0.18, "concrete": 0.12}
    ys, xs = np.nonzero(fp)

    if level == 2:
        # part of the roof caves in (one side of a jagged line): rubble + holes + exposed rafters
        ang = rng.uniform(0, np.pi)
        cx, cy = xs.mean() + rng.normal(0, size * 0.12), ys.mean() + rng.normal(0, size * 0.12)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        side = ((xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)) + (fbm(h, w, rng, 6, 2) - 0.5) * size * 0.25 > 0
        caved = fp & (side if rng.random() < 0.5 else ~side)
        if caved.sum() < 0.15 * fp.sum():
            caved = fp & (fbm(h, w, rng, 4, 2) > 0.5)
        missing_tiles(sc, crop, mcrop, fp & ~caved, rng, 2 + rng.integers(0, 3), (0.6, 1.6), scatter=False)
        if scenario == "earthquake":
            cracks(sc, crop, fp & ~caved, rng, 4)
        col, cov, z = rubble_layer(sc, src, caved, pal, rng, 0.8, ground, dust=0.25, voids=0.35)
        # rafters still spanning the hole, along the slope
        ax = roof_axis(src, fp, float(xs.mean()), float(ys.mean()), size / 2) + np.pi / 2
        n_r = int(np.sqrt(caved.sum()) * g / 0.9)
        roof_sections(sc, src, caved, (col, cov, z), rng, frac=0.4, inward=0.3)
        draw_pieces(sc, pal, rng, (col, cov, z), caved, n_r, (0.1, 0.1), 0.8, planks=1.0, angle=ax)
        compose(sc, crop, mcrop, col, cov, z)
        region = fp
    else:
        remove_shadow(sc, crop, mcrop, fp)
        noise = fbm(h, w, rng, 6, 3)
        spill = cv2.dilate(fp.astype(np.uint8), disk(max(2, (1.0 + 1.5 * rng.random()) / g))) > 0
        region = fp | (spill & (noise > 0.45) & np.isin(mcrop, GROUND_ROAD))
        region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, disk(max(1, 0.3 / g))) > 0
        col, cov, z = rubble_layer(sc, src, region, pal, rng, 1.6 if not windy else 0.9,
                                   ground if not burnt else np.float32([70, 72, 75]), dust=0.35,
                                   planks=0.15 if windy else 0.03)
        if not burnt:
            roof_sections(sc, src, fp, (col, cov, z), rng, frac=0.45 if not windy else 0.25)
            draw_pieces(sc, pal, rng, (col, cov, z), region, region.sum() * sc.gsd ** 2 * 0.8, (0.04, 0.15), 1.0)
        compose(sc, crop, mcrop, col, cov, z)

    full = np.zeros(mask.shape, bool)
    full[y0:y1, x0:x1] = region
    if not burnt:
        a = wind if wind is not None else rng.uniform(0, 2 * np.pi)
        scatter_debris(sc, img, mask, full, a, (4 + 3 * level) if windy else (0.8 + 0.6 * level),
                       0.25 + 0.1 * level, rng, pal, focused=windy)
    mcrop[region | fp] = C.BLD_BY_LEVEL[level]


def scatter_debris(sc, img, mask, origin, angle, length_m, density, rng, pal, focused=False):
    """Discrete pieces thrown around a damaged building (downwind if there is wind), each with its
    own shadow. Labels: debris on ground and roads."""
    g = sc.gsd
    L = length_m / g
    ys, xs = np.nonzero(origin)
    if len(ys) == 0:
        return
    x0, x1 = max(0, int(xs.min() - L)), min(sc.W, int(xs.max() + L))
    y0, y1 = max(0, int(ys.min() - L)), min(sc.H, int(ys.max() + L))
    crop, mcrop = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    o = origin[y0:y1, x0:x1].astype(np.uint8)
    h, w = o.shape
    d = cv2.distanceTransform(1 - o, cv2.DIST_L2, 5)
    prob = np.exp(-d / (0.35 * L))
    if focused:  # wind: mostly downwind
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cyc, cxc = np.nonzero(o)
        vx, vy = xx - cxc.mean(), yy - cyc.mean()
        along = (vx * np.cos(angle) + vy * np.sin(angle)) / (np.hypot(vx, vy) + 1e-3)
        prob *= np.clip(0.15 + along, 0, 1.15)
    free = np.isin(mcrop, GROUND_ROAD) & (o == 0)
    where = free & (rng.random((h, w)) < prob * 0.02)
    if not where.any():
        return
    col = crop.astype(np.float32).copy()
    cov = np.zeros((h, w), np.float32)
    z = np.zeros((h, w), np.float32)
    area = origin.sum() * g * g
    draw_pieces(sc, pal, rng, (col, cov, z), where, area * density, (0.05, 0.25), 0.1,
                base_z=0.0, planks=0.12 if focused else 0.02, angle=angle if focused else None)
    compose(sc, crop, mcrop, col, cov, z)
    mcrop[(cov > 0.5) & free] = C.DEBRIS


def scorch(sc, img, mask, b, b_lab, rng):
    """Minor fire damage: soot on the roof edge, some burnt / fallen tiles."""
    g = sc.gsd
    x, y, w0, h0 = b["bbox"]
    crop = img[y:y + h0, x:x + w0]
    fp = b_lab[y:y + h0, x:x + w0] == b["label"]
    d = cv2.distanceTransform(fp.astype(np.uint8), cv2.DIST_L2, 3) * g
    soot = np.clip(1 - d / 1.5, 0, 1) * (fbm(*fp.shape, rng, 6, 3) > 0.4) * fp
    soot = cv2.GaussianBlur(soot.astype(np.float32), (0, 0), max(1, 0.15 / g))
    over(crop, (crop * 0.25).astype(np.uint8), soot * 0.85)
    missing_tiles(sc, crop, mask[y:y + h0, x:x + w0], fp, rng, 1 + rng.integers(0, 2), (0.4, 1.0), scatter=False)
    mask[y:y + h0, x:x + w0][fp] = C.BLD_MINOR


def damage_building(sc, img, mask, b, b_lab, level, scenario, rng, wind=None):
    if level == 0:
        return
    if scenario == "wildfire":
        if level == 1:
            scorch(sc, img, mask, b, b_lab, rng)
        else:
            collapse(sc, img, mask, b, b_lab, level, rng, burnt=True, scenario=scenario)
        return
    if scenario == "flood":
        # the water shows the damage; houses lose part of their roof only when badly hit
        x, y, w0, h0 = b["bbox"]
        crop, mcrop = img[y:y + h0, x:x + w0], mask[y:y + h0, x:x + w0]
        fp = b_lab[y:y + h0, x:x + w0] == b["label"]
        if level >= 2:
            missing_tiles(sc, crop, mcrop, fp, rng, 1 + 2 * (level - 1), (0.5, 1.4), decking=0.2, scatter=False)
        mcrop[fp] = C.BLD_BY_LEVEL[level]
        return
    if level == 1:
        g = sc.gsd
        x, y, w0, h0 = b["bbox"]
        pad = int(2.5 / g)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(sc.W, x + w0 + pad), min(sc.H, y + h0 + pad)
        crop, mcrop = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        fp = b_lab[y0:y1, x0:x1] == b["label"]
        windy = scenario in ("hurricane", "tornado")
        missing_tiles(sc, crop, mcrop, fp, rng, 2 + rng.integers(0, 4 if windy else 2),
                      (0.5, 1.6) if windy else (0.3, 0.9), decking=0.35 if windy else 0.1)
        if windy and rng.random() < 0.45:
            tarp(sc, crop, fp, rng)
        if scenario == "earthquake":
            cracks(sc, crop, fp, rng, 3 + rng.integers(0, 4))
        mcrop[fp] = C.BLD_MINOR
    else:
        collapse(sc, img, mask, b, b_lab, level, rng, wind=wind, scenario=scenario)


# ---------------------------------------------------------------------------------------------
# flood
# ---------------------------------------------------------------------------------------------
def flood_depth(sc, mask, depth_field, coverage, rng):
    """At drone scale the water fills the street / yards: low-frequency slope from the hazard
    field + the ground type (roads lowest, lawns higher, buildings dry) + a little relief."""
    H, W = mask.shape
    g = sc.gsd
    D = max(H, W)
    smooth = cv2.GaussianBlur(depth_field.astype(np.float32), (0, 0), D / 8)
    smooth = (smooth - smooth.min()) / max(float(np.ptp(smooth)), 1e-6)
    elev = -0.6 * smooth
    elev += np.where(mask == C.ROAD, -0.12, 0.0) + np.where(np.isin(mask, [C.TREE]), 0.04, 0.0)
    elev += (fbm(H, W, rng, max(3, int(W * g / 4)), 3) - 0.5) * 0.08
    open_ground = ~np.isin(mask, C.BUILDINGS)
    level = float(np.quantile(elev[open_ground], np.clip(coverage, 0.05, 0.9)))
    d = np.clip(level - elev, 0, None) * open_ground  # metres (pseudo)
    d = cv2.GaussianBlur(d.astype(np.float32), (0, 0), max(1, 0.3 / g)) * open_ground
    return d


def flood_water(sc, img, mask, depth_field, coverage, rng, flow):
    g = sc.gsd
    H, W = mask.shape
    d = flood_depth(sc, mask, depth_field, coverage, rng)
    wet = d > 0.01
    wet = cv2.morphologyEx(wet.astype(np.uint8), cv2.MORPH_OPEN, disk(max(1, 0.2 / g))) > 0
    if not wet.any():
        return wet, d
    din = cv2.distanceTransform(wet.astype(np.uint8), cv2.DIST_L2, 5) * g
    dout = cv2.distanceTransform((~wet).astype(np.uint8), cv2.DIST_L2, 5) * g
    # muddy water: brown-olive, darker when deep, with suspended sediment swirls along the flow
    base = np.float32([rng.integers(78, 90), rng.integers(102, 112), rng.integers(116, 128)])
    swirl = directional_blur(fbm(H, W, rng, max(4, int(W * g / 3)), 3), flow, 1.5 / g)
    depth_vis = np.clip(din / 2.0, 0, 1) ** 0.7
    col = base * (1.06 - 0.25 * depth_vis[..., None]) + (swirl[..., None] - 0.5) * 18
    # what is under shallow water stays visible: tinted, a bit blurred (refraction)
    under = cv2.GaussianBlur(img, (0, 0), max(0.6, 0.02 / g)).astype(np.float32) * np.float32([0.6, 0.72, 0.8])
    alpha = np.clip(0.3 + 0.66 * depth_vis, 0, 0.95)
    water = under * (1 - alpha[..., None]) + col * alpha[..., None]
    # small wind ripples -> sky reflection and glints
    lt = get_light(sc)
    rip = directional_blur(fbm(H, W, rng, max(8, int(W * g / 0.35)), 2), flow + np.pi / 2, 0.25 / g)
    lam = lt.lambert((rip * 0.03).astype(np.float32))
    water *= (0.86 + 0.18 * lam)[..., None]
    glint = np.clip(lam - 1.18, 0, 1) * (fbm(H, W, rng, max(4, int(W * g / 2)), 2) > 0.6)
    water += (glint * 260)[..., None]
    sky = np.clip(directional_blur(fbm(H, W, rng, 3, 2), flow, 3 / g) - 0.55, 0, 1) * 30
    water += sky[..., None]
    a = feather(wet, max(1, int(0.03 / g)))[..., None]
    out = img.astype(np.float32) * (1 - a) + water * a
    # foam and floating debris along the edge and in eddies
    band = wet & (din < 0.35) & (fbm(H, W, rng, max(6, int(W * g / 0.8)), 2) > 0.5)
    foam = band & (rng.random((H, W)) < 0.12)
    foam = cv2.dilate(foam.astype(np.uint8), disk(max(1, 0.02 / g))) > 0
    out[foam] = out[foam] * 0.65 + np.float32([170, 180, 186]) * 0.35
    # wet ground and mud deposit just outside the water
    damp = np.clip(1 - dout / 0.8, 0, 1) * (~wet) * ~np.isin(mask, C.BUILDINGS)
    damp *= 0.6 + 0.4 * fbm(H, W, rng, max(4, int(W * g / 1.0)), 2)
    out *= (1 - 0.35 * damp)[..., None]
    img[:] = out.clip(0, 255).astype(np.uint8)
    # floating pieces (planks, branches, plastic) drifting along the flow
    pal = Palette(sc, sc.img0, np.isin(sc.mask0, C.BUILDINGS), rng)
    pal.weights = {"wood": 0.6, "plaster": 0.2, "concrete": 0.2}
    col_l = img.astype(np.float32).copy()
    cov = np.zeros((H, W), np.float32)
    z = np.zeros((H, W), np.float32)
    where = wet & (din > 0.2) & (din < 1.5) & (rng.random((H, W)) < 0.002)
    draw_pieces(sc, pal, rng, (col_l, cov, z), where, wet.sum() * g * g * 0.012, (0.05, 0.2), 0.03,
                base_z=0.0, planks=0.35, angle=flow)
    compose(sc, img, mask, col_l, cov, z)
    mask[wet] = C.WATER
    return wet, d


# ---------------------------------------------------------------------------------------------
# dust after an earthquake: local, around the collapses (not a global fog)
# ---------------------------------------------------------------------------------------------
def dust(sc, img, mask_after, rng, amount):
    g = sc.gsd
    src = np.isin(mask_after, [C.BLD_MAJOR, C.BLD_TOTAL, C.DEBRIS]).astype(np.float32)
    if src.sum() == 0:
        return
    cloud = cv2.GaussianBlur(src, (0, 0), 2.5 / g)
    cloud = cloud / max(float(cloud.max()), 1e-6) * fbm(sc.H, sc.W, rng, 4, 4)
    over(img, np.broadcast_to(np.float32([168, 176, 184]), img.shape), np.clip(cloud * amount, 0, 0.45))


# ---------------------------------------------------------------------------------------------
# vegetation (wind) and burnt ground (fire)
# ---------------------------------------------------------------------------------------------
def wind_litter(sc, img, mask, field, wind, rng, strength):
    """Leaves and broken branches torn from the trees / hedges and blown downwind (at drone scale a
    whole tree cannot simply be moved: its crown would look like a cut-out)."""
    g = sc.gsd
    H, W = mask.shape
    tree = (sc.mask0 == C.TREE).astype(np.uint8)
    if tree.sum() == 0:
        return 0
    # source: crowns, pushed downwind
    L = 3.0 / g
    src = tree.astype(np.float32) * field
    spread = np.zeros_like(src)
    for t in np.linspace(0, 1, 10):
        M = np.float32([[1, 0, np.cos(wind) * L * t], [0, 1, np.sin(wind) * L * t]])
        spread = np.maximum(spread, cv2.warpAffine(src, M, (W, H)) * (1 - 0.8 * t))
    free = np.isin(mask, GROUND_ROAD)
    where = free & (rng.random((H, W)) < spread * strength * 0.004)
    n = int(where.sum())
    if n == 0:
        return 0
    col = img.astype(np.float32).copy()
    cov = np.zeros((H, W), np.float32)
    z = np.zeros((H, W), np.float32)
    ys, xs = np.nonzero(where)
    for y, x in zip(ys, xs):
        if rng.random() < 0.25:  # twig / branch
            Lb = rng.uniform(0.3, 1.2) / g
            a = wind + rng.normal(0, 0.6)
            p2 = (int(x + np.cos(a) * Lb), int(y + np.sin(a) * Lb))
            c = np.float32([45, 62, 82]) * rng.uniform(0.8, 1.2)
            th = max(1, int(rng.uniform(0.02, 0.05) / g))
            cv2.line(col, (int(x), int(y)), p2, c.tolist(), th)
            cv2.line(cov, (int(x), int(y)), p2, 1.0, th)
            cv2.line(z, (int(x), int(y)), p2, 0.04, th)
        else:  # clump of leaves
            r = max(1, int(rng.uniform(0.04, 0.15) / g))
            c = (np.float32([40, 95, 70]) if rng.random() < 0.7 else np.float32([50, 90, 120])) * rng.uniform(0.7, 1.2)
            cv2.circle(col, (int(x), int(y)), r, c.tolist(), -1)
            cv2.circle(cov, (int(x), int(y)), r, 1.0, -1)
            cv2.circle(z, (int(x), int(y)), r, 0.03, -1)
    compose(sc, img, mask, col, cov, z)
    return n


def burnt_texture(sc, before, after, mask, rng):
    """Burnt ground at drone scale. Only vegetation turns to charcoal and ash; paving, gravel and
    objects in the fire area are just blackened by soot. Fine grain at its real size."""
    g = sc.gsd
    H, W = mask.shape
    Lb = cv2.cvtColor(before, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    La = cv2.cvtColor(after, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    burnt = (La < 0.75 * Lb) & ~np.isin(mask, C.BUILDINGS + [C.WATER])
    if burnt.sum() == 0:
        return
    f = before.astype(np.float32)
    veg = (((2 * f[..., 1] - f[..., 0] - f[..., 2]) > 10) | (sc.mask0 == C.TREE))
    veg = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_CLOSE, disk(max(1, 0.05 / g))) > 0
    a = cv2.GaussianBlur(burnt.astype(np.float32), (0, 0), max(1, 0.05 / g))[..., None]
    out = after.astype(np.float32)
    # not vegetation: soot over the original material (its texture stays readable)
    soot = f * (0.45 + 0.2 * fbm(H, W, rng, max(4, int(W * g / 1.5)), 3))[..., None]
    nv = (~veg)[..., None] * a
    out = out * (1 - nv) + soot * nv
    # vegetation: charcoal grain + soft ash patches
    vg = veg[..., None] * a
    tex = out * (1 + grain(H, W, rng, max(1, 0.015 / g), 0.22) + grain(H, W, rng, max(1, 0.08 / g), 0.1))[..., None]
    ashp = np.clip((fbm(H, W, rng, max(4, int(W * g / 0.8)), 3) - 0.62) / 0.2, 0, 1)[..., None] * 0.55
    tex = tex * (1 - ashp) + np.float32([150, 153, 156]) * ashp
    out = out * (1 - vg) + tex * vg
    after[:] = out.clip(0, 255).astype(np.uint8)

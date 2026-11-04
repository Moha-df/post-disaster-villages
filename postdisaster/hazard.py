"""
Hazard fields: WHERE and HOW STRONGLY the disaster hits, as a smooth map in [0, 1].

Damage is not drawn independently per building: it follows a physically-motivated field
(distance to an epicentre, wind gusts along a direction, a tornado track, flood depth on a
pseudo-elevation model, a fire front driven by vegetation). Neighbouring houses therefore get
similar damage, like in real post-event imagery.
"""

import cv2
import numpy as np

from . import classes as C


# ---------------------------------------------------------------------------------------------
# noise helpers
# ---------------------------------------------------------------------------------------------
def fbm(h, w, rng, scale=4, octaves=4, gain=0.5):
    """Fractal (multi-octave) smooth noise in [0, 1]."""
    h, w = max(1, int(h)), max(1, int(w))
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    for o in range(octaves):
        cells = scale * 2 ** o
        gh, gw = max(2, round(cells * h / max(h, w))), max(2, round(cells * w / max(h, w)))
        out += amp * cv2.resize(rng.random((gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        amp *= gain
    out -= out.min()
    return out / max(out.max(), 1e-6)


def directional_blur(field, angle, length):
    """Smears a field along a direction (gust streaks, smoke plumes, debris trails)."""
    L = max(3, int(length)) | 1
    k = np.zeros((L, L), np.float32)
    c = L // 2
    dx, dy = np.cos(angle), np.sin(angle)
    cv2.line(k, (int(c - dx * c), int(c - dy * c)), (int(c + dx * c), int(c + dy * c)), 1.0, 1)
    k /= k.sum()
    return cv2.filter2D(field, -1, k, borderType=cv2.BORDER_REFLECT)


def normalize(x):
    x = x - x.min()
    return x / max(float(x.max()), 1e-6)


# ---------------------------------------------------------------------------------------------
# one hazard model per scenario. Each returns a dict with at least `field` (HxW in [0,1])
# ---------------------------------------------------------------------------------------------
def earthquake(mask, rng, sev):
    H, W = mask.shape
    D = max(H, W)
    ang = rng.uniform(0, 2 * np.pi)
    r = D * rng.uniform(0.2, 1.2)  # epicentre can be outside the tile
    ex, ey = W / 2 + r * np.cos(ang), H / 2 + r * np.sin(ang)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - ex, yy - ey)
    attenuation = np.exp(-d / (D * rng.uniform(0.6, 1.2)))
    site = fbm(H, W, rng, scale=3, octaves=3)  # soft soils amplify shaking
    field = normalize(0.7 * attenuation + 0.3 * site)
    return dict(field=np.clip(field * (0.45 + 0.65 * sev), 0, 1),
                epicentre=[round(float(ex), 1), round(float(ey), 1)])


def hurricane(mask, rng, sev):
    H, W = mask.shape
    wind = rng.uniform(0, 2 * np.pi)
    gusts = directional_blur(fbm(H, W, rng, scale=6, octaves=4), wind, max(H, W) * 0.25)
    field = normalize(0.35 + gusts)
    return dict(field=np.clip(field * (0.4 + 0.65 * sev), 0, 1), wind=float(wind))


def tornado(mask, rng, sev):
    H, W = mask.shape
    D = max(H, W)
    ang = rng.uniform(0, np.pi)
    cx, cy = W * rng.uniform(0.3, 0.7), H * rng.uniform(0.3, 0.7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    across = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    along = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    wobble = np.sin(along / D * rng.uniform(4, 9) + rng.uniform(0, 6)) * D * 0.04
    half = D * rng.uniform(0.07, 0.13) * (0.6 + 0.6 * sev)
    core = np.clip(1.25 - np.abs(across + wobble) / half, 0, 1)
    field = np.clip(core * (0.85 + 0.3 * fbm(H, W, rng, 8, 3)), 0, 1)
    # debris is thrown along the rotation, i.e. along the track
    return dict(field=field, wind=float(ang), track_angle=float(ang))


def flood(mask, rng, sev):
    """Pseudo-DEM: low near existing water and roads, smooth relief elsewhere.
    The water level is set so that a target fraction of the ground is under water."""
    H, W = mask.shape
    D = max(H, W)
    water = mask == C.WATER
    relief = fbm(H, W, rng, scale=2, octaves=4)
    if water.sum() > 0.002 * H * W:
        d = cv2.distanceTransform((~water).astype(np.uint8), cv2.DIST_L2, 5) / D
        elev = 0.65 * np.clip(d / 0.5, 0, 1) + 0.35 * relief
    else:  # no river in the tile: a valley crossing the image
        ang = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        across = -(xx - W * rng.uniform(.3, .7)) * np.sin(ang) + (yy - H * rng.uniform(.3, .7)) * np.cos(ang)
        across += (fbm(H, W, rng, 2, 2) - .5) * D * 0.3
        elev = 0.65 * np.clip(np.abs(across) / (0.5 * D), 0, 1) + 0.35 * relief
    elev = elev - 0.02 * (mask == C.ROAD) + 0.03 * np.isin(mask, C.BUILDINGS)  # roads low, houses raised
    target = float(np.clip(0.10 + 0.30 * sev + rng.uniform(-0.05, 0.05), 0.08, 0.75))
    open_ground = ~np.isin(mask, C.BUILDINGS)
    level = float(np.quantile(elev[open_ground], target))
    depth = np.clip((level - elev) / max(level - elev.min(), 1e-6), 0, 1)  # 0 dry .. 1 deepest
    depth[water] = np.maximum(depth[water], 0.6)
    return dict(field=depth, depth=depth, level=level, coverage=target)


def wildfire(mask, img, rng, sev):
    """Fire front entering from one side; it burns where there is fuel (vegetation)."""
    H, W = mask.shape
    D = max(H, W)
    f = img.astype(np.float32)
    exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / 255.0  # excess green
    fuel = np.clip(0.4 * (mask == C.TREE) + 0.6 * np.clip(exg * 4, 0, 1) + 0.25 * (mask == C.BACKGROUND), 0, 1)
    fuel = cv2.GaussianBlur(fuel, (0, 0), D / 80)
    wind = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    along = ((xx - W / 2) * np.cos(wind) + (yy - H / 2) * np.sin(wind)) / D + 0.5  # 0 = where it enters
    reach = 0.2 + 0.55 * sev
    front = np.clip((reach - along) / 0.25, 0, 1)
    noise = directional_blur(fbm(H, W, rng, 5, 4), wind, D * 0.1)
    burn = np.clip(front * (0.45 + 0.8 * fuel) * (0.6 + 0.6 * noise), 0, 1)
    return dict(field=burn, wind=float(wind), fuel=fuel)


def compute(scenario, mask, img, rng, sev):
    if scenario == "wildfire":
        return wildfire(mask, img, rng, sev)
    return {"earthquake": earthquake, "hurricane": hurricane, "tornado": tornado,
            "flood": flood}[scenario](mask, rng, sev)


# ---------------------------------------------------------------------------------------------
# hazard -> damage level per building
# ---------------------------------------------------------------------------------------------
THRESHOLDS = {  # latent damage z -> levels 1, 2, 3 (minor, major, destroyed)
    "earthquake": (0.22, 0.42, 0.60),
    "hurricane": (0.30, 0.55, 0.76),
    "tornado": (0.18, 0.45, 0.70),
    "flood": (0.08, 0.35, 0.80),
    "wildfire": (0.35, 0.52, 0.66),
}


def damage_levels(buildings, b_lab, hz, scenario, rng):
    """Ordinal model: z = mean hazard on the footprint + vulnerability + noise."""
    field = hz["field"]
    areas = np.array([b["area"] for b in buildings], np.float32)
    med = float(np.median(areas)) if len(areas) else 1.0
    thr = np.array(THRESHOLDS[scenario])
    levels = []
    for b in buildings:
        x, y, w, h = b["bbox"]
        fp = b_lab[y:y + h, x:x + w] == b["label"]
        if scenario == "flood":  # depth just around the house matters
            ring = cv2.dilate(fp.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            x0, y0 = max(0, x - 3), max(0, y - 3)
            sub = field[y0:y + h + 3, x0:x + w + 3]
            pad = np.zeros_like(sub, bool)
            pad[y - y0:y - y0 + h, x - x0:x - x0 + w] = ring[:sub.shape[0] - (y - y0), :sub.shape[1] - (x - x0)]
            haz = float(sub[pad].max()) if pad.any() else 0.0
        else:
            haz = float(field[y:y + h, x:x + w][fp].mean())
        # small, light houses are more fragile than big ones
        vuln = 0.08 * np.clip(np.log(med / max(b["area"], 1)), -1, 1)
        z = haz + vuln + rng.normal(0, 0.08)
        levels.append(int((z > thr).sum()))
        b["hazard"] = round(haz, 3)
    return levels

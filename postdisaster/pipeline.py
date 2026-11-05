"""Full scenario: hazard field -> building damage -> scene-wide effects -> sensor matching."""

import cv2
import numpy as np

from . import classes as C
from . import drone as D
from . import effects as E
from . import hazard as Z

SCENARIOS = ["earthquake", "hurricane", "tornado", "flood", "wildfire"]
DRONE_GSD = 0.12  # below 12 cm/px the high-resolution (drone) renderer is used


def render_mode(gsd, opts=None):
    """'aerial' (2D rendering, 0.25-0.5 m/px) or 'drone' (2.5D rendering with relief and shadows)."""
    r = (opts or {}).get("render", "auto")
    return r if r in ("aerial", "drone") else ("drone" if gsd < DRONE_GSD else "aerial")


def split_buildings(img, mask, min_area):
    """Building pixels -> separate houses. Touching houses are split with a watershed on the
    IMAGE (roof edges), seeded by the eroded footprints."""
    bld = (mask == C.BLD_NONE).astype(np.uint8)
    core = cv2.erode(bld, E.disk(2))
    n, seeds = cv2.connectedComponents(core)
    markers = seeds.astype(np.int32) + 1  # 1 = not a building
    markers[(bld > 0) & (core == 0)] = 0  # unknown band, decided by the watershed
    cv2.watershed(cv2.GaussianBlur(img, (3, 3), 0), markers)
    lab = np.where(bld > 0, np.maximum(markers - 1, 0), 0).astype(np.int32)
    # watershed lines (-1) and lost thin parts: give them to the nearest house
    lost = (bld > 0) & (lab <= 0)
    if lost.any():
        _, idx = cv2.distanceTransformWithLabels((lab <= 0).astype(np.uint8), cv2.DIST_L2, 3,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(lab > 0)
        if len(ys):
            lut = np.zeros(idx.max() + 1, np.int32)
            lut[idx[ys, xs]] = lab[ys, xs]
            lab[lost] = lut[idx[lost]]
    # small parts that never got a seed (tiny houses) become their own instance
    rest = ((bld > 0) & (lab <= 0)).astype(np.uint8)
    m, rl = cv2.connectedComponents(rest)
    lab[rl > 0] = rl[rl > 0] + lab.max()
    out = []
    for i in np.unique(lab[lab > 0]):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < min_area:
            lab[lab == i] = 0
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        out.append(dict(id=len(out), label=int(i), area=int(len(ys)),
                        bbox=[x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1],
                        centroid=[round(float(xs.mean()), 1), round(float(ys.mean()), 1)]))
    return out, lab


def generate(img, mask, scenario, sev, gsd, rng, buildings=None, opts=None):
    """img: BGR uint8, mask: RescueNet ids (before). Returns after image, after mask, info."""
    sc = E.Scene(img, mask, gsd, rng)
    after, m = img.copy(), mask.copy()
    min_area = int(np.ceil(6 / gsd ** 2))  # ~6 m2
    blds, b_lab = buildings if buildings is not None else split_buildings(img, mask, min_area)
    blds = [dict(b) for b in blds]

    hz = Z.compute(scenario, mask, img, rng, sev, opts)
    wind = hz.get("wind")
    levels = Z.damage_levels(blds, b_lab, hz, scenario, rng)
    flow = rng.uniform(0, 2 * np.pi)
    # draw the worst damage last so its debris lies on top
    mode = render_mode(gsd, opts)
    R = D if mode == "drone" else E
    for b, lvl in sorted(zip(blds, levels), key=lambda t: t[1]):
        b["damage_level"] = lvl
        R.damage_building(sc, after, m, b, b_lab, lvl, scenario, rng,
                          wind=flow if scenario == "flood" else wind)

    # tree crowns hanging over a damaged house stay on top (occlusion), except in a fire
    if scenario != "wildfire":
        f = img.astype(np.float32)
        crown = (2 * f[..., 1] - f[..., 0] - f[..., 2] > 12) & np.isin(mask, C.BUILDINGS) & (m != mask)
        crown = cv2.morphologyEx(crown.astype(np.uint8), cv2.MORPH_OPEN, E.disk(1)) > 0
        E.over(after, img, E.feather(crown, 1) * 0.95)

    extra = {"render": mode}
    if scenario in ("hurricane", "tornado"):
        strength = 1.0 if scenario == "tornado" else 0.55
        if mode == "drone":
            extra["litter_pieces"] = D.wind_litter(sc, after, m, hz["field"], wind, rng, strength * (0.5 + sev))
        else:
            extra["fallen_trees"] = E.topple_trees(sc, after, m, hz["field"], wind, rng, strength * (0.5 + sev))
        E.vegetation_stress(sc, after, m, hz["field"], 0.8 + 0.6 * sev, rng, "wind")
    elif scenario == "flood":
        # 10 m halo, capped so very high resolution scenes (drone, ~2 cm/px) stay tractable
        mud = cv2.GaussianBlur((hz["depth"] > 0.02).astype(np.float32), (0, 0), min(sc.m(10), 60.0))
        E.vegetation_stress(sc, after, m, np.clip(mud * 2, 0, 1), 0.7, rng, "mud")
        if mode == "drone":
            wet, _ = D.flood_water(sc, after, m, hz["depth"], hz.get("coverage", 0.3), rng, flow)
        else:
            wet = E.flood_water(sc, after, m, hz["depth"], rng, flow)
        extra["flooded_fraction"] = round(float(wet.mean()), 3)
    elif scenario == "wildfire":
        extra["burnt_fraction"] = round(E.burn_scar(sc, after, m, hz["field"], rng), 3)
        if mode == "drone":
            D.burnt_texture(sc, img, after, m, rng)
    elif scenario == "earthquake":
        extra["epicentre"] = hz["epicentre"]

    after = E.match_sensor(img, after)
    if scenario == "earthquake":  # dust from the collapses
        if mode == "drone":
            D.dust(sc, after, m, rng, 0.25 + 0.2 * sev)
        else:
            E.haze(sc, after, rng, 0.12 + 0.12 * sev)
    if scenario == "wildfire":
        E.smoke(sc, after, hz["field"], wind, rng, sev)
    if wind is not None:
        extra["wind_direction_deg"] = round(float(np.degrees(wind)) % 360, 1)

    return after, m, dict(buildings=blds, b_lab=b_lab, hazard=hz["field"], extra=extra)


def change_map(before, after, mask_b, mask_a):
    """Semantic change: label changed, or the pixels were clearly repainted."""
    diff = cv2.absdiff(before, after).max(2)
    ch = (mask_b != mask_a) | (diff > 30)
    return cv2.morphologyEx(ch.astype(np.uint8), cv2.MORPH_OPEN, E.disk(1)) > 0

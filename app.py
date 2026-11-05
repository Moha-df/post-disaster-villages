"""
app.py : interface web interactive (Gradio)

    python app.py            ->  http://127.0.0.1:7860
"""

import inspect
import json
import os
import tempfile
import time
import zipfile
import zlib
from functools import lru_cache

import cv2
import gradio as gr
import numpy as np

import generate as G
import site_doc
from postdisaster import augment as AUG
from postdisaster import classes as C
from postdisaster import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
VILLAGE_DIR = os.path.join(HERE, "data", "villages")
DRONE_DIR = os.path.join(HERE, "data", "drone")  # python download_drone.py
GSD = 0.3
LABELS = {"earthquake": "Séisme", "hurricane": "Ouragan", "tornado": "Tornade",
          "flood": "Inondation", "wildfire": "Feu de forêt"}
POINT_HELP = {"earthquake": "l'épicentre", "tornado": "le centre du couloir de la tornade",
              "wildfire": "le point de départ du feu"}
WIND = ("hurricane", "tornado", "wildfire")
COUNTRIES = {"aachen": "Allemagne", "muenster": "Allemagne", "koeln": "Allemagne", "tyrolw": "Autriche",
             "baybay": "Philippines", "mahe": "Seychelles", "zanzibar": "Tanzanie", "austin": "États-Unis",
             "graz": "Autriche (drone)"}
DMG_HEX = ["#3cc83c", "#f5d800", "#ff8c00", "#dc1e1e"]
DMG_FR = ["Intact", "Mineur", "Majeur", "Détruit"]


def drone_gsd():
    """name -> m/px of the stored drone scenes (Semantic Drone Dataset, ~2 cm/px)."""
    f = os.path.join(DRONE_DIR, "scenes.json")
    if not os.path.exists(f):
        return {}
    with open(f, encoding="utf-8") as fh:
        return {s["name"]: s["gsd_m"] for s in json.load(fh)["scenes"]}


def villages():
    out = []
    for d in (VILLAGE_DIR, DRONE_DIR):
        if os.path.isdir(os.path.join(d, "images")):
            out += sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(d, "images")) if f.endswith(".png"))
    return [(f"{COUNTRIES.get(n.rsplit('_', 1)[0], '?')}  ·  {n}", n) for n in out]


@lru_cache(maxsize=32)
def load_village(name, side):
    """Image BGR + RescueNet mask + building instances, resized so the long side = `side`."""
    drone = drone_gsd()
    d = DRONE_DIR if name in drone else VILLAGE_DIR
    img = cv2.imread(os.path.join(d, "images", name + ".png"), cv2.IMREAD_COLOR)
    raw = cv2.imread(os.path.join(d, "masks", name + ".png"), cv2.IMREAD_GRAYSCALE)
    return prepare(img, raw, side, drone.get(name, GSD))


def prepare(img, raw, side, gsd0=GSD):
    """gsd0: metres per pixel of the image as stored (OpenEarthMap tiles: ~0.3)."""
    s = side / max(img.shape[:2])
    if abs(s - 1) > 1e-3:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    raw = cv2.resize(raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = C.oem_to_rescuenet(raw)
    gsd = gsd0 / s
    blds = P.split_buildings(img, mask, int(np.ceil(6 / gsd ** 2)))
    return img, mask, gsd, blds


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mark(img_bgr, point, scenario, wind_deg):
    """Before image with the clicked point and the wind arrow drawn on it."""
    v = rgb(img_bgr).copy()
    H, W = v.shape[:2]
    r = max(6, W // 50)
    if point is not None and scenario in POINT_HELP:
        x, y = int(point[0] * W), int(point[1] * H)
        cv2.circle(v, (x, y), r + 3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(v, (x, y), r, (255, 212, 0), 3, cv2.LINE_AA)
        cv2.circle(v, (x, y), 3, (255, 212, 0), -1, cv2.LINE_AA)
    if wind_deg is not None and scenario in WIND:
        a = np.radians(wind_deg)
        c = np.array([W - 4 * r, 4 * r])
        d = np.array([np.cos(a), np.sin(a)]) * 3 * r
        p0, p1 = tuple((c - d).astype(int)), tuple((c + d).astype(int))
        cv2.arrowedLine(v, p0, p1, (255, 255, 255), 6, cv2.LINE_AA, tipLength=0.4)
        cv2.arrowedLine(v, p0, p1, (255, 212, 0), 3, cv2.LINE_AA, tipLength=0.4)
    return v


# ---------------------------------------------------------------------------------------------
# HTML fragments
# ---------------------------------------------------------------------------------------------
def stats_html(scenario, severity, counts, n_bld, changed, extra, fired, W, H, gsd, dt, has_point):
    tiles = "".join(
        f"<div class='tile'><div class='bar' style='background:{DMG_HEX[i]}'></div>"
        f"<div class='num'>{int(counts[i])}</div><div class='lbl'>{DMG_FR[i]}</div></div>" for i in range(4))
    rows = [("Bâtiments", f"{n_bld}"), ("Pixels modifiés", f"{100 * changed:.1f} %")]
    if "wind_direction_deg" in extra:
        rows.append(("Vent / trajectoire", f"{extra['wind_direction_deg']:.0f}°"))
    if "epicentre" in extra:
        e = extra["epicentre"]
        rows.append(("Épicentre", f"{e[0]:.0f}, {e[1]:.0f} px" + ("" if has_point else " (aléatoire)")))
    if "flooded_fraction" in extra:
        rows.append(("Surface inondée", f"{100 * extra['flooded_fraction']:.0f} %"))
    if "burnt_fraction" in extra:
        rows.append(("Surface brûlée", f"{100 * extra['burnt_fraction']:.0f} %"))
    if "fallen_trees" in extra:
        rows.append(("Arbres tombés", f"{len(extra['fallen_trees'])}"))
    if "litter_pieces" in extra:
        rows.append(("Feuilles / branches", f"{extra['litter_pieces']}"))
    if "render" in extra:
        rows.append(("Rendu", "drone (relief, ombres)" if extra["render"] == "drone" else "aérien (2D)"))
    res = f"{100 * gsd:.1f} cm/px" if gsd < 0.1 else f"{gsd:.2f} m/px"
    rows.append(("Résolution", f"{W} × {H} px, {res}"))
    rows.append(("Calcul", f"{dt:.2f} s"))
    kv = "".join(f"<div class='k'>{k}</div><div class='v'>{v}</div>" for k, v in rows)
    chips = "".join(f"<span class='code'>{f}</span>" for f in fired) or "<span class='muted'>aucune</span>"
    return f"""
<div class='stats'>
  <div class='stats-head'><span class='eyebrow'>Résultat</span>
    <span class='title'>{LABELS[scenario]}</span><span class='pill'>sévérité {severity:.2f}</span></div>
  <div class='tiles'>{tiles}</div>
  <div class='kv'>{kv}</div>
  <div class='aug'><span class='k'>Albumentations appliquées</span><div>{chips}</div></div>
</div>"""


def legend(names, palette, extra=None):
    items = [(n, palette[i]) for i, n in enumerate(names)] + (extra or [])
    return "<div class='legend'>" + "".join(
        f"<span><i style='background:rgb({int(c[0])},{int(c[1])},{int(c[2])})'></i>{n}</span>"
        for n, c in items) + "</div>"


HEADER = """
<div class='tape'></div>
<div class='hero'>
  <h1>Village <em>/</em><br>post-catastrophe</h1>
  <p>Une image aérienne d'un village intact devient la même scène après un séisme, un ouragan, une tornade,
     une inondation ou un feu. Les dégâts suivent une carte d'aléa et chaque pixel reste labellisé.</p>
  <div class='meta'>
    <div><span class='coord'>Source</span> <b>OpenEarthMap</b>, 44 pays</div>
    <div><span class='coord'>Résolution</span> <b>0,25 à 0,5 m/px</b></div>
    <div><span class='coord'>Labels</span> <b>RescueNet / xBD</b></div>
    <div><span class='coord'>Augmentation</span> <b>Albumentations</b></div>
  </div>
</div>"""

PLACEHOLDER = "<div class='stats empty'>En attente du premier calcul</div>"


# ---------------------------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------------------------
def run(village, upload_img, upload_mask, scenario, severity, seed, wind_auto, wind_deg, coverage,
        point, quality, sensor, render="auto"):
    t0 = time.time()
    side = 512 if quality == "fast" else 1000
    if upload_img is not None:
        if upload_mask is None:
            raise gr.Error("Pour une image personnelle, ajoutez aussi son masque OpenEarthMap (PNG, valeurs 0 à 8).")
        raw = upload_mask if upload_mask.ndim == 2 else upload_mask[..., 0]
        img, mask, gsd, blds = prepare(cv2.cvtColor(upload_img, cv2.COLOR_RGB2BGR), raw.astype(np.uint8), side)
        stem = "mon_image"
    else:
        img, mask, gsd, blds = load_village(village, side)
        stem = village
    H, W = mask.shape
    opts = {}
    if point is not None and scenario in POINT_HELP:
        opts["point"] = (point[0] * W, point[1] * H)
    if not wind_auto and scenario in WIND:
        opts["wind_deg"] = wind_deg
    if scenario == "flood":
        opts["coverage"] = coverage
    opts["render"] = render

    rng = np.random.default_rng([int(seed), zlib.crc32(stem.encode()), P.SCENARIOS.index(scenario), 0])
    after, ma, info = P.generate(img, mask, scenario, severity, gsd, rng, buildings=blds, opts=opts)
    change = P.change_map(img, after, mask, ma)
    fired, clean = [], after.copy()
    if sensor:
        after, fired = AUG.apply_post_event(after, scenario, int(rng.integers(1 << 30)))
    xbd = C.rescuenet_to_xbd(ma)
    counts = np.bincount([b["damage_level"] for b in info["buildings"]], minlength=4)

    html = stats_html(scenario, severity, counts, len(info["buildings"]), float(change.mean()), info["extra"],
                      fired, W, H, gsd, time.time() - t0, "point" in opts)
    dmg = C.colorize(xbd, C.XBD_PALETTE)
    dmg[change & (xbd == 0)] = (90, 90, 90)
    overlay = cv2.addWeighted(after, 0.5, dmg, 0.5, 0)
    state = dict(stem=stem, scenario=scenario, severity=severity, seed=int(seed), gsd=gsd, img=img, mask=mask,
                 after=after, clean=clean, ma=ma, xbd=xbd, change=change, hazard=info["hazard"], info=info,
                 fired=fired, extra=info["extra"])
    return ((rgb(img), rgb(after)), rgb(C.colorize(ma)), rgb(overlay), rgb(G.heat(info["hazard"])),
            (change * 255).astype(np.uint8), html, state)


def make_zip(state):
    if not state:
        raise gr.Error("Générez d'abord une image.")
    name = f"{state['stem']}_{state['scenario']}_s{state['seed']}"
    meta = dict(village=state["stem"], severity=round(state["severity"], 3), gsd_m=round(state["gsd"], 3),
                seed=state["seed"], post_event_albumentations=state["fired"], **state["extra"])
    path = os.path.join(tempfile.gettempdir(), name + ".zip")
    enc = lambda im, ext: cv2.imencode(ext, im)[1].tobytes()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"before/{name}.jpg", enc(state["img"], ".jpg"))
        z.writestr(f"after/{name}.jpg", enc(state["after"], ".jpg"))
        z.writestr(f"masks_before/{name}_lab.png", enc(state["mask"], ".png"))
        z.writestr(f"masks/{name}_lab.png", enc(state["ma"], ".png"))
        z.writestr(f"xbd/{name}_damage.png", enc(state["xbd"], ".png"))
        z.writestr(f"change/{name}_change.png", enc(state["change"].astype(np.uint8) * 255, ".png"))
        z.writestr(f"hazard/{name}_hazard.png", enc((state["hazard"] * 255).astype(np.uint8), ".png"))
        z.writestr(f"labels_xbd/{name}.json", json.dumps(
            G.xbd_json(name, state["info"], state["info"]["b_lab"], state["scenario"], meta), indent=1))
    return path


def crops(state, n, size, seed):
    if not state:
        raise gr.Error("Générez d'abord une image dans l'onglet Simulation.")
    size = int(min(size, *state["mask"].shape))
    out = []
    for j, c in enumerate(AUG.paired_crops(rgb(state["img"]), rgb(state["after"]), state["mask"], state["ma"],
                                           state["xbd"], state["change"], int(n), size, int(seed))):
        dmg = rgb(C.colorize(c["xbd"], C.XBD_PALETTE))
        out += [(c["before"], f"Crop {j + 1} · avant"), (c["after"], f"Crop {j + 1} · après"),
                (dmg, f"Crop {j + 1} · dégâts")]
    return out


def post_event_demo(state, seed):
    if not state:
        raise gr.Error("Générez d'abord une image dans l'onglet Simulation.")
    out, fired = AUG.apply_post_event(state["clean"], state["scenario"], int(seed))
    chips = "".join(f"<span class='code'>{f}</span>" for f in fired) or "<span class='muted'>aucune</span>"
    return (rgb(state["clean"]), rgb(out)), f"<div class='aug'><span class='k'>Transformations tirées</span><div>{chips}</div></div>"


# ---------------------------------------------------------------------------------------------
# export en lot : toutes les combinaisons village x catastrophe x version, dans un seul zip
# ---------------------------------------------------------------------------------------------
def batch_export(vills, scenarios, n_per, sev_min, sev_max, random_point, random_wind, quality,
                 n_crops, crop_size, sensor, seed, progress=gr.Progress()):
    if not vills or not scenarios:
        raise gr.Error("Choisissez au moins un village et une catastrophe.")
    side = 512 if quality == "fast" else 1000
    sev_lo, sev_hi = sorted((float(sev_min), float(sev_max)))
    jobs = [(v, s, i) for v in vills for s in scenarios for i in range(int(n_per))]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(tempfile.gettempdir(), f"post_disaster_lot_{stamp}.zip")
    enc = lambda im, ext, q=92: cv2.imencode(ext, im, [cv2.IMWRITE_JPEG_QUALITY, q] if ext == ".jpg" else [])[1].tobytes()
    rows, thumbs, totals, t0 = [], [], np.zeros(4, int), time.time()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for v in vills:
            img, mask, gsd, _ = load_village(v, side)
            z.writestr(f"before/{v}.jpg", enc(img, ".jpg"))
            z.writestr(f"masks_before/{v}_lab.png", enc(mask, ".png"))
        for k, (v, sc, i) in enumerate(jobs):
            progress((k, len(jobs)), desc=f"{v} · {LABELS[sc]} · version {i + 1}")
            img, mask, gsd, blds = load_village(v, side)
            H, W = mask.shape
            rng = np.random.default_rng([int(seed), zlib.crc32(v.encode()), P.SCENARIOS.index(sc), i])
            sev = float(rng.uniform(sev_lo, sev_hi))
            opts = {}
            if random_point and sc in POINT_HELP:  # ailleurs dans le village à chaque version
                opts["point"] = (float(rng.uniform(0.1, 0.9)) * W, float(rng.uniform(0.1, 0.9)) * H)
            if random_wind and sc in WIND:
                opts["wind_deg"] = float(rng.uniform(0, 360))
            after, ma, info = P.generate(img, mask, sc, sev, gsd, rng, buildings=blds, opts=opts)
            change = P.change_map(img, after, mask, ma)
            fired = []
            if sensor:
                after, fired = AUG.apply_post_event(after, sc, int(rng.integers(1 << 30)))
            xbd = C.rescuenet_to_xbd(ma)
            name = f"{v}_{sc}_{i}"
            meta = dict(village=v, severity=round(sev, 3), gsd_m=round(gsd, 3), seed=int(seed),
                        point_px=[round(c, 1) for c in opts["point"]] if "point" in opts else None,
                        post_event_albumentations=fired, **info["extra"])
            z.writestr(f"after/{name}.jpg", enc(after, ".jpg"))
            z.writestr(f"masks/{name}_lab.png", enc(ma, ".png"))
            z.writestr(f"xbd/{name}_damage.png", enc(xbd, ".png"))
            z.writestr(f"change/{name}_change.png", enc(change.astype(np.uint8) * 255, ".png"))
            z.writestr(f"hazard/{name}_hazard.png", enc((info["hazard"] * 255).astype(np.uint8), ".png"))
            z.writestr(f"labels_xbd/{name}.json", json.dumps(G.xbd_json(name, info, info["b_lab"], sc, meta), indent=1))
            prev = G.preview(img, after, info["hazard"], mask, ma, xbd, change, info, panel=420)
            z.writestr(f"previews/{name}_preview.jpg", enc(prev, ".jpg", 85))
            cs = int(min(crop_size, H, W))
            for j, c in enumerate(AUG.paired_crops(rgb(img), rgb(after), mask, ma, xbd, change, int(n_crops), cs,
                                                   int(rng.integers(1 << 30)))):
                cn = f"{name}_aug{j}"
                z.writestr(f"augmented/before/{cn}.jpg", enc(cv2.cvtColor(c["before"], cv2.COLOR_RGB2BGR), ".jpg", 90))
                z.writestr(f"augmented/after/{cn}.jpg", enc(cv2.cvtColor(c["after"], cv2.COLOR_RGB2BGR), ".jpg", 90))
                z.writestr(f"augmented/masks_before/{cn}_lab.png", enc(c["mask_before"], ".png"))
                z.writestr(f"augmented/masks/{cn}_lab.png", enc(c["mask_after"], ".png"))
                z.writestr(f"augmented/xbd/{cn}_damage.png", enc(c["xbd"], ".png"))
                z.writestr(f"augmented/change/{cn}_change.png", enc(c["change"] * 255, ".png"))
            counts = np.bincount([b["damage_level"] for b in info["buildings"]], minlength=4)
            totals += counts
            rows.append(dict(name=name, village=v, scenario=sc, severity=round(sev, 3),
                             buildings=len(info["buildings"]), **{C.DAMAGE_NAME[l]: int(counts[l]) for l in range(4)},
                             changed_pct=round(100 * float(change.mean()), 2), albumentations=" ".join(fired)))
            if len(thumbs) < 12:
                thumbs.append((rgb(after), f"{v} · {LABELS[sc]}"))
        head = list(rows[0])
        z.writestr("manifest.csv", "\n".join([",".join(head)] + [",".join(str(r[h]) for h in head) for r in rows]))
        z.writestr("summary.json", json.dumps(dict(
            samples=len(rows), crops_per_sample=int(n_crops), crop_size=int(crop_size), resolution_px=side,
            severity_range=[sev_lo, sev_hi], random_point=bool(random_point), random_wind=bool(random_wind),
            villages=list(vills), scenarios=list(scenarios), seed=int(seed),
            buildings_per_damage_level=dict(zip(C.DAMAGE_NAME, map(int, totals))),
            label_scheme="RescueNet Kaggle v1.0", classes=C.RN_NAMES, xbd_classes=C.XBD_NAMES), indent=1))
    size_mb = os.path.getsize(path) / 1e6
    n_files = len(rows) * (7 + 6 * int(n_crops)) + 2 * len(vills) + 2
    tiles = "".join(f"<div class='tile'><div class='bar' style='background:{DMG_HEX[i]}'></div>"
                    f"<div class='num'>{int(totals[i])}</div><div class='lbl'>{DMG_FR[i]}</div></div>" for i in range(4))
    kv = "".join(f"<div class='k'>{k}</div><div class='v'>{v}</div>" for k, v in [
        ("Images après", f"{len(rows)}"), ("Crops", f"{len(rows) * int(n_crops)}"),
        ("Fichiers", f"{n_files}"), ("Taille du zip", f"{size_mb:.1f} Mo"),
        ("Résolution", f"{side} px"), ("Durée", f"{time.time() - t0:.0f} s")])
    html = (f"<div class='stats'><div class='stats-head'><span class='eyebrow'>Lot prêt</span>"
            f"<span class='title'>{len(rows)} scènes</span><span class='pill'>bâtiments par niveau</span></div>"
            f"<div class='tiles'>{tiles}</div><div class='kv'>{kv}</div></div>")
    return path, html, thumbs


# ---------------------------------------------------------------------------------------------
# design : console de cartographie de terrain
#   fond noir graphite, courbes de niveau animées + grille de points (canvas),
#   un seul accent jaune signalisation, typo condensée de signalétique + Plex
# ---------------------------------------------------------------------------------------------
INK = "#0b0b0a"
PANEL = "#131312"
LINE = "#2a2925"
PAPER = "#ece8dd"
DIM = "#8f8a7e"
SIGNAL = "#ffd400"

_DARK = dict(
    body_background_fill=INK,
    body_text_color=PAPER,
    body_text_color_subdued=DIM,
    background_fill_primary=PANEL,
    background_fill_secondary="#191917",
    border_color_primary=LINE,
    border_color_accent=SIGNAL,
    block_background_fill=PANEL,
    block_border_color=LINE,
    block_label_background_fill="transparent",
    block_label_text_color=DIM,
    block_title_text_color=PAPER,
    input_background_fill="#0e0e0d",
    input_border_color=LINE,
    input_border_color_focus=SIGNAL,
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_hover="#1c1c1a",
    button_secondary_border_color=LINE,
    button_secondary_text_color=PAPER,
    button_primary_background_fill=SIGNAL,
    button_primary_background_fill_hover="#ffe14d",
    button_primary_text_color=INK,
    button_primary_border_color=SIGNAL,
    color_accent=SIGNAL,
    color_accent_soft="rgba(255,212,0,0.12)",
    slider_color=SIGNAL,
    checkbox_background_color="#0e0e0d",
    checkbox_background_color_selected=SIGNAL,
    checkbox_border_color="#45433d",
    checkbox_border_color_selected=SIGNAL,
    checkbox_label_background_fill="transparent",
    checkbox_label_background_fill_selected="transparent",
    checkbox_label_text_color=PAPER,
    panel_background_fill=PANEL,
    table_even_background_fill="transparent",
    table_odd_background_fill="transparent",
)
_SET = inspect.signature(gr.themes.Base.set).parameters
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.yellow,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), gr.themes.GoogleFont("Big Shoulders Display"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    radius_size=gr.themes.sizes.radius_none,
).set(block_border_width="1px", block_shadow="none", block_radius="2px", button_large_radius="2px",
      button_small_radius="2px", input_radius="2px", block_label_text_weight="400",
      block_title_text_weight="500",
      **_DARK, **{k + "_dark": v for k, v in _DARK.items() if k + "_dark" in _SET})

# Démarrage côté navigateur : mode sombre + fond animé (courbes de niveau sur un champ de bruit
# qui évolue lentement, grille de points qui s'éclairent près de la souris).
BOOT_JS = r"""
() => {
  document.body.classList.add('dark');
  if (document.getElementById('terrain')) return;
  const cv = document.createElement('canvas');
  cv.id = 'terrain';
  cv.setAttribute('aria-hidden', 'true');
  cv.style.cssText = 'position:fixed;left:0;top:0;z-index:-1;pointer-events:none;display:block;';
  document.body.appendChild(cv);
  const ctx = cv.getContext('2d');
  const S = 28;
  let W = 0, H = 0;
  function resize() {
    const d = Math.min(window.devicePixelRatio || 1, 2);
    W = innerWidth; H = innerHeight;
    cv.width = W * d; cv.height = H * d;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    ctx.setTransform(d, 0, 0, d, 0, 0);
  }
  resize();
  addEventListener('resize', resize);
  let mx = -1e4, my = -1e4;
  addEventListener('pointermove', e => { mx = e.clientX; my = e.clientY; });

  const P = new Uint8Array(512);
  for (let i = 0; i < 256; i++) P[i] = i;
  for (let i = 255; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [P[i], P[j]] = [P[j], P[i]]; }
  for (let i = 0; i < 256; i++) P[i + 256] = P[i];
  const sm = t => t * t * (3 - 2 * t);
  const hs = (x, y, z) => P[(P[(P[x & 255] + y) & 255] + z) & 255] / 255;
  const lerp = (a, b, t) => a + (b - a) * t;
  function noise(x, y, z) {
    const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
    const u = sm(x - xi), v = sm(y - yi), w = sm(z - zi);
    return lerp(
      lerp(lerp(hs(xi, yi, zi), hs(xi + 1, yi, zi), u), lerp(hs(xi, yi + 1, zi), hs(xi + 1, yi + 1, zi), u), v),
      lerp(lerp(hs(xi, yi, zi + 1), hs(xi + 1, yi, zi + 1), u), lerp(hs(xi, yi + 1, zi + 1), hs(xi + 1, yi + 1, zi + 1), u), v),
      w);
  }
  const field = (x, y, t) => 0.68 * noise(x * 0.0028, y * 0.0028, t) + 0.32 * noise(x * 0.007 + 37, y * 0.007, t * 1.6);
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const LEVELS = [0.32, 0.38, 0.44, 0.50, 0.56, 0.62, 0.68];
  let t = Math.random() * 10, last = 0;

  function frame(now) {
    requestAnimationFrame(frame);
    if (now - last < 60) return;
    last = now;
    if (!still) t += 0.0035;
    ctx.clearRect(0, 0, W, H);
    const cols = Math.ceil(W / S) + 1, rows = Math.ceil(H / S) + 1;
    const v = new Float32Array(cols * rows);
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) v[j * cols + i] = field(i * S, j * S, t);

    ctx.lineWidth = 1;
    for (const L of LEVELS) {
      ctx.strokeStyle = L === 0.50 ? 'rgba(255,212,0,0.22)' : 'rgba(236,232,221,0.075)';
      ctx.beginPath();
      for (let j = 0; j < rows - 1; j++) for (let i = 0; i < cols - 1; i++) {
        const a = v[j * cols + i], b = v[j * cols + i + 1], c = v[(j + 1) * cols + i + 1], d = v[(j + 1) * cols + i];
        const k = (a > L ? 8 : 0) | (b > L ? 4 : 0) | (c > L ? 2 : 0) | (d > L ? 1 : 0);
        if (k === 0 || k === 15) continue;
        const x = i * S, y = j * S, f = (p, q) => (L - p) / (q - p);
        const T = [x + S * f(a, b), y], R = [x + S, y + S * f(b, c)], B = [x + S * f(d, c), y + S], Lf = [x, y + S * f(a, d)];
        const seg = (p, q) => { ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); };
        switch (k) {
          case 1: case 14: seg(Lf, B); break;
          case 2: case 13: seg(B, R); break;
          case 3: case 12: seg(Lf, R); break;
          case 4: case 11: seg(T, R); break;
          case 5: seg(Lf, T); seg(B, R); break;
          case 6: case 9: seg(T, B); break;
          case 7: case 8: seg(Lf, T); break;
          case 10: seg(Lf, B); seg(T, R); break;
        }
      }
      ctx.stroke();
    }

    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      const x = i * S, y = j * S, val = v[j * cols + i];
      const near = Math.max(0, 1 - Math.hypot(x - mx, y - my) / 170);
      const r = 0.7 + 0.6 * val + 1.5 * near;
      ctx.fillStyle = near > 0.04 ? `rgba(255,212,0,${0.25 + 0.65 * near})` : `rgba(236,232,221,${0.08 + 0.16 * val})`;
      ctx.fillRect(x - r / 2, y - r / 2, r, r);
    }
  }
  requestAnimationFrame(frame);
}
"""

CSS = """
:root, .dark {color-scheme: dark;}
body {background:#0b0b0a !important;}
gradio-app, .gradio-container, .main, .wrap.app, .app {background: transparent !important;}
#terrain {position:fixed; inset:0; z-index:-1; pointer-events:none;}
gradio-app {position:relative; z-index:1;}
body::after {content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
  background: radial-gradient(ellipse at 50% 40%, transparent 40%, rgba(0,0,0,.55) 100%);}

.gradio-container {max-width: 1440px !important; margin: 0 auto !important; padding: 0 28px !important;
                   font-family: 'IBM Plex Sans', sans-serif !important;}
footer {display:none !important;}

/* ---------- bandeau ruban de danger ---------- */
.tape {height:8px; margin: 0 -28px; background: repeating-linear-gradient(-45deg, #ffd400 0 14px, #0b0b0a 14px 28px);}

/* ---------- en-tête ---------- */
.hero {display:grid; grid-template-columns: 1fr auto; gap: 8px 40px; align-items:end;
       padding: 34px 0 22px; border-bottom: 1px solid #2a2925; margin-bottom: 18px;}
.hero h1 {grid-column:1; font-family:'Big Shoulders Display', sans-serif; font-weight:800; font-size:76px;
          line-height:.86; letter-spacing:.005em; text-transform:uppercase; margin:0; color:#ece8dd;}
.hero h1 em {font-style:normal; color:#ffd400;}
.hero p {grid-column:1; max-width:660px; margin:14px 0 0; color:#8f8a7e; font-size:15px; line-height:1.55;}
.meta {grid-column:2; grid-row:1 / span 2; align-self:end; font-family:'IBM Plex Mono', monospace; font-size:11.5px;
       line-height:1.9; color:#8f8a7e; text-align:right; border-left:1px solid #2a2925; padding-left:22px;}
.meta b {color:#ece8dd; font-weight:500;}
.coord {font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:.08em; color:#ffd400;
        text-transform:uppercase;}

/* ---------- onglets ---------- */
button[role=tab] {font-family:'IBM Plex Mono', monospace !important; text-transform:uppercase; letter-spacing:.1em;
                  font-size:12px !important;}
button[role=tab][aria-selected=true] {color:#ffd400 !important;}

/* ---------- panneau ---------- */
.gradio-container {counter-reset: sec;}
.section {counter-increment: sec; display:flex; align-items:center; gap:10px; margin:18px 0 6px;
          font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:.14em; text-transform:uppercase;
          color:#8f8a7e;}
.section::before {content: counter(sec, decimal-leading-zero); color:#ffd400;}
.section::after {content:""; flex:1; border-top:1px dashed #3a3833;}
.hint {font-size:13px; color:#8f8a7e; line-height:1.5; padding:2px 0;}
.hint b {color:#ece8dd; font-weight:500;}

#scenario .wrap, .quality .wrap {display:grid !important; gap:0; border:1px solid #2a2925;}
#scenario .wrap {grid-template-columns:repeat(3,1fr);}
.quality .wrap {grid-template-columns:1fr 1fr;}
.render3 .wrap {grid-template-columns:repeat(3,1fr);}
#scenario label, .quality label {justify-content:flex-start; padding:12px 12px !important; margin:0 !important;
  border-radius:0 !important; border:0 !important; border-right:1px solid #2a2925 !important;
  border-bottom:1px solid #2a2925 !important; background:#131312 !important; font-size:14px; font-weight:500;
  cursor:pointer; transition: background .15s, color .15s;}
#scenario label input, .quality label input {display:none;}
#scenario label:hover, .quality label:hover {background:#1c1c1a !important;}
#scenario label.selected, .quality label.selected {background:#ffd400 !important; color:#0b0b0a !important;}

button {letter-spacing:.02em;}
button.primary {font-family:'IBM Plex Mono', monospace !important; text-transform:uppercase; letter-spacing:.14em;
                font-weight:600 !important;}
button.secondary:hover {border-color:#ffd400 !important; color:#ffd400 !important;}

/* ---------- visionneuse : repères de cadrage ---------- */
#viewer {position:relative;}
#viewer::before, #viewer::after {content:""; position:absolute; width:22px; height:22px; z-index:5; pointer-events:none;}
#viewer::before {left:-1px; top:-1px; border-left:2px solid #ffd400; border-top:2px solid #ffd400;}
#viewer::after {right:-1px; bottom:-1px; border-right:2px solid #ffd400; border-bottom:2px solid #ffd400;}
#picker img {cursor:crosshair;}

/* ---------- résultats ---------- */
.stats {border:1px solid #2a2925; background:#131312; padding:0;}
.stats.empty {color:#8f8a7e; padding:40px; text-align:center; font-family:'IBM Plex Mono', monospace; font-size:12px;}
.stats-head {display:flex; align-items:baseline; gap:14px; padding:14px 18px; border-bottom:1px solid #2a2925;}
.stats-head .title {font-family:'Big Shoulders Display', sans-serif; font-size:30px; font-weight:800;
                    text-transform:uppercase; letter-spacing:.01em; line-height:1;}
.eyebrow {font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:#8f8a7e;}
.pill {font-family:'IBM Plex Mono', monospace; font-size:12px; color:#ffd400; margin-left:auto;}
.tiles {display:grid; grid-template-columns:repeat(4,1fr); border-bottom:1px solid #2a2925;}
.tile {padding:14px 18px 12px; border-right:1px solid #2a2925; position:relative;}
.tile:last-child {border-right:0;}
.tile .bar {position:absolute; left:18px; top:16px; width:8px; height:8px;}
.tile .num {font-family:'Big Shoulders Display', sans-serif; font-size:52px; font-weight:800; line-height:.95;
            font-variant-numeric:tabular-nums; margin-top:14px;}
.tile .lbl {font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:.12em; text-transform:uppercase;
            color:#8f8a7e; position:absolute; left:34px; top:13px;}
.kv {display:grid; grid-template-columns: repeat(2, auto 1fr); gap:6px 18px; padding:14px 18px;
     font-size:13px; border-bottom:1px solid #2a2925;}
.kv .k, .aug .k {font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:.08em; text-transform:uppercase;
                 color:#8f8a7e; padding-top:2px;}
.kv .v {font-family:'IBM Plex Mono', monospace; color:#ece8dd;}
.aug {padding:12px 18px 10px; display:flex; flex-direction:column; gap:8px;}
.code {display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:12px; padding:2px 7px; margin:0 6px 6px 0;
       border:1px solid #3a3833; color:#ece8dd;}
.muted {color:#8f8a7e;}

.legend {display:flex; flex-wrap:wrap; gap:6px 18px; padding:10px 2px 2px; font-family:'IBM Plex Mono', monospace;
         font-size:11.5px; color:#8f8a7e;}
.legend span {display:inline-flex; align-items:center; gap:7px;}
.legend i {width:9px; height:9px; display:inline-block;}

.explain {font-size:15px; line-height:1.7; color:#b5b0a4; max-width:820px; padding:6px 0;}
.explain b {color:#ece8dd; font-weight:600;}
.explain p {margin:0 0 14px;}
.doc h3 {font-family:'IBM Plex Mono', monospace; font-size:13px; letter-spacing:.12em; text-transform:uppercase;
         color:#ffd400; margin:26px 0 10px; font-weight:500;}
.doc code {font-family:'IBM Plex Mono', monospace; font-size:13px; color:#ece8dd; background:#1c1c1a;
           padding:1px 5px; border:1px solid #2a2925;}
.doc table {border-collapse:collapse; margin:0 0 16px; font-size:14px; width:100%;}
.doc th, .doc td {border:1px solid #2a2925; padding:7px 10px; text-align:left; vertical-align:top;}
.doc th {background:#1c1c1a; color:#ece8dd; font-weight:500; font-family:'IBM Plex Mono', monospace; font-size:12px;
         letter-spacing:.06em;}
.doc ul, .doc ol {margin:0 0 14px; padding-left:22px;}
.doc li {margin:0 0 8px;}
.doc li ul {margin-top:8px;}
.doc-tabs button[role=tab] {font-size:11px !important; letter-spacing:.06em;}
.doc pre {font-family:'IBM Plex Mono', monospace; font-size:12.5px; line-height:1.6; color:#ece8dd;
          background:#131312; border:1px solid #2a2925; padding:14px 16px; overflow-x:auto; margin:0 0 14px;}

/* ---------- une seule chorégraphie : apparition échelonnée au chargement ---------- */
@keyframes enter {from {opacity:0; transform: translateY(8px);} to {opacity:1; transform:none;}}
.hero, .tape {animation: enter .6s ease-out both;}
.gradio-container .column > * {animation: enter .5s ease-out backwards;}
/* the scene list must open above the next blocks of the column */
#village {position:relative; z-index:50;}
.gradio-container .column > *:nth-child(2) {animation-delay:.04s;}
.gradio-container .column > *:nth-child(3) {animation-delay:.08s;}
.gradio-container .column > *:nth-child(4) {animation-delay:.12s;}
.gradio-container .column > *:nth-child(5) {animation-delay:.16s;}
.gradio-container .column > *:nth-child(6) {animation-delay:.20s;}
.gradio-container .column > *:nth-child(n+7) {animation-delay:.24s;}
.tile {animation: enter .35s ease-out backwards;}
.tile:nth-child(2) {animation-delay:.05s;} .tile:nth-child(3) {animation-delay:.10s;} .tile:nth-child(4) {animation-delay:.15s;}

::selection {background:#ffd400; color:#0b0b0a;}
@media (max-width: 900px) {
  .hero {grid-template-columns:1fr;} .meta {grid-column:1; grid-row:auto; text-align:left; border:0; padding:0;}
  .hero h1 {font-size:52px;} .tiles {grid-template-columns:repeat(2,1fr);} .kv {grid-template-columns:auto 1fr;}
  #scenario .wrap {grid-template-columns:repeat(2,1fr);}
}
@media (prefers-reduced-motion: reduce) {*, *::before, *::after {animation:none !important; transition:none !important;}}
"""



def section(txt):
    return gr.HTML(f"<div class='section'>{txt}</div>")


def build():
    vlist = villages()
    with gr.Blocks(title="Village vers post-catastrophe") as demo:
        gr.HTML(HEADER)
        state = gr.State(None)
        point = gr.State(None)

        with gr.Tab("Simulation"):
            with gr.Row(equal_height=False):
                # ---------------- controls ----------------
                with gr.Column(scale=4, min_width=340):
                    section("Scène")
                    village = gr.Dropdown(vlist, value=vlist[0][1], show_label=False, container=False,
                                          filterable=False, elem_id="village")

                    section("Catastrophe")
                    scenario = gr.Radio([(v, k) for k, v in LABELS.items()], value="tornado",
                                        show_label=False, container=False, elem_id="scenario")

                    section("Paramètres")
                    with gr.Group():
                        severity = gr.Slider(0.05, 1.0, value=0.75, step=0.05, label="Sévérité")
                        seed = gr.Number(value=7, precision=0, label="Graine")
                    dice = gr.Button("Tirer une nouvelle graine", size="sm", variant="secondary")
                    with gr.Group(visible=True) as wind_box:
                        wind_auto = gr.Checkbox(value=True, label="Direction du vent aléatoire")
                        wind = gr.Slider(0, 359, value=45, step=1, interactive=False,
                                         label="Direction du vent (degrés, 0 = vers l'est, 90 = vers le bas)")
                    coverage = gr.Slider(0.05, 0.8, value=0.35, step=0.05, label="Surface inondée", visible=False)

                    section("Point d'impact")
                    point_md = gr.HTML(f"<div class='hint'>Cliquez sur l'image pour placer <b>{POINT_HELP['tornado']}</b>.</div>")
                    picker = gr.Image(show_label=False, interactive=False, height=290, elem_id="picker",
                                      buttons=[])
                    clear_pt = gr.Button("Effacer le point", size="sm", variant="secondary")

                    section("Rendu")
                    quality = gr.Radio([("Rapide · 512 px", "fast"), ("Complet · 1000 px", "full")], value="fast",
                                       show_label=False, container=False, elem_classes=["quality"])
                    render = gr.Radio([("Auto", "auto"), ("Aérien · 2D", "aerial"), ("Drone · relief", "drone")],
                                      value="auto", show_label=False, container=False,
                                      elem_classes=["quality", "render3"])
                    gr.HTML("<div class='hint'><b>Auto</b> : rendu drone (gravats en relief, éclairage et ombres "
                            "estimés sur la photo) sous 12 cm/px, rendu aérien au-dessus.</div>")
                    with gr.Group():
                        sensor = gr.Checkbox(value=True, label="Aspect post-événement (Albumentations)")
                        live = gr.Checkbox(value=True, label="Mise à jour automatique")
                    go = gr.Button("Générer", variant="primary", size="lg")

                    with gr.Accordion("Utiliser ma propre image", open=False):
                        gr.HTML("<div class='hint'>Une image aérienne (environ 0,3 m/px) et son masque "
                                "OpenEarthMap en PNG : 1 sol nu, 2 prairie, 3 aménagé, 4 route, 5 arbre, 6 eau, "
                                "7 agriculture, 8 bâtiment.</div>")
                        up_img = gr.Image(label="Image", type="numpy")
                        up_mask = gr.Image(label="Masque OpenEarthMap", type="numpy", image_mode="L")

                # ---------------- results ----------------
                with gr.Column(scale=8):
                    slider = gr.ImageSlider(label="Avant / après · faites glisser le curseur", height=640,
                                            elem_id="viewer")
                    stats = gr.HTML(PLACEHOLDER)
                    with gr.Tabs():
                        with gr.Tab("Dégâts"):
                            d_img = gr.Image(show_label=False, height=520)
                            gr.HTML(legend(["Intact", "Mineur", "Majeur", "Détruit"], C.XBD_PALETTE[1:],
                                           [("Autre changement", (90, 90, 90))]))
                        with gr.Tab("Masque RescueNet"):
                            m_img = gr.Image(show_label=False, height=520)
                            gr.HTML(legend(C.RN_NAMES, C.RN_PALETTE))
                        with gr.Tab("Carte d'aléa"):
                            h_img = gr.Image(show_label=False, height=520)
                            gr.HTML("<div class='hint'>Plus c'est clair, plus la catastrophe frappe fort. "
                                    "Le niveau de dégât de chaque maison dépend de cette carte et de sa vulnérabilité.</div>")
                        with gr.Tab("Changements"):
                            c_img = gr.Image(show_label=False, height=520)
                            gr.HTML("<div class='hint'>En blanc, les pixels modifiés par la catastrophe.</div>")
                    with gr.Row():
                        dl = gr.Button("Télécharger le résultat (.zip)", variant="secondary")
                        dl_file = gr.File(show_label=False, height=52)

        with gr.Tab("Albumentations"):
            gr.HTML("""<div class='explain'><b>Albumentations</b> transforme des images existantes de façon aléatoire
              pour enrichir un jeu d'entraînement. Il ne dessine pas les dégâts : il intervient après.
              Il sert ici à deux choses.</div>""")
            section("Aspect post-événement")
            gr.HTML("<div class='hint'>L'image après est prise par un autre capteur, un autre jour : tons, bruit, "
                    "compression, ombres de nuages. Seuls les pixels changent, les masques restent exacts.</div>")
            with gr.Row():
                pe_seed = gr.Number(value=1, precision=0, label="Graine", scale=1)
                pe_btn = gr.Button("Nouveau tirage", variant="primary", scale=1)
            pe_slider = gr.ImageSlider(label="Rendu brut / avec Albumentations", height=560)
            pe_md = gr.HTML()
            section("Crops alignés pour l'entraînement")
            gr.HTML("<div class='hint'>Chaque crop est un même découpage montré trois fois : avant, après, dégâts. "
                    "La géométrie (symétries, rotation, recadrage sur les zones touchées) est identique pour les trois ; "
                    "les couleurs de l'avant et de l'après sont tirées séparément.</div>")
            with gr.Row():
                n_crops = gr.Slider(1, 8, value=4, step=1, label="Nombre de crops")
                crop_size = gr.Slider(128, 512, value=256, step=64, label="Taille (px)")
                crop_seed = gr.Number(value=3, precision=0, label="Graine")
            crop_btn = gr.Button("Générer les crops", variant="primary")
            gallery = gr.Gallery(columns=6, height="auto", object_fit="contain", show_label=False)

        with gr.Tab("Export en lot"):
            gr.HTML("""<div class='explain'>Les résultats vous conviennent ? Générez <b>tout un jeu de données</b>
              d'un coup : chaque village choisi passe par chaque catastrophe choisie, plusieurs fois, avec
              le point d'impact, la direction du vent et la sévérité tirés au hasard à chaque version.
              Chaque scène est livrée avec ses masques, ses labels xBD et ses crops Albumentations.</div>""")
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=340):
                    section("Villages et catastrophes")
                    b_vills = gr.CheckboxGroup(vlist, value=[v for _, v in vlist], label="Villages")
                    b_scen = gr.CheckboxGroup([(v, k) for k, v in LABELS.items()], value=list(LABELS),
                                              label="Catastrophes")
                    section("Variations")
                    with gr.Group():
                        b_n = gr.Slider(1, 10, value=2, step=1, label="Versions par village et par catastrophe")
                        with gr.Row():
                            b_smin = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label="Sévérité min")
                            b_smax = gr.Slider(0.05, 1.0, value=0.95, step=0.05, label="Sévérité max")
                        b_point = gr.Checkbox(value=True, label="Point d'impact à un endroit différent à chaque version")
                        b_wind = gr.Checkbox(value=True, label="Direction du vent aléatoire")
                    section("Crops et rendu")
                    with gr.Group():
                        b_crops = gr.Slider(0, 16, value=8, step=1, label="Crops Albumentations par scène")
                        b_csize = gr.Slider(128, 512, value=512, step=64, label="Taille des crops (px)")
                        b_sensor = gr.Checkbox(value=True, label="Aspect post-événement (Albumentations)")
                        b_seed = gr.Number(value=2026, precision=0, label="Graine du lot")
                    b_quality = gr.Radio([("Rapide · 512 px", "fast"), ("Complet · 1000 px", "full")], value="full",
                                         show_label=False, container=False, elem_classes=["quality"])
                    b_count = gr.HTML()
                    b_go = gr.Button("Générer le lot et créer le zip", variant="primary", size="lg")
                with gr.Column(scale=8):
                    b_stats = gr.HTML("<div class='stats empty'>Aucun lot généré pour l'instant</div>")
                    b_file = gr.File(label="Archive .zip", height=80)
                    b_gallery = gr.Gallery(columns=4, height="auto", object_fit="contain", show_label=False)
                    gr.HTML("""<div class='hint'>Contenu du zip : <b>before/</b>, <b>after/</b>, <b>masks_before/</b>,
                      <b>masks/</b> (RescueNet), <b>xbd/</b> et <b>labels_xbd/</b> (dégâts), <b>change/</b>,
                      <b>hazard/</b>, <b>previews/</b>, <b>augmented/</b> (crops alignés), <b>manifest.csv</b>,
                      <b>summary.json</b>.</div>""")

            def estimate(v, s, n, cr, q):
                k = len(v or []) * len(s or []) * int(n)
                sec = k * (0.7 if q == "fast" else 3.5) + k * int(cr) * 0.05
                return (f"<div class='hint'><b>{k}</b> scènes, <b>{k * int(cr)}</b> crops — "
                        f"environ <b>{max(1, round(sec / 60))} min</b></div>")

            for comp in (b_vills, b_scen, b_n, b_crops, b_quality):
                comp.change(estimate, [b_vills, b_scen, b_n, b_crops, b_quality], b_count)
            demo.load(estimate, [b_vills, b_scen, b_n, b_crops, b_quality], b_count)
            b_go.click(batch_export, [b_vills, b_scen, b_n, b_smin, b_smax, b_point, b_wind, b_quality,
                                      b_crops, b_csize, b_sensor, b_seed], [b_file, b_stats, b_gallery])

        with gr.Tab("À propos"):
            gr.HTML("""<div class='explain'>
              <p><b>Entrée.</b> Une photo aérienne et son masque de segmentation, annoté à la main dans
              OpenEarthMap : chaque pixel porte un numéro de classe (bâtiment, route, arbre, eau...).
              On sait donc déjà où sont les maisons.</p>
              <p><b>Simulation.</b> Une carte d'aléa physique décide où la catastrophe frappe (distance à l'épicentre,
              rafales de vent, couloir de tornade, profondeur d'eau, front de feu). Chaque maison reçoit un niveau de
              dégât, que le code dessine sur la photo tout en écrivant l'étiquette correspondante.</p>
              <p><b>Sortie.</b> L'image après, son masque RescueNet, la carte des dégâts au format xBD, la carte de
              changements et la carte d'aléa. Les étiquettes sont exactes par construction : c'est l'intérêt des
              données synthétiques pour entraîner un modèle de détection des dégâts.</p>
              <p><b>Images drone.</b> Les scènes de Graz (Semantic Drone Dataset, environ 2 cm/px) utilisent un
              rendu à part : les débris ont une hauteur, sont éclairés par le soleil estimé sur la photo et projettent
              leur ombre. Le choix est automatique (option Rendu).</p>
              <p><b>Limites.</b> Les images sont synthétiques : un modèle doit être évalué sur de vraies images
              post-catastrophe. Le relief de l'inondation est simulé, pas mesuré.</p>
            </div>""")

        with gr.Tab("Documentation"):
            with gr.Tabs(elem_classes=["doc-tabs"]):
                for title, html in site_doc.PAGES:
                    with gr.Tab(title):
                        gr.HTML(html)

        # ---- events ---------------------------------------------------------------------------
        inputs = [village, up_img, up_mask, scenario, severity, seed, wind_auto, wind, coverage, point, quality, sensor,
                  render]
        outputs = [slider, m_img, d_img, h_img, c_img, stats, state]

        def refresh_picker(v, ui, sc, pt, wa, wd):
            img = cv2.cvtColor(ui, cv2.COLOR_RGB2BGR) if ui is not None else load_village(v, 512)[0]
            return mark(img, pt, sc, None if wa else wd)

        def on_scenario(sc):
            txt = (f"<div class='hint'>Cliquez sur l'image pour placer <b>{POINT_HELP[sc]}</b>.</div>"
                   if sc in POINT_HELP else "<div class='hint'>Pas de point à placer pour ce scénario.</div>")
            return gr.update(visible=sc == "flood"), txt, gr.update(visible=sc in WIND)

        def on_click(evt: gr.SelectData, v, ui):
            img = ui if ui is not None else rgb(load_village(v, 512)[0])
            H, W = img.shape[:2]
            return (evt.index[0] / W, evt.index[1] / H)

        maybe_run = lambda *a: run(*a[:-1]) if a[-1] else [gr.skip()] * len(outputs)

        go.click(run, inputs, outputs)
        dice.click(lambda: int(np.random.randint(0, 100000)), None, seed)
        scenario.change(on_scenario, scenario, [coverage, point_md, wind_box])
        wind_auto.change(lambda a: gr.update(interactive=not a), wind_auto, wind)
        picker.select(on_click, [village, up_img], point)
        clear_pt.click(lambda: None, None, point)
        village.change(lambda: None, None, point)

        for comp in (village, scenario, point, wind_auto, wind, up_img):
            comp.change(refresh_picker, [village, up_img, scenario, point, wind_auto, wind], picker)
        for comp in (village, scenario, seed, wind_auto, quality, sensor, point, render):
            comp.change(maybe_run, inputs + [live], outputs)
        for comp in (severity, wind, coverage):
            comp.release(maybe_run, inputs + [live], outputs)

        dl.click(make_zip, state, dl_file)
        pe_btn.click(post_event_demo, [state, pe_seed], [pe_slider, pe_md])
        crop_btn.click(crops, [state, n_crops, crop_size, crop_seed], gallery)

        demo.load(refresh_picker, [village, up_img, scenario, point, wind_auto, wind], picker)
        demo.load(run, inputs, outputs)
    return demo


if __name__ == "__main__":
    build().launch(theme=THEME, css=CSS, js=BOOT_JS)

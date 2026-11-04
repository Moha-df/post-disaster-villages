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
from postdisaster import augment as AUG
from postdisaster import classes as C
from postdisaster import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
VILLAGE_DIR = os.path.join(HERE, "data", "villages")
GSD = 0.3
LABELS = {"earthquake": "Séisme", "hurricane": "Ouragan", "tornado": "Tornade",
          "flood": "Inondation", "wildfire": "Feu de forêt"}
POINT_HELP = {"earthquake": "l'épicentre", "tornado": "le centre du couloir de la tornade",
              "wildfire": "le point de départ du feu"}
WIND = ("hurricane", "tornado", "wildfire")
COUNTRIES = {"aachen": "Allemagne", "muenster": "Allemagne", "koeln": "Allemagne", "tyrolw": "Autriche",
             "baybay": "Philippines", "mahe": "Seychelles", "zanzibar": "Tanzanie", "austin": "États-Unis"}
DMG_HEX = ["#3cc83c", "#f5d800", "#ff8c00", "#dc1e1e"]
DMG_FR = ["Intact", "Mineur", "Majeur", "Détruit"]


def villages():
    d = os.path.join(VILLAGE_DIR, "images")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".png"))
    return [(f"{COUNTRIES.get(n.rsplit('_', 1)[0], '?')}  ·  {n}", n) for n in names]


@lru_cache(maxsize=32)
def load_village(name, side):
    """Image BGR + RescueNet mask + building instances, resized so the long side = `side`."""
    img = cv2.imread(os.path.join(VILLAGE_DIR, "images", name + ".png"), cv2.IMREAD_COLOR)
    raw = cv2.imread(os.path.join(VILLAGE_DIR, "masks", name + ".png"), cv2.IMREAD_GRAYSCALE)
    return prepare(img, raw, side)


def prepare(img, raw, side):
    s = side / max(img.shape[:2])
    if abs(s - 1) > 1e-3:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    raw = cv2.resize(raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = C.oem_to_rescuenet(raw)
    gsd = GSD / s
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
        cv2.circle(v, (x, y), r, (45, 212, 191), 3, cv2.LINE_AA)
        cv2.circle(v, (x, y), 3, (45, 212, 191), -1, cv2.LINE_AA)
    if wind_deg is not None and scenario in WIND:
        a = np.radians(wind_deg)
        c = np.array([W - 4 * r, 4 * r])
        d = np.array([np.cos(a), np.sin(a)]) * 3 * r
        p0, p1 = tuple((c - d).astype(int)), tuple((c + d).astype(int))
        cv2.arrowedLine(v, p0, p1, (255, 255, 255), 6, cv2.LINE_AA, tipLength=0.4)
        cv2.arrowedLine(v, p0, p1, (45, 212, 191), 3, cv2.LINE_AA, tipLength=0.4)
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
    rows.append(("Résolution", f"{W} × {H} px, {gsd:.2f} m/px"))
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
<div class='hero'>
  <div>
    <div class='eyebrow'>Données synthétiques pour l'évaluation des dégâts</div>
    <h1>Village <span>→</span> post-catastrophe</h1>
    <p>Une image aérienne d'un village intact devient la même scène après un séisme, un ouragan, une tornade,
       une inondation ou un feu. Les dégâts suivent une carte d'aléa physique et chaque pixel reste labellisé.</p>
  </div>
  <div class='badges'>
    <span>OpenEarthMap</span><span>RescueNet</span><span>xBD</span><span>Albumentations</span>
  </div>
</div>"""

PLACEHOLDER = "<div class='stats empty'>Le résultat s'affichera ici.</div>"


# ---------------------------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------------------------
def run(village, upload_img, upload_mask, scenario, severity, seed, wind_auto, wind_deg, coverage,
        point, quality, sensor):
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
# design : thème sombre, accent turquoise -> indigo, fond texturé animé
# ---------------------------------------------------------------------------------------------
ACCENT = "#2dd4bf"
ACCENT_2 = "#818cf8"

_DARK = dict(
    body_background_fill="#06080c",
    body_text_color="#e6edf3",
    body_text_color_subdued="#8b98a9",
    background_fill_primary="rgba(15,19,26,0.72)",
    background_fill_secondary="rgba(22,28,38,0.85)",
    border_color_primary="rgba(148,163,184,0.14)",
    block_background_fill="rgba(13,17,24,0.66)",
    block_border_color="rgba(148,163,184,0.13)",
    block_label_background_fill="transparent",
    block_label_text_color="#8b98a9",
    block_title_text_color="#e6edf3",
    input_background_fill="rgba(8,11,16,0.75)",
    input_border_color="rgba(148,163,184,0.16)",
    input_border_color_focus=ACCENT,
    button_secondary_background_fill="rgba(22,28,38,0.9)",
    button_secondary_background_fill_hover="rgba(32,40,54,0.95)",
    button_secondary_border_color="rgba(148,163,184,0.18)",
    button_secondary_text_color="#e6edf3",
    button_primary_background_fill=f"linear-gradient(120deg, {ACCENT}, {ACCENT_2})",
    button_primary_background_fill_hover=f"linear-gradient(120deg, {ACCENT_2}, {ACCENT})",
    button_primary_text_color="#04110f",
    color_accent=ACCENT,
    color_accent_soft="rgba(45,212,191,0.14)",
    slider_color=ACCENT,
    checkbox_background_color="rgba(8,11,16,0.75)",
    checkbox_background_color_selected=ACCENT,
    checkbox_border_color="rgba(148,163,184,0.3)",
    checkbox_border_color_selected=ACCENT,
    checkbox_label_background_fill="transparent",
    checkbox_label_background_fill_selected="transparent",
    checkbox_label_text_color="#e6edf3",
    panel_background_fill="rgba(13,17,24,0.66)",
    table_even_background_fill="transparent",
    table_odd_background_fill="transparent",
)
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Manrope"), gr.themes.GoogleFont("Space Grotesk"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(block_border_width="1px", block_shadow="0 10px 30px -12px rgba(0,0,0,.6)",
      block_label_text_weight="500", block_title_text_weight="600",
      **_DARK, **{k + "_dark": v for k, v in _DARK.items()
                  if k + "_dark" in inspect.signature(gr.themes.Base.set).parameters})

# force le mode sombre de Gradio, quel que soit le réglage du système
FORCE_DARK = """() => { document.body.classList.add('dark');
  document.querySelector('gradio-app')?.classList.add('dark'); }"""

# grain : bruit SVG (feTurbulence) en data URI
_NOISE = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'>"
          "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/>"
          "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .55 0'/></filter>"
          "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")")

CSS = """
:root, .dark {color-scheme: dark;}
body, gradio-app, .gradio-container, .main, .wrap.app {background: transparent !important;}
body {background-color: #06080c !important;}

/* ---------- fond : dégradé animé + grain + grille ---------- */
body::before {
  content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  background:
    radial-gradient(40% 35% at 15% 10%, rgba(45,212,191,.22), transparent 70%),
    radial-gradient(35% 35% at 90% 15%, rgba(129,140,248,.22), transparent 70%),
    radial-gradient(45% 40% at 60% 100%, rgba(56,189,248,.12), transparent 70%),
    radial-gradient(30% 30% at 0% 90%, rgba(168,85,247,.10), transparent 70%),
    linear-gradient(180deg, #06080c 0%, #0a0e15 55%, #06080c 100%);
  animation: drift 28s ease-in-out infinite alternate;
}
body::after {
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image: NOISE,
    linear-gradient(rgba(148,163,184,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,163,184,.045) 1px, transparent 1px);
  background-size: 220px 220px, 44px 44px, 44px 44px;
  opacity:.55; mix-blend-mode: overlay;
  -webkit-mask-image: radial-gradient(ellipse at 50% 0%, #000 30%, transparent 85%);
          mask-image: radial-gradient(ellipse at 50% 0%, #000 30%, transparent 85%);
}
gradio-app {position:relative; z-index:1;}
@keyframes drift {
  0%   {transform: translate3d(0,0,0) rotate(0deg) scale(1);}
  50%  {transform: translate3d(-3%,2%,0) rotate(4deg) scale(1.06);}
  100% {transform: translate3d(3%,-2%,0) rotate(-3deg) scale(1.02);}
}

.gradio-container {max-width: 1440px !important; margin: 0 auto !important; padding: 0 22px !important;
                   font-family: 'Manrope', system-ui, sans-serif !important;}
footer {display: none !important;}
h1, h2, h3, .title, .num, .section, .eyebrow, button[role=tab] {font-family: 'Space Grotesk', system-ui, sans-serif !important;}

/* ---------- verre ---------- */
.block, .form, .stats {backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%);}
.block {transition: border-color .3s, box-shadow .3s;}
.block:hover {border-color: rgba(45,212,191,.28) !important;}

/* ---------- apparition ---------- */
@keyframes rise {from {opacity:0; transform: translateY(14px) scale(.985); filter: blur(4px);}
                 to   {opacity:1; transform: none; filter: none;}}
.hero > * {animation: rise .8s cubic-bezier(.2,.7,.2,1) both;}
.hero > *:nth-child(2) {animation-delay: .15s;}
.gradio-container .column > * {animation: rise .7s cubic-bezier(.2,.7,.2,1) both;}
.gradio-container .column > *:nth-child(2) {animation-delay: .05s;}
.gradio-container .column > *:nth-child(3) {animation-delay: .10s;}
.gradio-container .column > *:nth-child(4) {animation-delay: .15s;}
.gradio-container .column > *:nth-child(5) {animation-delay: .20s;}
.gradio-container .column > *:nth-child(6) {animation-delay: .25s;}
.gradio-container .column > *:nth-child(7) {animation-delay: .30s;}
.gradio-container .column > *:nth-child(n+8) {animation-delay: .35s;}

/* ---------- en-tête ---------- */
.hero {display:flex; justify-content:space-between; align-items:flex-end; gap:24px; flex-wrap:wrap;
       padding:40px 4px 26px; margin-bottom:10px; position:relative;}
.hero::after {content:""; position:absolute; left:0; right:0; bottom:0; height:1px;
  background: linear-gradient(90deg, transparent, rgba(45,212,191,.6), rgba(129,140,248,.6), transparent);
  background-size: 200% 100%; animation: sweep 6s linear infinite;}
.hero h1 {font-size:42px; line-height:1.05; margin:8px 0 12px; font-weight:700; letter-spacing:-.035em;
  background: linear-gradient(100deg, #f8fafc 20%, #2dd4bf 45%, #818cf8 60%, #f8fafc 80%);
  background-size: 250% 100%; -webkit-background-clip:text; background-clip:text; color:transparent;
  animation: shine 9s ease-in-out infinite;}
.hero p {max-width:720px; margin:0; color:#8b98a9; font-size:15px; line-height:1.6;}
@keyframes shine {0%,100% {background-position: 100% 0;} 50% {background-position: 0% 0;}}
@keyframes sweep {from {background-position: 200% 0;} to {background-position: -200% 0;}}
.eyebrow {font-size:11px; letter-spacing:.18em; text-transform:uppercase; font-weight:600;
          color:#2dd4bf; display:inline-flex; align-items:center; gap:8px;}
.eyebrow::before {content:""; width:7px; height:7px; border-radius:50%; background:#2dd4bf;
                  animation: ping 2.2s ease-out infinite;}
.stats-head .eyebrow::before {display:none;}
@keyframes ping {0% {box-shadow:0 0 0 0 rgba(45,212,191,.6);} 100% {box-shadow:0 0 0 10px rgba(45,212,191,0);}}
.badges {display:flex; gap:8px; flex-wrap:wrap;}
.badges span {font-size:12px; padding:6px 12px; border-radius:999px; border:1px solid rgba(148,163,184,.18);
  background: rgba(13,17,24,.6); color:#b6c2d1; backdrop-filter: blur(8px);
  animation: float 6s ease-in-out infinite; transition: border-color .25s, color .25s, box-shadow .25s;}
.badges span:nth-child(2) {animation-delay:-1.5s;} .badges span:nth-child(3) {animation-delay:-3s;}
.badges span:nth-child(4) {animation-delay:-4.5s;}
.badges span:hover {border-color:#2dd4bf; color:#fff; box-shadow:0 0 18px -4px rgba(45,212,191,.6);}
@keyframes float {0%,100% {transform: translateY(0);} 50% {transform: translateY(-4px);}}

/* ---------- onglets ---------- */
button[role=tab] {transition: color .25s !important;}
button[role=tab][aria-selected=true] {color:#2dd4bf !important;}

/* ---------- panneau de contrôle ---------- */
.section {font-size:11px; letter-spacing:.16em; text-transform:uppercase; font-weight:600;
          color:#8b98a9; margin:16px 2px 6px; display:flex; align-items:center; gap:10px;}
.section::after {content:""; flex:1; height:1px; background: linear-gradient(90deg, rgba(148,163,184,.25), transparent);}
.hint {font-size:13px; color:#8b98a9; padding:2px 4px; line-height:1.5;}
.hint b {color:#e6edf3;}

#scenario .wrap, #quality .wrap {display:grid !important; gap:8px;}
#scenario .wrap {grid-template-columns:repeat(3,1fr);}
#quality .wrap {grid-template-columns:1fr 1fr;}
#scenario label, #quality label {justify-content:center; padding:11px 6px !important; border-radius:10px !important;
  font-weight:600; font-size:13.5px; cursor:pointer; position:relative; overflow:hidden;
  background: rgba(13,17,24,.7) !important; border:1px solid rgba(148,163,184,.16) !important;
  transition: transform .2s, border-color .25s, box-shadow .3s, background .3s !important;}
#scenario label input, #quality label input {display:none;}
#scenario label:hover, #quality label:hover {transform: translateY(-2px); border-color: rgba(45,212,191,.5) !important;}
#scenario label.selected, #quality label.selected {
  background: linear-gradient(120deg, rgba(45,212,191,.22), rgba(129,140,248,.22)) !important;
  border-color: #2dd4bf !important; color:#fff !important;
  box-shadow: 0 0 0 1px rgba(45,212,191,.35), 0 8px 26px -8px rgba(45,212,191,.55);}
#scenario label.selected::after {content:""; position:absolute; inset:0;
  background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.16) 50%, transparent 70%);
  background-size: 250% 100%; animation: sweep 2.6s ease-in-out infinite;}

/* ---------- boutons ---------- */
button.primary {position:relative; overflow:hidden; font-weight:700 !important; letter-spacing:.01em;
  transition: transform .2s, box-shadow .3s !important; box-shadow: 0 10px 30px -10px rgba(45,212,191,.6);}
button.primary:hover {transform: translateY(-2px); box-shadow: 0 14px 34px -10px rgba(129,140,248,.7);}
button.primary::after {content:""; position:absolute; top:0; left:-60%; width:40%; height:100%;
  background: linear-gradient(100deg, transparent, rgba(255,255,255,.45), transparent);
  transform: skewX(-20deg); animation: glint 3.8s ease-in-out infinite;}
@keyframes glint {0%,60% {left:-60%;} 100% {left:130%;}}
button.secondary {transition: transform .2s, border-color .2s !important;}
button.secondary:hover {transform: translateY(-1px); border-color: rgba(45,212,191,.5) !important;}

/* ---------- images ---------- */
#viewer {border-radius:16px !important; overflow:hidden;}
#viewer img, .image-frame img {animation: reveal .6s ease both;}
@keyframes reveal {from {opacity:.2; filter: blur(6px) saturate(.6);} to {opacity:1; filter:none;}}
#picker img {cursor:crosshair; transition: transform .4s;}
#picker:hover img {transform: scale(1.015);}

/* ---------- résultats ---------- */
.stats {padding:20px 20px 16px; border-radius:16px; position:relative;
  background: linear-gradient(rgba(13,17,24,.8), rgba(13,17,24,.8)) padding-box,
              linear-gradient(120deg, rgba(45,212,191,.55), rgba(129,140,248,.35), rgba(148,163,184,.08)) border-box;
  border:1px solid transparent; animation: rise .5s ease both;}
.stats.empty {color:#8b98a9; text-align:center; padding:44px;}
.stats-head {display:flex; align-items:baseline; gap:12px; margin-bottom:16px; flex-wrap:wrap;}
.stats-head .title {font-size:24px; font-weight:700; letter-spacing:-.02em;}
.pill {font-size:12px; padding:4px 10px; border-radius:999px; background:rgba(45,212,191,.12); color:#5eead4;
       font-weight:600; border:1px solid rgba(45,212,191,.3);}
.tiles {display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px;}
.tile {border:1px solid rgba(148,163,184,.14); border-radius:12px; padding:14px 14px 12px;
  position:relative; overflow:hidden; background: rgba(6,8,12,.55);
  animation: pop .55s cubic-bezier(.2,.9,.3,1.3) both; transition: transform .25s, border-color .25s;}
.tile:nth-child(2) {animation-delay:.06s;} .tile:nth-child(3) {animation-delay:.12s;} .tile:nth-child(4) {animation-delay:.18s;}
.tile:hover {transform: translateY(-3px); border-color: rgba(148,163,184,.35);}
.tile .bar {position:absolute; left:0; top:0; right:0; height:3px;}
.tile .num {font-size:32px; font-weight:700; font-variant-numeric:tabular-nums; margin-top:2px; letter-spacing:-.02em;}
.tile .lbl {font-size:11.5px; color:#8b98a9; text-transform:uppercase; letter-spacing:.12em;}
@keyframes pop {from {opacity:0; transform: translateY(10px) scale(.92);} to {opacity:1; transform:none;}}
.kv {display:grid; grid-template-columns:auto 1fr; gap:7px 22px; font-size:13.5px; padding:4px 2px 14px;
     border-bottom:1px solid rgba(148,163,184,.12);}
.kv .k, .aug .k {color:#8b98a9;}
.kv .v {font-weight:600; font-variant-numeric:tabular-nums; color:#e6edf3;}
.aug {padding-top:12px; font-size:13px; display:flex; flex-direction:column; gap:8px;}
.code {display:inline-block; font-family:'JetBrains Mono', monospace; font-size:12px; padding:4px 9px; margin:0 6px 6px 0;
  border-radius:7px; background:rgba(129,140,248,.1); border:1px solid rgba(129,140,248,.28); color:#c7d2fe;
  animation: pop .45s ease both;}
.muted {color:#8b98a9;}

.legend {display:flex; flex-wrap:wrap; gap:8px 16px; padding:12px 4px 2px; font-size:12.5px; color:#8b98a9;}
.legend span {display:inline-flex; align-items:center; gap:7px;}
.legend i {width:11px; height:11px; border-radius:3px; display:inline-block;}

.explain {font-size:14.5px; line-height:1.7; color:#9aa7b8; max-width:900px; padding:6px 4px;}
.explain b {color:#e6edf3;}
.explain p {margin:0 0 12px;}

/* ---------- chargement ---------- */
.progress-bar, .eta-bar {background: linear-gradient(90deg, rgba(45,212,191,.35), rgba(129,140,248,.35)) !important;}

::-webkit-scrollbar {width:10px; height:10px;}
::-webkit-scrollbar-thumb {background: rgba(148,163,184,.2); border-radius:10px;}
::selection {background: rgba(45,212,191,.35);}

@media (max-width: 900px) {.tiles {grid-template-columns:repeat(2,1fr);} #scenario .wrap {grid-template-columns:repeat(2,1fr);}
  .hero h1 {font-size:32px;}}
@media (prefers-reduced-motion: reduce) {*, *::before, *::after {animation: none !important; transition: none !important;}}
""".replace("NOISE", _NOISE)


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
                    village = gr.Dropdown(vlist, value=vlist[0][1], show_label=False, container=False)

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
                                       show_label=False, container=False, elem_id="quality")
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
            section("1 · Aspect post-événement")
            gr.HTML("<div class='hint'>L'image après est prise par un autre capteur, un autre jour : tons, bruit, "
                    "compression, ombres de nuages. Seuls les pixels changent, les masques restent exacts.</div>")
            with gr.Row():
                pe_seed = gr.Number(value=1, precision=0, label="Graine", scale=1)
                pe_btn = gr.Button("Nouveau tirage", variant="primary", scale=1)
            pe_slider = gr.ImageSlider(label="Rendu brut / avec Albumentations", height=560)
            pe_md = gr.HTML()
            section("2 · Crops alignés pour l'entraînement")
            gr.HTML("<div class='hint'>Chaque crop est un même découpage montré trois fois : avant, après, dégâts. "
                    "La géométrie (symétries, rotation, recadrage sur les zones touchées) est identique pour les trois ; "
                    "les couleurs de l'avant et de l'après sont tirées séparément.</div>")
            with gr.Row():
                n_crops = gr.Slider(1, 8, value=4, step=1, label="Nombre de crops")
                crop_size = gr.Slider(128, 512, value=256, step=64, label="Taille (px)")
                crop_seed = gr.Number(value=3, precision=0, label="Graine")
            crop_btn = gr.Button("Générer les crops", variant="primary")
            gallery = gr.Gallery(columns=6, height="auto", object_fit="contain", show_label=False)

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
              <p><b>Limites.</b> Les images sont synthétiques : un modèle doit être évalué sur de vraies images
              post-catastrophe. Le relief de l'inondation est simulé, pas mesuré.</p>
            </div>""")

        # ---- events ---------------------------------------------------------------------------
        inputs = [village, up_img, up_mask, scenario, severity, seed, wind_auto, wind, coverage, point, quality, sensor]
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
        for comp in (village, scenario, seed, wind_auto, quality, sensor, point):
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
    build().launch(theme=THEME, css=CSS, js=FORCE_DARK)

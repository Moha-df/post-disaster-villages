"""
app.py : interface web interactive (Gradio)

    python app.py            ->  http://127.0.0.1:7860
"""

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
        cv2.circle(v, (x, y), r, (234, 88, 12), 3, cv2.LINE_AA)
        cv2.circle(v, (x, y), 3, (234, 88, 12), -1, cv2.LINE_AA)
    if wind_deg is not None and scenario in WIND:
        a = np.radians(wind_deg)
        c = np.array([W - 4 * r, 4 * r])
        d = np.array([np.cos(a), np.sin(a)]) * 3 * r
        p0, p1 = tuple((c - d).astype(int)), tuple((c + d).astype(int))
        cv2.arrowedLine(v, p0, p1, (255, 255, 255), 6, cv2.LINE_AA, tipLength=0.4)
        cv2.arrowedLine(v, p0, p1, (234, 88, 12), 3, cv2.LINE_AA, tipLength=0.4)
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
# design
# ---------------------------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#f6f5f3",
    body_background_fill_dark="#121110",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1c1b19",
    block_border_width="1px",
    block_border_color="#e7e5e1",
    block_border_color_dark="#2c2a27",
    block_shadow="0 1px 2px rgba(0,0,0,.04)",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="#78716c",
    block_label_text_weight="500",
    block_title_text_weight="600",
    button_primary_background_fill="#ea580c",
    button_primary_background_fill_hover="#c2410c",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#23211f",
    button_secondary_border_color="#e7e5e1",
    slider_color="#ea580c",
    checkbox_background_color_selected="#ea580c",
    input_background_fill="#fafaf9",
    input_background_fill_dark="#171615",
)

CSS = """
.gradio-container {max-width: 1440px !important; margin: 0 auto !important; padding: 0 20px !important;}
footer {display: none !important;}

.hero {display:flex; justify-content:space-between; align-items:flex-end; gap:24px; flex-wrap:wrap;
       padding:28px 4px 20px; border-bottom:1px solid var(--border-color-primary); margin-bottom:8px;}
.hero h1 {font-size:30px; line-height:1.15; margin:6px 0 8px; font-weight:700; letter-spacing:-.02em;}
.hero h1 span {color:#ea580c;}
.hero p {max-width:720px; margin:0; color:var(--body-text-color-subdued); font-size:14.5px; line-height:1.55;}
.eyebrow {font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:#ea580c; font-weight:600;}
.badges {display:flex; gap:6px; flex-wrap:wrap;}
.badges span {font-size:12px; padding:5px 10px; border-radius:999px; border:1px solid var(--border-color-primary);
              background:var(--block-background-fill); color:var(--body-text-color-subdued);}

.section {font-size:11px; letter-spacing:.12em; text-transform:uppercase; font-weight:600;
          color:var(--body-text-color-subdued); margin:14px 2px 4px;}
.hint {font-size:13px; color:var(--body-text-color-subdued); padding:2px 4px; line-height:1.45;}
.hint b {color:var(--body-text-color);}

#scenario .wrap {display:grid !important; grid-template-columns:repeat(3,1fr); gap:6px;}
#scenario label {justify-content:center; padding:10px 6px !important; border-radius:8px !important;
                 font-weight:500; font-size:13.5px; transition:all .15s;}
#scenario label input {display:none;}
#scenario label.selected {background:#ea580c !important; color:#fff !important; border-color:#ea580c !important;}
#quality .wrap {display:grid !important; grid-template-columns:1fr 1fr; gap:6px;}
#quality label {justify-content:center; border-radius:8px !important;}
#quality label input {display:none;}
#quality label.selected {background:#292524 !important; color:#fff !important;}

#viewer {border-radius:12px; overflow:hidden;}
#picker img {cursor:crosshair;}

.stats {padding:18px 18px 14px; border:1px solid var(--border-color-primary); border-radius:12px;
        background:var(--block-background-fill);}
.stats.empty {color:var(--body-text-color-subdued); text-align:center; padding:40px;}
.stats-head {display:flex; align-items:baseline; gap:10px; margin-bottom:14px; flex-wrap:wrap;}
.stats-head .title {font-size:20px; font-weight:700; letter-spacing:-.01em;}
.pill {font-size:12px; padding:3px 9px; border-radius:999px; background:rgba(234,88,12,.1); color:#ea580c;
       font-weight:600;}
.tiles {display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:14px;}
.tile {border:1px solid var(--border-color-primary); border-radius:10px; padding:12px 12px 10px;
       position:relative; overflow:hidden;}
.tile .bar {position:absolute; left:0; top:0; right:0; height:4px;}
.tile .num {font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; margin-top:2px;}
.tile .lbl {font-size:12px; color:var(--body-text-color-subdued);}
.kv {display:grid; grid-template-columns:auto 1fr; gap:6px 18px; font-size:13.5px; padding:4px 2px 12px;
     border-bottom:1px solid var(--border-color-primary);}
.kv .k, .aug .k {color:var(--body-text-color-subdued);}
.kv .v {font-weight:500; font-variant-numeric:tabular-nums;}
.aug {padding-top:12px; font-size:13px; display:flex; flex-direction:column; gap:8px;}
.code {display:inline-block; font-family:var(--font-mono); font-size:12px; padding:3px 8px; margin:0 6px 6px 0;
       border-radius:6px; background:var(--background-fill-secondary); border:1px solid var(--border-color-primary);}
.muted {color:var(--body-text-color-subdued);}

.legend {display:flex; flex-wrap:wrap; gap:6px 14px; padding:10px 4px 2px; font-size:12.5px;
         color:var(--body-text-color-subdued);}
.legend span {display:inline-flex; align-items:center; gap:6px;}
.legend i {width:11px; height:11px; border-radius:3px; display:inline-block; border:1px solid rgba(0,0,0,.15);}

.explain {font-size:14px; line-height:1.6; color:var(--body-text-color-subdued); max-width:900px; padding:4px;}
.explain b {color:var(--body-text-color);}
@media (max-width: 900px) {.tiles {grid-template-columns:repeat(2,1fr);} #scenario .wrap {grid-template-columns:repeat(2,1fr);}}
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
    build().launch(theme=THEME, css=CSS)

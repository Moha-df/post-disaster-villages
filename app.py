"""
app.py : interface web interactive (Gradio)
============================================
    python app.py            ->  http://127.0.0.1:7860

- choisir un village (ou envoyer sa propre image + masque OpenEarthMap)
- choisir la catastrophe, la sévérité, la graine, la direction du vent, la hauteur d'eau
- CLIQUER sur l'image « avant » pour placer l'épicentre / le centre de la tornade / le départ du feu
- voir le résultat en direct : curseur avant/après, masques RescueNet & xBD, aléa, changements
- onglet Albumentations : rendu post-événement et crops alignés pour l'entraînement
- télécharger le résultat (images, masques, JSON xBD) en .zip
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
LABELS = {"earthquake": "🌍 Séisme", "hurricane": "🌀 Ouragan", "tornado": "🌪️ Tornade",
          "flood": "🌊 Inondation", "wildfire": "🔥 Feu de forêt"}
POINT_HELP = {"earthquake": "épicentre", "tornado": "centre du couloir de la tornade",
              "wildfire": "point de départ du feu"}
COUNTRIES = {"aachen": "Allemagne", "muenster": "Allemagne", "koeln": "Allemagne", "tyrolw": "Autriche",
             "baybay": "Philippines", "mahe": "Seychelles", "zanzibar": "Tanzanie", "austin": "USA"}


def villages():
    d = os.path.join(VILLAGE_DIR, "images")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".png"))
    return [(f"{n}  ({COUNTRIES.get(n.rsplit('_', 1)[0], '?')})", n) for n in names]


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
    r = max(6, W // 60)
    if point is not None and scenario in POINT_HELP:
        x, y = int(point[0] * W), int(point[1] * H)
        cv2.circle(v, (x, y), r, (255, 40, 40), 3)
        cv2.circle(v, (x, y), 3, (255, 40, 40), -1)
    if wind_deg is not None and scenario in ("hurricane", "tornado", "wildfire"):
        a = np.radians(wind_deg)
        c = np.array([W - 4 * r, 4 * r])
        d = np.array([np.cos(a), np.sin(a)]) * 3 * r
        cv2.arrowedLine(v, tuple((c - d).astype(int)), tuple((c + d).astype(int)), (255, 255, 255), 4, tipLength=0.4)
        cv2.arrowedLine(v, tuple((c - d).astype(int)), tuple((c + d).astype(int)), (30, 30, 30), 2, tipLength=0.4)
    return v


# ---------------------------------------------------------------------------------------------
# main action
# ---------------------------------------------------------------------------------------------
def run(village, upload_img, upload_mask, scenario, severity, seed, wind_auto, wind_deg, coverage,
        point, quality, sensor):
    t0 = time.time()
    side = 512 if quality.startswith("Rapide") else 1000
    if upload_img is not None:
        if upload_mask is None:
            raise gr.Error("Pour une image perso, il faut aussi son masque OpenEarthMap (PNG, valeurs 0-8).")
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
    if not wind_auto and scenario in ("hurricane", "tornado", "wildfire"):
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
    extra = info["extra"]
    lines = [f"### {LABELS[scenario]} — sévérité {severity:.2f}",
             "| Intact | Mineur | Majeur | Détruit |", "|:-:|:-:|:-:|:-:|",
             "| " + " | ".join(f"**{c}**" for c in counts) + " |", "",
             f"- Bâtiments : **{len(info['buildings'])}** — pixels modifiés : **{100 * change.mean():.1f} %**"]
    if "wind_direction_deg" in extra:
        lines.append(f"- Vent / trajectoire : **{extra['wind_direction_deg']:.0f}°**")
    if "epicentre" in extra:
        lines.append(f"- Épicentre : ({extra['epicentre'][0]:.0f}, {extra['epicentre'][1]:.0f}) px"
                     + ("" if opts.get("point") else " — *hors image possible ; cliquez pour le placer*"))
    if "flooded_fraction" in extra:
        lines.append(f"- Surface inondée : **{100 * extra['flooded_fraction']:.0f} %**")
    if "burnt_fraction" in extra:
        lines.append(f"- Surface brûlée : **{100 * extra['burnt_fraction']:.0f} %**")
    if "fallen_trees" in extra:
        lines.append(f"- Arbres tombés : **{len(extra['fallen_trees'])}**")
    lines.append(f"- Albumentations (post-événement) : {', '.join(f'`{f}`' for f in fired) or '*aucune*'}")
    lines.append(f"- Image {W}×{H} px, {gsd:.2f} m/px — calculé en **{time.time() - t0:.1f} s**")

    dmg = C.colorize(xbd, C.XBD_PALETTE)
    dmg[change & (xbd == 0)] = (90, 90, 90)
    overlay = cv2.addWeighted(after, 0.55, dmg, 0.45, 0)
    state = dict(stem=stem, scenario=scenario, severity=severity, seed=int(seed), gsd=gsd, img=img, mask=mask,
                 after=after, clean=clean, ma=ma, xbd=xbd, change=change, hazard=info["hazard"], info=info, fired=fired,
                 extra=extra)
    return ((rgb(img), rgb(after)), rgb(C.colorize(ma)), rgb(overlay), rgb(G.heat(info["hazard"])),
            (change * 255).astype(np.uint8), "\n".join(lines), state)


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
        raise gr.Error("Générez d'abord une image (onglet Catastrophe).")
    size = int(min(size, *state["mask"].shape))
    out = []
    for j, c in enumerate(AUG.paired_crops(rgb(state["img"]), rgb(state["after"]), state["mask"], state["ma"],
                                           state["xbd"], state["change"], int(n), size, int(seed))):
        dmg = rgb(C.colorize(c["xbd"], C.XBD_PALETTE))
        out += [(c["before"], f"#{j} avant"), (c["after"], f"#{j} après"), (dmg, f"#{j} dégâts (xBD)")]
    return out


def post_event_demo(state, seed):
    if not state:
        raise gr.Error("Générez d'abord une image (onglet Catastrophe).")
    # the post-event look on the clean render: what albumentations adds
    out, fired = AUG.apply_post_event(state["clean"], state["scenario"], int(seed))
    return (rgb(state["clean"]), rgb(out)), "Transformations appliquées : " + (", ".join(f"`{f}`" for f in fired) or "*aucune*")


# ---------------------------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------------------------
CSS = """
#title {text-align:center; margin-bottom:0}
#subtitle {text-align:center; opacity:.75; margin-top:0}
.legend span {display:inline-block; padding:2px 8px; margin:2px; border-radius:4px; color:#000; font-size:12px}
"""
LEGEND = ("<div class='legend'>" + "".join(
    f"<span style='background:rgb{tuple(int(v) for v in C.RN_PALETTE[i])};"
    f"color:{'#fff' if C.RN_PALETTE[i].mean() < 110 else '#000'}'>{i} {n}</span>"
    for i, n in enumerate(C.RN_NAMES)) + "</div>")
LEGEND_XBD = ("<div class='legend'>" + "".join(
    f"<span style='background:rgb{tuple(int(v) for v in C.XBD_PALETTE[i])};"
    f"color:{'#fff' if i == 0 else '#000'}'>{n}</span>" for i, n in enumerate(C.XBD_NAMES))
    + "<span style='background:#5a5a5a;color:#fff'>autre changement</span></div>")


def build():
    vlist = villages()
    with gr.Blocks(title="Village → catastrophe") as demo:
        gr.Markdown("# 🏘️ → 🌪️ Village → post-catastrophe", elem_id="title")
        gr.Markdown("Images aériennes OpenEarthMap · dégâts procéduraux guidés par une carte d'aléa · "
                    "labels RescueNet & xBD · Albumentations", elem_id="subtitle")
        state = gr.State(None)
        point = gr.State(None)

        with gr.Tab("Catastrophe"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    village = gr.Dropdown(vlist, value=vlist[0][1], label="Village")
                    scenario = gr.Radio([(v, k) for k, v in LABELS.items()], value="tornado", label="Catastrophe")
                    severity = gr.Slider(0.05, 1.0, value=0.75, step=0.05, label="Sévérité")
                    with gr.Row():
                        seed = gr.Number(value=7, precision=0, label="Graine", min_width=80)
                        dice = gr.Button("🎲 Nouvelle graine", size="sm")
                    with gr.Group(visible=True) as wind_box:
                        wind_auto = gr.Checkbox(value=True, label="Direction du vent automatique")
                        wind = gr.Slider(0, 359, value=45, step=1, label="Direction du vent (°, 0 = vers l'est, 90 = vers le bas)",
                                         interactive=False)
                    coverage = gr.Slider(0.05, 0.8, value=0.35, step=0.05, label="Surface inondée (inondation)",
                                         visible=False)
                    point_md = gr.Markdown("👆 Cliquez sur l'image ci-dessous pour placer le **centre du couloir de la tornade**.")
                    picker = gr.Image(label="Avant — cliquez pour placer le point", interactive=False, height=300)
                    clear_pt = gr.Button("Effacer le point", size="sm")
                    with gr.Row():
                        quality = gr.Radio(["Rapide (512 px)", "Complète (1000 px)"], value="Rapide (512 px)",
                                           label="Résolution")
                    sensor = gr.Checkbox(value=True, label="Rendu post-événement (Albumentations)")
                    live = gr.Checkbox(value=True, label="⚡ Mise à jour en direct")
                    go = gr.Button("Générer", variant="primary")
                    with gr.Accordion("Ma propre image", open=False):
                        gr.Markdown("Image aérienne (~0,3 m/px) **et** son masque OpenEarthMap "
                                    "(PNG, 1 sol nu, 2 prairie, 3 aménagé, 4 route, 5 arbre, 6 eau, 7 agriculture, 8 bâtiment).")
                        up_img = gr.Image(label="Image", type="numpy")
                        up_mask = gr.Image(label="Masque OpenEarthMap", type="numpy", image_mode="L")
                with gr.Column(scale=2):
                    slider = gr.ImageSlider(label="Avant ↔ Après (glissez)", height=620)
                    with gr.Tabs():
                        with gr.Tab("Masque RescueNet"):
                            m_img = gr.Image(show_label=False, height=480)
                            gr.HTML(LEGEND)
                        with gr.Tab("Dégâts (xBD)"):
                            d_img = gr.Image(show_label=False, height=480)
                            gr.HTML(LEGEND_XBD)
                        with gr.Tab("Carte d'aléa"):
                            h_img = gr.Image(show_label=False, height=480)
                            gr.Markdown("Clair = aléa fort. Les dégâts suivent ce champ (+ vulnérabilité de chaque maison).")
                        with gr.Tab("Changements"):
                            c_img = gr.Image(show_label=False, height=480)
                    stats = gr.Markdown()
                    with gr.Row():
                        dl = gr.Button("📦 Télécharger (.zip : images, masques, JSON xBD)")
                        dl_file = gr.File(label="Fichier", height=60)

        with gr.Tab("Albumentations"):
            gr.Markdown("**1. Rendu post-événement** : l'image après est prise par un autre capteur, un autre jour. "
                        "Seulement des transformations *pixel* (les masques restent exacts).")
            with gr.Row():
                pe_seed = gr.Number(value=1, precision=0, label="Graine Albumentations")
                pe_btn = gr.Button("Appliquer / nouveau tirage")
            pe_slider = gr.ImageSlider(label="Rendu brut ↔ avec Albumentations", height=520)
            pe_md = gr.Markdown()
            gr.Markdown("**2. Crops alignés** pour entraîner un modèle de détection de changement : même géométrie "
                        "(`D4`, `Affine`, `CropNonEmptyMaskIfExists`) pour avant/après/masques, couleurs indépendantes.")
            with gr.Row():
                n_crops = gr.Slider(1, 8, value=4, step=1, label="Nombre de crops")
                crop_size = gr.Slider(128, 512, value=256, step=64, label="Taille (px)")
                crop_seed = gr.Number(value=3, precision=0, label="Graine")
                crop_btn = gr.Button("Générer les crops", variant="primary")
            gallery = gr.Gallery(columns=6, height="auto", object_fit="contain")

        # ---- events ---------------------------------------------------------------------------
        inputs = [village, up_img, up_mask, scenario, severity, seed, wind_auto, wind, coverage, point, quality, sensor]
        outputs = [slider, m_img, d_img, h_img, c_img, stats, state]

        def refresh_picker(v, ui, sc, pt, wa, wd):
            img = cv2.cvtColor(ui, cv2.COLOR_RGB2BGR) if ui is not None else load_village(v, 512)[0]
            return mark(img, pt, sc, None if wa else wd)

        def on_scenario(sc):
            txt = (f"👆 Cliquez sur l'image ci-dessous pour placer le **{POINT_HELP[sc]}**."
                   if sc in POINT_HELP else "ℹ️ Pas de point à placer pour ce scénario.")
            return (gr.update(visible=sc == "flood"), txt,
                    gr.update(visible=sc in ("hurricane", "tornado", "wildfire")))

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
    build().launch(theme=gr.themes.Soft(primary_hue="orange"), css=CSS)

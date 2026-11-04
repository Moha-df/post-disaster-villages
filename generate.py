"""
generate.py
===========
Undamaged village (aerial image + OpenEarthMap mask)  ->  the same village after a disaster.

  python generate.py                                    # all villages x all 5 scenarios
  python generate.py --scenarios flood wildfire --n-per-scenario 2 --severity 0.7
  python generate.py --aug-crops 4 --crop-size 512      # + albumentations crops for training

Outputs (in --out):
  before/<village>.jpg                      undamaged image (once per village)
  masks_before/<village>_lab.png            its labels, RescueNet ids
  after/<name>.jpg                          post-disaster image             <name> = <village>_<scenario>_<i>
  masks/<name>_lab.png                      post-disaster labels, RescueNet Kaggle v1.0 ids
  xbd/<name>_damage.png                     per-pixel building damage, xBD scale (0 bg .. 4 destroyed)
  labels_xbd/<name>.json                    one polygon per building + damage subtype (xBD JSON layout)
  change/<name>_change.png                  255 = changed by the disaster
  hazard/<name>_hazard.png                  hazard field used to place the damage (0..255)
  previews/<name>_preview.jpg               before | after | hazard  /  mask before | mask after | damage
  augmented/...                             aligned albumentations crops (see --aug-crops)
  manifest.csv, summary.json
"""

import argparse
import csv
import glob
import json
import os
import zlib

import cv2
import numpy as np

from postdisaster import augment as AUG
from postdisaster import classes as C
from postdisaster import pipeline as P


def polygon_wkt(fp_full):
    cnts, _ = cv2.findContours(fp_full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True)[:, 0]
    if len(c) < 3:
        return None
    pts = ", ".join(f"{x} {y}" for x, y in list(c) + [c[0]])
    return f"POLYGON (({pts}))"


def xbd_json(name, info, b_lab, scenario, meta):
    feats = []
    for b in info["buildings"]:
        x, y, w, h = b["bbox"]
        full = np.zeros(b_lab.shape, bool)
        full[y:y + h, x:x + w] = b_lab[y:y + h, x:x + w] == b["label"]
        wkt = polygon_wkt(full)
        if wkt is None:
            continue
        feats.append(dict(properties=dict(feature_type="building", subtype=C.DAMAGE_NAME[b["damage_level"]],
                                          uid=f"{name}_{b['id']}", hazard=b["hazard"],
                                          area_m2=round(b["area"] * meta["gsd_m"] ** 2, 1)), wkt=wkt))
    return dict(features=dict(xy=feats), metadata=dict(img_name=name + ".jpg", disaster_type=scenario, **meta))


def heat(field):
    return cv2.applyColorMap((np.clip(field, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def title(im, text):
    cv2.rectangle(im, (0, 0), (im.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(im, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def preview(before, after, hazard, mb, ma, xbd, change, info, panel=520):
    s = panel / max(before.shape[:2])
    rs = lambda im, nn=False: cv2.resize(im, None, fx=s, fy=s,
                                         interpolation=cv2.INTER_NEAREST if nn else cv2.INTER_AREA)
    dmg = C.colorize(xbd, C.XBD_PALETTE)
    dmg[change & (xbd == 0)] = (90, 90, 90)
    counts = np.bincount([b["damage_level"] for b in info["buildings"]], minlength=4)
    top = [title(rs(before), "BEFORE"), title(rs(after), "AFTER"), title(rs(heat(hazard)), "HAZARD FIELD")]
    bot = [title(rs(C.colorize(mb), True), "MASK BEFORE (RescueNet)"),
           title(rs(C.colorize(ma), True), "MASK AFTER (RescueNet)"),
           title(rs(dmg, True), "DAMAGE " + " ".join(f"{n[:5]}:{c}" for n, c in zip(C.DAMAGE_NAME, counts)))]
    return np.vstack([np.hstack(top), np.hstack(bot)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default="data/villages/images")
    p.add_argument("--masks", default="data/villages/masks", help="OpenEarthMap masks (same file names)")
    p.add_argument("--out", default="output")
    p.add_argument("--scenarios", nargs="+", default=P.SCENARIOS, choices=P.SCENARIOS)
    p.add_argument("--n-per-scenario", type=int, default=1)
    p.add_argument("--severity", type=float, default=0.75, help="0..1 (+/- 0.1 random)")
    p.add_argument("--gsd", type=float, default=0.3, help="metres per pixel (OpenEarthMap: 0.25-0.5)")
    p.add_argument("--no-sensor-shift", action="store_true", help="skip the albumentations post-event look")
    p.add_argument("--aug-crops", type=int, default=2, help="aligned albumentations crops per sample (0 = none)")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    dirs = ["before", "masks_before", "after", "masks", "xbd", "labels_xbd", "change", "hazard", "previews"]
    if a.aug_crops:
        dirs += [f"augmented/{k}" for k in ("before", "after", "masks_before", "masks", "xbd", "change")]
    for d in dirs:
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.images, "*.png")) + glob.glob(os.path.join(a.images, "*.jpg")))
    rows, totals = [], np.zeros((len(P.SCENARIOS), 4), int)
    for k, ip in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(ip))[0]
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        raw = cv2.imread(os.path.join(a.masks, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None or raw is None:
            print(f"skip {stem}: image or mask missing")
            continue
        mask = C.oem_to_rescuenet(raw)
        blds = P.split_buildings(img, mask, int(np.ceil(6 / a.gsd ** 2)))
        cv2.imwrite(os.path.join(a.out, "before", stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(a.out, "masks_before", stem + "_lab.png"), mask)
        print(f"[{k}/{len(files)}] {stem}: {len(blds[0])} buildings")

        for scenario in a.scenarios:
            for i in range(a.n_per_scenario):
                seed = [a.seed, zlib.crc32(stem.encode()), P.SCENARIOS.index(scenario), i]
                rng = np.random.default_rng(seed)
                sev = float(np.clip(a.severity + rng.uniform(-0.1, 0.1), 0.05, 1))
                after, ma, info = P.generate(img, mask, scenario, sev, a.gsd, rng, buildings=blds)
                change = P.change_map(img, after, mask, ma)
                fired = []
                if not a.no_sensor_shift:
                    after, fired = AUG.apply_post_event(after, scenario, int(rng.integers(1 << 30)))
                xbd = C.rescuenet_to_xbd(ma)
                name = f"{stem}_{scenario}_{i}"
                meta = dict(village=stem, severity=round(sev, 3), gsd_m=a.gsd, seed=seed,
                            post_event_albumentations=fired, **info["extra"])

                cv2.imwrite(os.path.join(a.out, "after", name + ".jpg"), after, [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.imwrite(os.path.join(a.out, "masks", name + "_lab.png"), ma)
                cv2.imwrite(os.path.join(a.out, "xbd", name + "_damage.png"), xbd)
                cv2.imwrite(os.path.join(a.out, "change", name + "_change.png"), change.astype(np.uint8) * 255)
                cv2.imwrite(os.path.join(a.out, "hazard", name + "_hazard.png"),
                            (info["hazard"] * 255).astype(np.uint8))
                with open(os.path.join(a.out, "labels_xbd", name + ".json"), "w", encoding="utf-8") as f:
                    json.dump(xbd_json(name, info, info["b_lab"], scenario, meta), f, indent=1)
                cv2.imwrite(os.path.join(a.out, "previews", name + "_preview.jpg"),
                            preview(img, after, info["hazard"], mask, ma, xbd, change, info),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])

                for j, c in enumerate(AUG.paired_crops(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                                       cv2.cvtColor(after, cv2.COLOR_BGR2RGB), mask, ma, xbd,
                                                       change, a.aug_crops, a.crop_size, seed=int(rng.integers(1 << 30)))):
                    cn = f"{name}_aug{j}"
                    base = os.path.join(a.out, "augmented")
                    cv2.imwrite(os.path.join(base, "before", cn + ".jpg"), cv2.cvtColor(c["before"], cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cv2.imwrite(os.path.join(base, "after", cn + ".jpg"), cv2.cvtColor(c["after"], cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cv2.imwrite(os.path.join(base, "masks_before", cn + "_lab.png"), c["mask_before"])
                    cv2.imwrite(os.path.join(base, "masks", cn + "_lab.png"), c["mask_after"])
                    cv2.imwrite(os.path.join(base, "xbd", cn + "_damage.png"), c["xbd"])
                    cv2.imwrite(os.path.join(base, "change", cn + "_change.png"), c["change"] * 255)

                counts = np.bincount([b["damage_level"] for b in info["buildings"]], minlength=4)
                totals[P.SCENARIOS.index(scenario)] += counts
                rows.append(dict(name=name, village=stem, scenario=scenario, severity=round(sev, 3),
                                 buildings=len(info["buildings"]),
                                 **{C.DAMAGE_NAME[l]: int(counts[l]) for l in range(4)},
                                 changed_pct=round(100 * float(change.mean()), 2),
                                 albumentations=" ".join(fired)))
                print(f"   {name}: sev {sev:.2f}  " + "  ".join(f"{C.DAMAGE_NAME[l]} {counts[l]}" for l in range(4))
                      + f"  changed {100 * change.mean():.1f}%  aug: {', '.join(fired) or '-'}")

    if rows:
        with open(os.path.join(a.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(samples=len(rows), label_scheme="RescueNet Kaggle v1.0", classes=C.RN_NAMES,
                       xbd_classes=C.XBD_NAMES,
                       buildings_per_damage_level={s: dict(zip(C.DAMAGE_NAME, map(int, totals[i])))
                                                   for i, s in enumerate(P.SCENARIOS) if s in a.scenarios},
                       args=vars(a)), f, indent=1)
    print(f"\nDone: {len(rows)} post-disaster images in {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

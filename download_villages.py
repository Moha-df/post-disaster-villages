"""
download_villages.py
====================
Picks undamaged VILLAGE scenes from OpenEarthMap (aerial/satellite, 0.25-0.5 m/px, 44 countries)
without downloading the whole 6 GB dataset.

How it stays small:
  1. the Hugging Face mirror stores the data as Parquet, so we read ONLY the `mask` column
     (lossless PNG, a few dozen KB each) of every row group through HTTP range requests;
  2. each mask is scored: a village = scattered houses + a lot of fields / trees / grassland,
     not a city centre and not an empty field;
  3. we keep the best scene of each REGION first (geographic diversity), then fill up;
  4. only the chosen images are downloaded.

Usage:
  python download_villages.py --n 8 --out data/villages
  python download_villages.py --n 30 --shards 0-10 --out data/villages_big
"""

import argparse
import io
import json
import os

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

REPO = "datasets/EVER-Z/open_earth_map"
N_TRAIN_SHARDS = 11
# OpenEarthMap classes
OEM = {0: "unknown", 1: "bareland", 2: "rangeland", 3: "developed space", 4: "road",
       5: "tree", 6: "water", 7: "agriculture land", 8: "building"}


# OpenEarthMap regions that are sites of real disasters (most come from xBD / disaster-mapping
# campaigns). Some of their tiles are POST-event (e.g. Tonga after cyclone Gita, debris and
# stripped trees), which would be wrong "undamaged" inputs, so they are skipped by default.
DISASTER_REGIONS = {
    "el_rodeo", "leilane_estates",                                   # volcanoes (Fuego, Kilauea)
    "houston", "panama_city", "wallace", "tonga",                    # hurricanes / cyclones
    "jeremie", "les_cayes", "port_a_piment", "saint_louis_du_sud",   # hurricane Matthew, Haiti
    "joplin", "tuscaloosa", "oklahoma", "little_rock", "tulsa",      # tornadoes
    "palu", "labuhan",                                               # tsunamis
    "pedrogao_grande", "santa_rosa", "thousand_oaks",                # wildfires
    "kagera", "mexico_city", "christchurch", "pisco", "ica", "chincha",  # earthquakes
}


def village_score(m):
    """Returns (is_village, score, stats) for one OpenEarthMap mask."""
    valid = max(1, int((m > 0).sum()))
    c = np.bincount(m.ravel(), minlength=9) / valid
    bld, road, dev = c[8], c[4], c[3]
    rural = c[1] + c[2] + c[5] + c[7]
    n, _, st, _ = cv2.connectedComponentsWithStats((m == 8).astype(np.uint8))
    areas = st[1:, 4]
    houses = int((areas >= 40).sum())
    med = float(np.median(areas)) if len(areas) else 0
    ok = (0.03 < bld < 0.15 and rural > 0.55 and dev < 0.15 and 15 <= houses <= 220
          and med < 4000 and road > 0.01 and (m == 0).mean() < 0.05)
    # a village: a cluster of detached houses (not a suburb) surrounded by fields and trees,
    # with some roads; water and trees make floods / fallen trees possible
    size_fit = np.exp(-((np.log(max(houses, 1)) - np.log(80)) ** 2) / 0.5)
    score = size_fit * rural * (1 + 2 * min(road, 0.1)) * (1 + 2 * min(c[6], 0.15) + 0.5 * c[5])
    return ok, float(score), dict(houses=houses, building=round(float(bld), 3),
                                  rural=round(float(rural), 3), road=round(float(road), 3),
                                  water=round(float(c[6]), 3), tree=round(float(c[5]), 3))


def image_quality(rgb):
    """Rejects tiles with no-data borders, clouds / haze, or very low contrast."""
    nodata = (rgb.max(2) < 10).mean()
    if nodata > 0.01:
        return False, f"{nodata:.0%} no-data border"
    dark = cv2.erode(rgb.min(2), np.ones((15, 15), np.uint8))  # dark channel prior: haze lifts it
    if np.median(dark) > 42:
        return False, f"hazy (dark channel {np.median(dark):.0f})"
    if rgb.std() < 20:
        return False, f"low contrast ({rgb.std():.0f})"
    return True, ""


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8, help="villages to keep")
    p.add_argument("--shards", default="0-10", help="train shards to scan, e.g. 0-3 or 1,5,9")
    p.add_argument("--per-region", type=int, default=1, help="max villages per region (first pass)")
    p.add_argument("--out", default="data/villages")
    p.add_argument("--allow-disaster-regions", action="store_true",
                   help="also use regions of real disasters (their tiles may be post-event)")
    a = p.parse_args()

    fs = HfFileSystem()
    cands = []
    for s in parse_range(a.shards):
        path = f"{REPO}/data/train-{s:05d}-of-{N_TRAIN_SHARDS:05d}.parquet"
        pf = pq.ParquetFile(fs.open(path, block_size=2 ** 20))
        for g in range(pf.metadata.num_row_groups):
            tb = pf.read_row_group(g, columns=["image_name", "mask"])
            for i, (name, mk) in enumerate(zip(tb.column(0).to_pylist(), tb.column(1).to_pylist())):
                if not a.allow_disaster_regions and name.rsplit("_", 1)[0] in DISASTER_REGIONS:
                    continue
                m = np.array(Image.open(io.BytesIO(mk["bytes"])))
                ok, score, stats = village_score(m)
                if ok:
                    cands.append(dict(name=name, shard=s, group=g, row=i, score=score, **stats))
        print(f"shard {s}: {len(cands)} village candidates so far")

    # diversity first: best village of each region, then the next best ones
    cands.sort(key=lambda c: -c["score"])
    region = lambda c: c["name"].rsplit("_", 1)[0]
    order = []
    for rnd in range(1, 50):  # round 1: best of each region, round 2: second best, ...
        seen = {}
        for c in cands:
            seen[region(c)] = seen.get(region(c), 0) + 1
            if seen[region(c)] == rnd:
                order.append(c)
        if len(order) >= len(cands):
            break

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    tables, keep, per = {}, [], {}
    for c in order:
        if len(keep) >= a.n:
            break
        if per.get(region(c), 0) >= a.per_region and len(per) < len({region(x) for x in cands}):
            continue
        key = (c["shard"], c["group"])
        if key not in tables:
            path = f"{REPO}/data/train-{c['shard']:05d}-of-{N_TRAIN_SHARDS:05d}.parquet"
            tables[key] = pq.ParquetFile(fs.open(path, block_size=2 ** 20)).read_row_group(
                c["group"], columns=["image", "mask"])
        tb = tables[key]
        img = Image.open(io.BytesIO(tb.column(0)[c["row"]].as_py()["bytes"])).convert("RGB")
        ok, why = image_quality(np.array(img))
        if not ok:
            print(f"  skip {c['name']}: {why}")
            continue
        stem = os.path.splitext(c["name"])[0]
        img.save(os.path.join(a.out, "images", stem + ".png"))
        with open(os.path.join(a.out, "masks", stem + ".png"), "wb") as f:
            f.write(tb.column(1)[c["row"]].as_py()["bytes"])
        keep.append(c)
        per[region(c)] = per.get(region(c), 0) + 1
        print(f"  {stem}: {c['houses']} houses, buildings {c['building']:.0%}, "
              f"rural {c['rural']:.0%}, water {c['water']:.0%}")
    with open(os.path.join(a.out, "villages.json"), "w", encoding="utf-8") as f:
        json.dump(dict(source="OpenEarthMap (Xia et al., WACV 2023), HF mirror EVER-Z/open_earth_map",
                       classes=OEM, villages=keep), f, indent=1)
    print(f"Done: {len(keep)} villages from {len(per)} regions in {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

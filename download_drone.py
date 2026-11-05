"""
download_drone.py
=================
High-resolution DRONE scenes from the Semantic Drone Dataset (TU Graz, Institute of Computer Graphics
and Vision): 400 nadir photos of suburban houses, 6000 x 4000 px, 5-30 m above ground, 24 classes.

The Kaggle copy (bulentsiyah/semantic-drone-dataset) is a 4 GB zip. Like download_villages.py, only
what is needed is read, with HTTP range requests: the zip index, then the chosen photos and masks.

Each scene is resized to a long side of 1536 px (about 2.2 cm/pixel, estimated from the size of the
cars) and its mask is converted to OpenEarthMap ids, so that generate.py and app.py use it as is.

Usage:
  python download_drone.py                       # the scenes used in the paper
  python download_drone.py --ids 128 239 016 287 --out data/drone

Licence: free for non-commercial use (research, teaching, publications), see
the Semantic Drone Dataset page of TU Graz (ICG) - cite the dataset when you use it.
"""

import argparse
import io
import json
import os
import urllib.error
import urllib.request
import zipfile

import cv2
import numpy as np

KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/bulentsiyah/semantic-drone-dataset"
ROOT = "dataset/semantic_drone_dataset"
LONG_SIDE = 1536
GSD_FULL = 0.0055  # m/px at 6000 px (cars: ~4.5 m long, ~770 px)

# Semantic Drone Dataset ids -> OpenEarthMap ids
#  SDD: 0 unlabeled 1 paved-area 2 dirt 3 grass 4 gravel 5 water 6 rocks 7 pool 8 vegetation 9 roof
#       10 wall 11 window 12 door 13 fence 14 fence-pole 15 person 16 dog 17 car 18 bicycle 19 tree
#       20 bald-tree 21 ar-marker 22 obstacle 23 conflicting
#  OEM: 0 unknown 1 bareland 2 rangeland 3 developed space 4 road 5 tree 6 water 7 agriculture 8 building
SDD_TO_OEM = {0: 3, 1: 4, 2: 1, 3: 2, 4: 1, 5: 6, 6: 1, 7: 6, 8: 5, 9: 8, 10: 8, 11: 8, 12: 8, 13: 3, 14: 3,
              15: 3, 16: 3, 17: 4, 18: 3, 19: 5, 20: 5, 21: 3, 22: 3, 23: 3}


def sdd_to_oem(m):
    lut = np.full(256, 3, np.uint8)
    for k, v in SDD_TO_OEM.items():
        lut[k] = v
    return lut[m]


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url, block=1 << 20):
        self.url, self.pos, self.block, self.cache = url, 0, block, {}
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = [off, self.pos + off, self.size + off][whence]
        return self.pos

    def _get(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * self.block:
            data = self._get(self.pos, self.pos + n)
        else:
            out, p, end = [], self.pos, self.pos + n
            while p < end:
                b = p // self.block
                if b not in self.cache:
                    if len(self.cache) > 64:
                        self.cache.clear()
                    self.cache[b] = self._get(b * self.block, min(self.size, (b + 1) * self.block))
                take = self.cache[b][p - b * self.block:p - b * self.block + end - p]
                out.append(take)
                p += len(take)
            data = b"".join(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def open_zip():
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    try:
        urllib.request.build_opener(NoRedirect).open(KAGGLE)
    except urllib.error.HTTPError as e:  # Kaggle answers with a redirect to a signed storage URL
        return zipfile.ZipFile(RangeFile(e.headers["Location"]))
    raise RuntimeError("Kaggle did not return a download link")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="+", default=["128", "239", "016"], help="SDD image ids (000-598)")
    p.add_argument("--out", default="data/drone")
    a = p.parse_args()

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    z = open_zip()
    scenes = []
    for i in a.ids:
        img = cv2.imdecode(np.frombuffer(z.read(f"{ROOT}/original_images/{i}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        lab = cv2.imdecode(np.frombuffer(z.read(f"{ROOT}/label_images_semantic/{i}.png"), np.uint8),
                           cv2.IMREAD_GRAYSCALE)
        s = LONG_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        stem = f"graz_{i}"
        cv2.imwrite(os.path.join(a.out, "images", stem + ".png"), img)
        cv2.imwrite(os.path.join(a.out, "masks", stem + ".png"), sdd_to_oem(lab))
        scenes.append(dict(name=stem, sdd_id=i, gsd_m=round(GSD_FULL / s, 4), size=list(img.shape[:2])))
        print(f"  {stem}: {img.shape[1]}x{img.shape[0]} px, ~{GSD_FULL / s * 100:.1f} cm/px")
    with open(os.path.join(a.out, "scenes.json"), "w", encoding="utf-8") as f:
        json.dump(dict(source="Semantic Drone Dataset, TU Graz (Kaggle copy bulentsiyah/semantic-drone-dataset)",
                       classes="OpenEarthMap ids (converted)", scenes=scenes), f, indent=1)
    print(f"Done: {len(scenes)} drone scenes in {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

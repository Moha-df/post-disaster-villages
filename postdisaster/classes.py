"""Label schemes: OpenEarthMap (input), RescueNet Kaggle v1.0 and xBD (outputs)."""

import numpy as np

# --- input: OpenEarthMap -----------------------------------------------------------------------
OEM_NAMES = ["unknown", "bareland", "rangeland", "developed space", "road", "tree", "water",
             "agriculture land", "building"]

# --- output 1: RescueNet (Kaggle v1.0 numbering) ------------------------------------------------
RN_NAMES = ["background", "debris", "water", "building-no-damage", "building-minor-damage",
            "building-major-damage", "building-total-destruction", "vehicle", "road", "tree",
            "pool", "sand"]
BACKGROUND, DEBRIS, WATER, BLD_NONE, BLD_MINOR, BLD_MAJOR, BLD_TOTAL, VEHICLE, ROAD, TREE, POOL, SAND = range(12)
BLD_BY_LEVEL = [BLD_NONE, BLD_MINOR, BLD_MAJOR, BLD_TOTAL]
BUILDINGS = BLD_BY_LEVEL
GROUND = [BACKGROUND, SAND, DEBRIS]
RN_PALETTE = np.array([  # RGB
    (0, 0, 0), (200, 120, 60), (30, 120, 255), (60, 200, 60), (255, 230, 0), (255, 140, 0),
    (220, 0, 0), (255, 0, 255), (160, 160, 160), (0, 100, 0), (0, 255, 255), (230, 210, 150)],
    np.uint8)

# --- output 2: xBD / xView2 damage scale (per-pixel, buildings only) -----------------------------
XBD_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
XBD_PALETTE = np.array([(0, 0, 0), (60, 200, 60), (255, 230, 0), (255, 140, 0), (220, 0, 0)], np.uint8)
DAMAGE_NAME = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def oem_to_rescuenet(m):
    """OpenEarthMap -> RescueNet. Fields, grass and 'developed space' (yards, parking) are ground,
    bare land becomes 'sand' (bare soil), trees, roads, water and buildings map directly."""
    lut = np.full(256, BACKGROUND, np.uint8)
    lut[1] = SAND
    lut[4], lut[5], lut[6], lut[8] = ROAD, TREE, WATER, BLD_NONE
    return lut[m]


def rescuenet_to_xbd(m):
    lut = np.zeros(256, np.uint8)
    for lvl, c in enumerate(BLD_BY_LEVEL):
        lut[c] = lvl + 1
    return lut[m]


def colorize(mask, palette=RN_PALETTE):
    """Mask -> BGR colour image (for previews)."""
    return palette[np.clip(mask, 0, len(palette) - 1)][..., ::-1].copy()

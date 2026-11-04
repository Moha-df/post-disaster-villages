"""
Albumentations (https://albumentations.ai) in two places:

1. post_event(): the "after" image is taken by ANOTHER flight / satellite on ANOTHER day.
   Pixel-level transforms only (masks stay valid): tone and white balance, sensor noise,
   JPEG compression, lower resolution, cloud shadows after storms, uneven light under dust or
   smoke (plasma brightness/contrast).

2. paired_crops(): dataset enrichment for change detection / damage segmentation.
   The SAME geometric transform (D4 symmetries, small rotation/scale, crop centred on changed
   areas) is applied to before, after and every mask through `additional_targets`, so they stay
   pixel-aligned; colour transforms are drawn INDEPENDENTLY for before and after.
"""

import albumentations as A
import cv2
import numpy as np


def seeded(compose, seed):
    """Gives EVERY transform its own seed. With `Compose(seed=...)` in albumentations 2.0 all
    children draw the same first random number, so they fire all together or not at all."""
    k = [0]

    def walk(ts):
        for t in ts:
            k[0] += 1
            t.set_random_seed(seed * 1009 + k[0])
            if isinstance(t, A.BaseCompose):
                walk(t.transforms)

    walk(compose.transforms)
    return compose


def post_event(scenario, seed):
    common = [
        A.RandomToneCurve(scale=0.08, p=0.5),
        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=6, p=0.6),
        A.OneOf([A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.2)),
                 A.GaussNoise(std_range=(0.01, 0.03))], p=0.5),
        A.Downscale(scale_range=(0.7, 0.9), p=0.2),
        A.ImageCompression(quality_range=(70, 95), p=0.6),
    ]
    weather = {
        # cloud shadows after the storm (PlasmaShadow: soft, irregular; RandomShadow draws
        # hard-edged polygons that look nothing like clouds from above)
        "hurricane": [A.PlasmaShadow(shadow_intensity_range=(0.15, 0.35), roughness=2.5, p=0.5),
                      A.PlasmaBrightnessContrast(brightness_range=(-0.1, 0.05), contrast_range=(-0.1, 0.05),
                                                 roughness=3, p=0.5)],
        "flood": [A.PlasmaShadow(shadow_intensity_range=(0.1, 0.3), roughness=2.5, p=0.4),
                  A.PlasmaBrightnessContrast(brightness_range=(-0.1, 0.05), contrast_range=(-0.1, 0.05),
                                             roughness=3, p=0.5)],
        "tornado": [A.PlasmaShadow(shadow_intensity_range=(0.1, 0.3), roughness=3, p=0.5)],
        # (RandomFog was tried: it blurs the whole tile and hides the damage, so dust and smoke
        #  are drawn by effects.haze / effects.smoke and only the light changes here)
        "earthquake": [A.PlasmaBrightnessContrast(brightness_range=(0.0, 0.08), contrast_range=(-0.12, 0.0),
                                                  roughness=2.5, p=0.6),
                       A.RandomGamma(gamma_limit=(90, 110), p=0.4)],
        "wildfire": [A.ColorJitter(brightness=0.05, contrast=0.08, saturation=0.1, hue=0.02, p=0.5),
                     A.PlasmaBrightnessContrast(brightness_range=(-0.08, 0.02), contrast_range=(-0.1, 0.0),
                                                roughness=2.5, p=0.5)],
    }[scenario]
    return seeded(A.Compose(weather + common, save_applied_params=True), seed)


def apply_post_event(after_bgr, scenario, seed):
    """Returns the transformed image and the names of the transforms that actually fired."""
    r = post_event(scenario, seed)(image=cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB))
    fired = [name for name, _ in r.get("applied_transforms", [])]
    return cv2.cvtColor(r["image"], cv2.COLOR_RGB2BGR), fired


TARGETS = {"after": "image", "mask_before": "mask", "mask_after": "mask", "xbd": "mask"}


def paired_crops(before, after, mask_before, mask_after, xbd, change, n, size, seed):
    """Yields n aligned crops (dicts with the same keys + 'change')."""
    geo = A.Compose([
        A.D4(p=1.0),
        A.Affine(rotate=(-20, 20), scale=(0.85, 1.2), border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.CropNonEmptyMaskIfExists(height=size, width=size),  # crops on damaged areas first
    ], additional_targets=TARGETS)
    seeded(geo, seed)
    color = lambda _: A.Compose([
        A.RandomBrightnessContrast(0.15, 0.15, p=0.7),
        A.HueSaturationValue(5, 15, 8, p=0.5),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 5)),
                 A.Sharpen(alpha=(0.1, 0.3))], p=0.4),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.4),
    ])
    cb, ca = seeded(color(0), seed + 1), seeded(color(0), seed + 2)
    for _ in range(n):
        r = geo(image=before, after=after, mask=change.astype(np.uint8), mask_before=mask_before,
                mask_after=mask_after, xbd=xbd)
        yield dict(before=cb(image=r["image"])["image"], after=ca(image=r["after"])["image"],
                   mask_before=r["mask_before"], mask_after=r["mask_after"], xbd=r["xbd"], change=r["mask"])

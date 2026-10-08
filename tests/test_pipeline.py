import numpy as np

from pbn import pipeline as P


def gradient_with_patch(patch_rgb: tuple[int, int, int], patch: slice) -> np.ndarray:
    """A smooth warm gradient (which soaks up K-means colors) with one small off-color square."""
    h, w = 200, 200
    ramp = np.linspace(40, 230, w, dtype=np.float32)
    rgb = np.stack([ramp, ramp * 0.8, ramp * 0.55], axis=-1)[None].repeat(h, axis=0)
    rgb[patch, patch] = patch_rgb
    return rgb.astype(np.uint8)


def nearest_delta_e(palette: np.ndarray, rgb: tuple[int, int, int]) -> float:
    target = P.to_lab(np.array([[rgb]], np.uint8))[0, 0]
    return float(np.linalg.norm(P.palette_lab(palette) - target, axis=1).min())


def test_rescue_adds_a_small_distinctive_color() -> None:
    pink = (200, 150, 150)  # like a cat's nose against warm fur
    rgb = gradient_with_patch(pink, slice(90, 102))  # 144 px, ~0.4% of the image
    without, _ = P.quantize(rgb, P.Params(n_colors=8, min_color_gap=0, rescue_colors=0))
    with_rescue, labels = P.quantize(rgb, P.Params(n_colors=8, min_color_gap=0))
    assert nearest_delta_e(without, pink) > 20  # plain K-means spends every color on the gradient
    assert nearest_delta_e(with_rescue, pink) < 5
    assert len(with_rescue) == 8
    assert len(np.unique(labels[90:102, 90:102])) == 1


def test_gradient_colors_are_spaced_apart() -> None:
    rgb = gradient_with_patch((0, 0, 0), slice(0, 0))  # plain gradient
    crowded, _ = P.quantize(rgb, P.Params(n_colors=16, min_color_gap=0, rescue_colors=0))
    spaced, labels = P.quantize(rgb, P.Params(n_colors=16, min_color_gap=10, rescue_colors=0))
    gaps = np.linalg.norm(P.palette_lab(spaced)[:, None] - P.palette_lab(spaced)[None], axis=2)
    np.fill_diagonal(gaps, np.inf)
    assert gaps.min() >= 10
    assert len(spaced) < len(crowded)
    # Still a ramp: colors step monotonically from dark to light.
    lightness = P.palette_lab(spaced)[labels[0], 0]
    assert (np.diff(lightness) >= 0).all()


def test_rescue_does_not_duplicate_colors() -> None:
    rgb = gradient_with_patch((40, 70, 200), slice(90, 105))
    palette, _ = P.quantize(rgb, P.Params(n_colors=6, rescue_colors=4))
    lab = P.palette_lab(palette)
    gaps = np.linalg.norm(lab[:, None] - lab[None], axis=2)
    np.fill_diagonal(gaps, np.inf)
    assert gaps.min() > 3


def test_majority_filter_keeps_catchlight() -> None:
    palette = np.array([[60, 50, 30], [250, 250, 250]], np.uint8)  # dark iris, white catchlight
    labels = np.zeros((60, 60), np.int32)
    labels[28:32, 28:32] = 1  # 16 px, thinner than the 7x7 filter can keep
    filtered = P.majority_filter(labels, 2, 7)
    assert not (filtered == 1).any()
    restored = P.restore_details(labels, filtered, palette, min_area=10, delta_e=35)
    assert (restored[28:32, 28:32] == 1).all()


def test_majority_filter_still_removes_specks() -> None:
    palette = np.array([[60, 50, 30], [250, 250, 250]], np.uint8)
    labels = np.zeros((60, 60), np.int32)
    labels[30, 30] = labels[10, 45] = 1  # single-pixel noise
    filtered = P.majority_filter(labels, 2, 7)
    restored = P.restore_details(labels, filtered, palette, min_area=10, delta_e=35)
    assert not (restored == 1).any()

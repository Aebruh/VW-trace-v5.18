from __future__ import annotations

"""Android-only low-allocation geometry adapter for GD VW Tracer v5.18.

This does not change the VW candidate search, rectangle sizes, angles, ordering,
containment tests, scoring, or region budgets. It replaces expensive
box -> affinity.rotate -> affinity.translate construction with the equivalent
four-corner Polygon directly, and adds persistent sub-stage diagnostics.
"""

import math
import os
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


def _rss_mb():
    try:
        for line in Path('/proc/self/status').read_text(errors='ignore').splitlines():
            if line.startswith('VmRSS:'):
                return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return None


def _progress(stage: str):
    path = os.environ.get('GDVW_STAGE_FILE')
    if not path:
        return
    try:
        ctx = os.environ.get('GDVW_REGION_CONTEXT', '').strip()
        label = f'{ctx} — {stage}' if ctx else stage
        rss = _rss_mb()
        if rss is not None:
            label += f' [RAM {rss} MB]'
        Path(path).write_text(label, encoding='utf8')
    except Exception:
        pass


def _direct_rect(cx, cy, width, height, angle_deg):
    """Equivalent to box -> rotate about origin -> translate, with one Polygon."""
    hw = float(width) * 0.5
    hh = float(height) * 0.5
    th = math.radians(float(angle_deg))
    c = math.cos(th)
    s = math.sin(th)

    # Keep a stable counter-clockwise ring. GEOS sees the same rectangle shape as
    # the desktop affinity path, but without allocating intermediate geometries.
    coords = []
    for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        coords.append((float(cx) + c * x - s * y,
                       float(cy) + s * x + c * y))
    return Polygon(coords)


def install_android_geometry(namespace: dict) -> None:
    original_rect_polygon = namespace['rect_polygon']
    original_grow_free = namespace['grow_inscribed_rectangle_free']

    def rect_polygon_low_alloc(cx, cy, width, height, angle_deg):
        return _direct_rect(cx, cy, width, height, angle_deg)

    def grow_inscribed_rectangle_free_low_alloc(safe_cage, safe_prepared,
                                                target_piece,
                                                center, angle,
                                                binary_steps=8,
                                                height_samples=7):
        # Same rotated target bounds as affinity.rotate(target_piece, -angle,
        # origin=center), calculated numerically from the exterior ring. Holes
        # cannot extend beyond the exterior, so they do not affect bounds.
        cx, cy = center
        pts = np.asarray(target_piece.exterior.coords, dtype=np.float64)
        if len(pts) == 0:
            return None

        th = math.radians(float(angle))
        c = math.cos(th)
        s = math.sin(th)
        dx = pts[:, 0] - float(cx)
        dy = pts[:, 1] - float(cy)

        # Rotation by -angle around (cx, cy).
        lx = float(cx) + c * dx + s * dy
        ly = float(cy) - s * dx + c * dy
        minx = float(np.min(lx)); maxx = float(np.max(lx))
        miny = float(np.min(ly)); maxy = float(np.max(ly))

        max_half_w = max(0.15, max(abs(maxx - cx), abs(cx - minx)))
        max_half_h = max(0.15, max(abs(maxy - cy), abs(cy - miny)))

        fracs = np.linspace(0.16, 1.0, height_samples)
        best = None
        for frac in fracs:
            half_h = max(0.12, max_half_h * float(frac))
            low, high = 0.12, max_half_w
            best_rect = None
            for _ in range(binary_steps):
                half_w = (low + high) / 2.0
                rect = _direct_rect(cx, cy, 2.0 * half_w, 2.0 * half_h, angle)
                if safe_prepared.covers(rect):
                    best_rect = rect
                    low = half_w
                else:
                    high = half_w
            if best_rect is None:
                continue
            gain = best_rect.intersection(target_piece).area
            score = gain + 0.035 * best_rect.area
            if best is None or score > best[0]:
                best = (score, best_rect, gain)

        for frac in np.linspace(0.16, 1.0, max(4, height_samples - 2)):
            half_w = max(0.12, max_half_w * float(frac))
            low, high = 0.12, max_half_h
            best_rect = None
            for _ in range(binary_steps):
                half_h = (low + high) / 2.0
                rect = _direct_rect(cx, cy, 2.0 * half_w, 2.0 * half_h, angle)
                if safe_prepared.covers(rect):
                    best_rect = rect
                    low = half_h
                else:
                    high = half_h
            if best_rect is None:
                continue
            gain = best_rect.intersection(target_piece).area
            score = gain + 0.035 * best_rect.area
            if best is None or score > best[0]:
                best = (score, best_rect, gain)

        if best is None or best[2] <= 0:
            return None
        return best[1], best[2]

    # Diagnostic wrappers around the exact existing region stages. These don't
    # alter returned values; they only update crash_stage.txt before/after calls.
    original_build_region_polygon = namespace['build_region_polygon']
    original_cage_edges = namespace['cage_edge_rectangles']
    original_fit_region = namespace['fit_region']
    original_quick_detail = namespace['quick_detail_fill']
    original_union = namespace['unary_union']
    original_mic = namespace['maximum_inscribed_circle']

    def build_region_polygon_logged(*args, **kwargs):
        _progress('building VW polygon')
        out = original_build_region_polygon(*args, **kwargs)
        _progress('VW polygon ready')
        return out

    def cage_edges_logged(*args, **kwargs):
        _progress('building VW edge rectangles')
        out = original_cage_edges(*args, **kwargs)
        _progress(f'VW edge rectangles ready ({len(out)})')
        return out

    def fit_region_logged(*args, **kwargs):
        _progress('fitting large interior rectangles')
        out = original_fit_region(*args, **kwargs)
        try:
            _progress(f'interior fill ready ({len(out[0])})')
        except Exception:
            _progress('interior fill ready')
        return out

    def quick_detail_logged(*args, **kwargs):
        _progress('fitting small-detail rectangles')
        out = original_quick_detail(*args, **kwargs)
        _progress('small-detail fill ready')
        return out

    def unary_union_logged(*args, **kwargs):
        _progress('GEOS unary union')
        return original_union(*args, **kwargs)

    def mic_logged(*args, **kwargs):
        _progress('finding inscribed center')
        return original_mic(*args, **kwargs)

    namespace['rect_polygon_android_original'] = original_rect_polygon
    namespace['grow_inscribed_rectangle_free_android_original'] = original_grow_free
    namespace['rect_polygon'] = rect_polygon_low_alloc
    namespace['grow_inscribed_rectangle_free'] = grow_inscribed_rectangle_free_low_alloc
    namespace['build_region_polygon'] = build_region_polygon_logged
    namespace['cage_edge_rectangles'] = cage_edges_logged
    namespace['fit_region'] = fit_region_logged
    namespace['quick_detail_fill'] = quick_detail_logged
    namespace['unary_union'] = unary_union_logged
    namespace['maximum_inscribed_circle'] = mic_logged

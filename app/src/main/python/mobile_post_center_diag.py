from __future__ import annotations

"""Android-only diagnostics for the exact geometry calls after the inscribed center.

This intentionally mirrors the original ``piece_candidate_centers`` logic and only
adds persistent checkpoints before/after each Shapely/GEOS operation. No scoring,
limits, fractions or candidate ordering are changed.
"""

import math
import os
from pathlib import Path


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


def install_post_center_diagnostics(namespace: dict) -> None:
    Point = namespace['Point']

    def piece_candidate_centers_logged(piece, safe_cage, limit: int = 7):
        centers = []

        try:
            _progress('candidate centers: calling inscribed center')
            mic = namespace['maximum_inscribed_circle'](
                piece,
                tolerance=max(0.08, math.sqrt(piece.area) * 0.006),
            )
            _progress('candidate centers: reading inscribed-center coordinates')
            mc = list(mic.coords)[0]
            centers.append((float(mc[0]), float(mc[1])))
        except Exception:
            _progress('candidate centers: inscribed center skipped after Python exception')

        _progress('candidate centers: representative_point')
        rp = piece.representative_point()
        _progress('candidate centers: representative_point ready')
        centers.append((rp.x, rp.y))

        _progress('candidate centers: centroid')
        centroid = piece.centroid
        _progress('candidate centers: centroid ready')

        _progress('candidate centers: covers centroid')
        centroid_inside = piece.covers(centroid)
        _progress('candidate centers: covers centroid ready')
        if centroid_inside:
            centers.append((centroid.x, centroid.y))

        _progress('candidate centers: minimum rotated rectangle')
        mrr = piece.minimum_rotated_rectangle
        _progress('candidate centers: minimum rotated rectangle ready')

        _progress('candidate centers: reading rectangle coordinates')
        coords = list(mrr.exterior.coords)[:4]
        _progress('candidate centers: rectangle coordinates ready')
        if len(coords) == 4:
            edges = []
            for a, b in zip(coords, coords[1:] + coords[:1]):
                vx, vy = b[0] - a[0], b[1] - a[1]
                L = math.hypot(vx, vy)
                if L > 1e-8:
                    edges.append((L, vx / L, vy / L))
            edges.sort(reverse=True, key=lambda x: x[0])
            if len(edges) >= 2:
                long_len, ux, uy = edges[0]
                vx, vy = -uy, ux

                _progress('candidate centers: selecting MRR anchor')
                if centroid_inside:
                    c = centroid
                else:
                    c = rp
                cx, cy = c.x, c.y

                for frac in (-0.28, 0.28):
                    centers.append((cx + ux * long_len * frac,
                                    cy + uy * long_len * frac))
                short_len = min(e[0] for e in edges)
                for frac in (-0.24, 0.24):
                    centers.append((cx + vx * short_len * frac,
                                    cy + vy * short_len * frac))

        unique = []
        for index, (x, y) in enumerate(centers, start=1):
            _progress(f'candidate centers: creating point {index}/{len(centers)}')
            pt = Point(float(x), float(y))
            _progress(f'candidate centers: safe_cage covers point {index}/{len(centers)}')
            if not safe_cage.covers(pt):
                continue
            if all(math.hypot(x - q[0], y - q[1]) > 0.65 for q in unique):
                unique.append((float(x), float(y)))

        _progress(f'candidate centers ready ({len(unique[:limit])})')
        return unique[:limit]

    namespace['piece_candidate_centers_android_original'] = namespace['piece_candidate_centers']
    namespace['piece_candidate_centers'] = piece_candidate_centers_logged

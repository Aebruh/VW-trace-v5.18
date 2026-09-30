"""Compatibility helpers for Chaquopy's Android Shapely 1.8 wheels.
Desktop Shapely 2.x behaviour is preserved when available.
"""
import os
from pathlib import Path

import numpy as np
from shapely.strtree import STRtree as _NativeSTRtree


class STRtree:
    """Expose Shapely 2-style integer-index query results on Shapely 1.8."""
    def __init__(self, geoms):
        self.geoms = list(geoms)
        self._tree = _NativeSTRtree(self.geoms)
        self._by_id = {id(g): i for i, g in enumerate(self.geoms)}
        self._by_wkb = {}
        for i, g in enumerate(self.geoms):
            self._by_wkb.setdefault(g.wkb, []).append(i)

    def _index(self, item):
        if isinstance(item, (int, np.integer)):
            return int(item)
        found = self._by_id.get(id(item))
        if found is not None:
            return found
        candidates = self._by_wkb.get(item.wkb, [])
        return candidates[0] if candidates else None

    def query(self, geom, predicate=None):
        result = self._tree.query(geom)
        arr = np.asarray(result)
        if arr.size == 0:
            return np.asarray([], dtype=np.int64)
        if np.issubdtype(arr.dtype, np.integer):
            ids = [int(x) for x in arr.tolist()]
        else:
            ids = [self._index(x) for x in result]
            ids = [i for i in ids if i is not None]
        if predicate:
            if predicate == 'intersects':
                ids = [i for i in ids if self.geoms[i].intersects(geom)]
            elif predicate == 'contains':
                ids = [i for i in ids if self.geoms[i].contains(geom)]
            elif predicate == 'within':
                ids = [i for i in ids if self.geoms[i].within(geom)]
            else:
                fn = getattr(self.geoms[0], predicate, None)
                if fn is None:
                    raise ValueError('Unsupported STRtree predicate: ' + str(predicate))
                ids = [i for i in ids if getattr(self.geoms[i], predicate)(geom)]
        return np.asarray(ids, dtype=np.int64)


def _rss_mb():
    try:
        for line in Path('/proc/self/status').read_text(errors='ignore').splitlines():
            if line.startswith('VmRSS:'):
                return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return None


def _compat_progress(stage):
    """Persist a fine-grained Android crash checkpoint when enabled."""
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


try:
    # Desktop / newer Shapely: keep the original native implementation.
    from shapely import maximum_inscribed_circle as _maximum_inscribed_circle

    def maximum_inscribed_circle(geom, tolerance=0.01):
        return _maximum_inscribed_circle(geom, tolerance=tolerance)
except Exception:
    # Chaquopy currently provides Shapely 1.8.x. Both GEOS polylabel and the
    # first OpenCV distance-transform fallback have hard-crashed on-device for
    # some valid residual polygons. This fallback therefore performs the center
    # search using NumPy/Python only. The tracer consumes only ``mic.coords[0]``
    # as a seed point, so a tiny compatibility object is sufficient and avoids
    # creating any additional GEOS geometry here.

    class _CompatLine:
        __slots__ = ('coords',)

        def __init__(self, a, b):
            self.coords = (tuple(a), tuple(b))

    def _polygon_parts(geom):
        if geom is None or geom.is_empty:
            return []
        if geom.geom_type == 'Polygon':
            return [geom]
        return [g for g in getattr(geom, 'geoms', [])
                if g.geom_type == 'Polygon' and not g.is_empty]

    def _ring_array(ring):
        arr = np.asarray(ring.coords, dtype=np.float64)
        if len(arr) < 3:
            return np.empty((0, 2), dtype=np.float64)
        if np.linalg.norm(arr[0] - arr[-1]) > 1e-12:
            arr = np.vstack((arr, arr[0]))
        return arr

    def _points_in_ring(points, ring_arr):
        """Vectorized ray-cast over points, looping only over ring edges."""
        n = len(points)
        inside = np.zeros(n, dtype=np.bool_)
        if len(ring_arr) < 4 or n == 0:
            return inside
        x = points[:, 0]
        y = points[:, 1]
        x1 = ring_arr[:-1, 0]
        y1 = ring_arr[:-1, 1]
        x2 = ring_arr[1:, 0]
        y2 = ring_arr[1:, 1]
        for i in range(len(x1)):
            crosses = (y1[i] > y) != (y2[i] > y)
            if not np.any(crosses):
                continue
            dy = y2[i] - y1[i]
            if abs(dy) < 1e-15:
                continue
            x_cross = x1[i] + (y - y1[i]) * (x2[i] - x1[i]) / dy
            inside ^= crosses & (x < x_cross)
        return inside

    def _collect_rings(parts):
        per_part = []
        all_rings = []
        for poly in parts:
            ext = _ring_array(poly.exterior)
            holes = [_ring_array(h) for h in poly.interiors]
            holes = [h for h in holes if len(h) >= 4]
            if len(ext) >= 4:
                per_part.append((ext, holes))
                all_rings.append(ext)
                all_rings.extend(holes)
        return per_part, all_rings

    def _inside_geometry(points, per_part):
        result = np.zeros(len(points), dtype=np.bool_)
        for ext, holes in per_part:
            here = _points_in_ring(points, ext)
            if holes and np.any(here):
                for hole in holes:
                    here &= ~_points_in_ring(points, hole)
            result |= here
        return result

    def _segments(all_rings):
        aa = []
        bb = []
        for ring in all_rings:
            if len(ring) >= 2:
                aa.append(ring[:-1])
                bb.append(ring[1:])
        if not aa:
            return (np.empty((0, 2), dtype=np.float64),
                    np.empty((0, 2), dtype=np.float64))
        return np.vstack(aa), np.vstack(bb)

    def _min_boundary_distance2(points, a, b, chunk=128):
        if len(points) == 0 or len(a) == 0:
            return np.full(len(points), np.inf, dtype=np.float64)
        ab = b - a
        denom = np.sum(ab * ab, axis=1)
        denom_safe = np.where(denom > 1e-18, denom, 1.0)
        out = np.empty(len(points), dtype=np.float64)
        for start in range(0, len(points), chunk):
            p = points[start:start + chunk]
            ap = p[:, None, :] - a[None, :, :]
            t = np.sum(ap * ab[None, :, :], axis=2) / denom_safe[None, :]
            t = np.clip(t, 0.0, 1.0)
            qx = a[None, :, 0] + t * ab[None, :, 0]
            qy = a[None, :, 1] + t * ab[None, :, 1]
            dx = p[:, None, 0] - qx
            dy = p[:, None, 1] - qy
            out[start:start + len(p)] = np.min(dx * dx + dy * dy, axis=1)
            del ap, t, qx, qy, dx, dy
        return out

    def _nearest_boundary_point(point, a, b):
        if len(a) == 0:
            return (float(point[0]) + 1e-6, float(point[1]))
        p = np.asarray(point, dtype=np.float64)
        ab = b - a
        denom = np.sum(ab * ab, axis=1)
        t = np.zeros(len(a), dtype=np.float64)
        good = denom > 1e-18
        if np.any(good):
            t[good] = np.sum((p - a[good]) * ab[good], axis=1) / denom[good]
        t = np.clip(t, 0.0, 1.0)
        q = a + ab * t[:, None]
        d2 = np.sum((q - p) ** 2, axis=1)
        best = q[int(np.argmin(d2))]
        return (float(best[0]), float(best[1]))

    def maximum_inscribed_circle(geom, tolerance=0.01):
        _compat_progress('inscribed center: reading polygon rings')
        parts = _polygon_parts(geom)
        if not parts:
            raise ValueError('Empty geometry')
        per_part, all_rings = _collect_rings(parts)
        if not per_part:
            raise ValueError('Geometry has no usable polygon rings')

        # Derive bounds from already-extracted coordinates rather than asking
        # GEOS for another geometry operation.
        exterior_points = np.vstack([ext[:-1] for ext, _ in per_part])
        minx = float(np.min(exterior_points[:, 0]))
        miny = float(np.min(exterior_points[:, 1]))
        maxx = float(np.max(exterior_points[:, 0]))
        maxy = float(np.max(exterior_points[:, 1]))
        spanx = max(maxx - minx, 1e-6)
        spany = max(maxy - miny, 1e-6)
        maxspan = max(spanx, spany)

        # Match the intent of Shapely's tolerance while bounding work on mobile.
        tol = max(float(tolerance), 0.04)
        target_step = max(tol, maxspan / 88.0)
        nx = int(np.clip(np.ceil(spanx / target_step) + 1, 9, 88))
        ny = int(np.clip(np.ceil(spany / target_step) + 1, 9, 88))
        xs = np.linspace(minx, maxx, nx, dtype=np.float64)
        ys = np.linspace(miny, maxy, ny, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        candidates = np.column_stack((gx.ravel(), gy.ravel()))
        del gx, gy, xs, ys

        _compat_progress(f'inscribed center: testing {len(candidates)} candidates')
        inside = _inside_geometry(candidates, per_part)
        candidates = candidates[inside]
        del inside

        if len(candidates) == 0:
            # Deterministic, allocation-light fallback from the exterior ring.
            ext = per_part[0][0][:-1]
            center = np.mean(ext, axis=0)
            candidates = center.reshape(1, 2)

        _compat_progress(f'inscribed center: measuring {len(candidates)} interior candidates')
        a, b = _segments(all_rings)
        d2 = _min_boundary_distance2(candidates, a, b)
        best_i = int(np.argmax(d2))
        center = candidates[best_i]
        del d2, candidates

        _compat_progress('inscribed center: choosing nearest boundary point')
        edge = _nearest_boundary_point(center, a, b)
        cx, cy = float(center[0]), float(center[1])
        ex, ey = edge
        if abs(ex - cx) + abs(ey - cy) < 1e-12:
            ex += 1e-6

        _compat_progress('inscribed center ready')
        return _CompatLine((cx, cy), (ex, ey))

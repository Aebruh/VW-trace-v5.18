"""Compatibility helpers for Chaquopy's Android Shapely 1.8 wheels.
Desktop Shapely 2.x behaviour is preserved when available.
"""
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


try:
    # Desktop / newer Shapely: keep the original native implementation.
    from shapely import maximum_inscribed_circle as _maximum_inscribed_circle

    def maximum_inscribed_circle(geom, tolerance=0.01):
        return _maximum_inscribed_circle(geom, tolerance=tolerance)
except Exception:
    # Chaquopy currently provides Shapely 1.8.x. Its polylabel fallback was
    # observed to hard-crash GEOS on Android for some otherwise valid residual
    # polygons. Use a small raster distance transform instead. The tracer only
    # consumes the first coordinate (the interior center seed), so this preserves
    # the intended meaning without invoking the unstable GEOS search.
    import cv2
    from shapely.geometry import LineString

    def _polygon_parts(geom):
        if geom is None or geom.is_empty:
            return []
        if geom.geom_type == 'Polygon':
            return [geom]
        return [g for g in getattr(geom, 'geoms', []) if g.geom_type == 'Polygon' and not g.is_empty]

    def _draw_ring(mask, ring, minx, miny, scale, pad, value):
        coords = np.asarray(ring.coords, dtype=np.float64)
        if len(coords) < 3:
            return
        pts = np.empty((len(coords), 2), dtype=np.int32)
        pts[:, 0] = np.rint((coords[:, 0] - minx) * scale).astype(np.int32) + pad
        pts[:, 1] = np.rint((coords[:, 1] - miny) * scale).astype(np.int32) + pad
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], int(value))

    def _nearest_on_ring(px, py, ring):
        coords = np.asarray(ring.coords, dtype=np.float64)
        if len(coords) < 2:
            return None, float('inf')
        a = coords[:-1]
        b = coords[1:]
        ab = b - a
        denom = np.sum(ab * ab, axis=1)
        p = np.asarray([px, py], dtype=np.float64)
        t = np.zeros(len(a), dtype=np.float64)
        good = denom > 1e-12
        if np.any(good):
            t[good] = np.sum((p - a[good]) * ab[good], axis=1) / denom[good]
        t = np.clip(t, 0.0, 1.0)
        q = a + ab * t[:, None]
        d2 = np.sum((q - p) ** 2, axis=1)
        i = int(np.argmin(d2))
        return q[i], float(d2[i])

    def _nearest_boundary(px, py, parts):
        best = None
        best_d2 = float('inf')
        for poly in parts:
            q, d2 = _nearest_on_ring(px, py, poly.exterior)
            if q is not None and d2 < best_d2:
                best, best_d2 = q, d2
            for hole in poly.interiors:
                q, d2 = _nearest_on_ring(px, py, hole)
                if q is not None and d2 < best_d2:
                    best, best_d2 = q, d2
        return best

    def maximum_inscribed_circle(geom, tolerance=0.01):
        parts = _polygon_parts(geom)
        if not parts:
            raise ValueError('Empty geometry')

        minx, miny, maxx, maxy = geom.bounds
        spanx = max(float(maxx - minx), 1e-6)
        spany = max(float(maxy - miny), 1e-6)
        maxspan = max(spanx, spany)

        # Nominally sample at half-pixel spacing, but cap the largest raster
        # dimension to ~768 so even a full 512 px region uses only a few MB.
        scale = min(2.0, 768.0 / maxspan)
        scale = max(scale, 0.5)
        pad = 3
        width = max(7, int(np.ceil(spanx * scale)) + 2 * pad + 1)
        height = max(7, int(np.ceil(spany * scale)) + 2 * pad + 1)
        mask = np.zeros((height, width), dtype=np.uint8)

        for poly in parts:
            _draw_ring(mask, poly.exterior, minx, miny, scale, pad, 255)
            for hole in poly.interiors:
                _draw_ring(mask, hole, minx, miny, scale, pad, 0)

        if not np.any(mask):
            # Extremely tiny geometry after quantization: use the mean exterior
            # coordinate as a deterministic seed rather than calling GEOS search.
            coords = np.asarray(parts[0].exterior.coords, dtype=np.float64)
            center = np.mean(coords[:-1], axis=0)
            cx, cy = float(center[0]), float(center[1])
        else:
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
            cx = float(minx + (ix - pad) / scale)
            cy = float(miny + (iy - pad) / scale)
            del dist

        edge = _nearest_boundary(cx, cy, parts)
        if edge is None:
            ex, ey = cx + 1e-6, cy
        else:
            ex, ey = float(edge[0]), float(edge[1])
            if abs(ex - cx) + abs(ey - cy) < 1e-12:
                ex += 1e-6

        return LineString([(cx, cy), (ex, ey)])

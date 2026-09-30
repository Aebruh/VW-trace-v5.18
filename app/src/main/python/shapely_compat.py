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
    from shapely import maximum_inscribed_circle as _maximum_inscribed_circle
    def maximum_inscribed_circle(geom, tolerance=0.01):
        return _maximum_inscribed_circle(geom, tolerance=tolerance)
except Exception:
    from shapely.ops import polylabel, nearest_points
    from shapely.geometry import LineString
    def maximum_inscribed_circle(geom, tolerance=0.01):
        # Shapely 2 returns a two-point line whose first point is the center.
        center = polylabel(geom, tolerance=max(float(tolerance), 1e-4))
        edge = nearest_points(center, geom.boundary)[1]
        return LineString([(center.x, center.y), (edge.x, edge.y)])

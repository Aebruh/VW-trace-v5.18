from __future__ import annotations

"""Low-memory Android adapter for the v5 VW vectorizer.

The original vectorizer intentionally keeps every connected-component crop in
memory until all regions have been sorted and processed. That's fine on desktop,
but on Android a detailed 512 px image can contain many overlapping component
bounding boxes, so those boolean crops can consume far more memory than the
source image itself.

This adapter does NOT change region detection, sorting, VW geometry, fill logic,
or object generation. It only stores each component mask bit-packed (1 bit per
pixel) and reconstructs one mask at a time immediately before calling the exact
original process_region_job function.
"""

import gc
import numpy as np
import cv2


def install_android_vectorize(namespace: dict) -> None:
    original_vectorize = namespace["vectorize"]
    process_region_job = namespace["process_region_job"]

    def vectorize_low_memory(labels, colors, background,
                             base_vw_area=46.0,
                             min_region=1,
                             rotation_threshold=30.0,
                             safety_margin=0.03,
                             workers=4):
        # Preserve the original desktop/multiprocessing path exactly. The Android
        # bridge explicitly requests workers=1, which selects the streamed path.
        workers_i = max(1, int(workers))
        if workers_i != 1:
            return original_vectorize(
                labels, colors, background,
                base_vw_area=base_vw_area,
                min_region=min_region,
                rotation_threshold=rotation_threshold,
                safety_margin=safety_margin,
                workers=workers_i,
            )

        # Same connected-component pass and same ordering as the original, but
        # retain each crop as packed bytes rather than a full bool ndarray.
        regions = []
        for ci in range(len(colors)):
            if ci == background:
                continue

            mask = (labels == ci).astype(np.uint8)
            n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

            for j in range(1, n):
                x, y, w, h, area = (int(v) for v in stats[j])
                if area < min_region:
                    continue

                component = (cc[y:y + h, x:x + w] == j)
                packed = np.packbits(component, axis=None).tobytes()
                regions.append((area, ci, x, y, w, h, packed))

            # These are large per-colour temporaries and are no longer needed.
            del mask, cc, stats
            gc.collect()

        # Identical stable order: large base masses first, details later.
        regions.sort(key=lambda r: r[0], reverse=True)
        total_regions = len(regions)
        detail_regions = sum(r[0] < 300 for r in regions)

        objects = []
        region_report = []
        cage_area_sum = 0.0
        covered_area_sum = 0.0

        for region_id in range(total_regions):
            area, ci, ox, oy, w, h, packed = regions[region_id]

            # Reconstruct exactly the same boolean crop the original code used.
            bits = np.frombuffer(packed, dtype=np.uint8)
            component = np.unpackbits(bits, count=w * h).reshape((h, w)).astype(np.bool_, copy=False)

            z_rank = region_id
            job = (
                area, ci, ox, oy, component, base_vw_area,
                rotation_threshold, safety_margin, region_id, z_rank,
            )
            result = process_region_job(job)

            if result is not None:
                local_objects, stat, poly_area, covered_area = result
                objects.extend(local_objects)
                region_report.append(stat)
                cage_area_sum += poly_area
                covered_area_sum += covered_area

            # Drop the packed mask and reconstructed ndarray as soon as this region
            # is complete. Replacing the list entry also releases its bytes early.
            regions[region_id] = None
            del component, bits, packed, job, result

            # GEOS/Shapely creates many short-lived Python wrappers during fitting.
            # Periodic collection prevents those wrappers piling up between regions.
            if region_id % 4 == 3:
                gc.collect()

        gc.collect()
        return objects, {
            "regions": total_regions,
            "detail_regions": detail_regions,
            "cage_area": round(cage_area_sum, 2),
            "covered_area": round(covered_area_sum, 2),
            "coverage_ratio": round(covered_area_sum / max(1.0, cage_area_sum), 5),
            "largest_region_stats": region_report[:25],
        }

    # Keep a reference for debugging/comparison and replace only the public entry.
    namespace["vectorize_desktop_original"] = original_vectorize
    namespace["vectorize"] = vectorize_low_memory

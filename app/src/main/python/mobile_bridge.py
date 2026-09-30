from __future__ import annotations

# Keep native numeric/image libraries from creating large worker pools on Android.
# These must be set before importing NumPy/OpenCV through the tracer modules.
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import gc
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

# IMPORTANT: only load the base tracer here. The v5.17/v5.18 optimizer stack is
# imported lazily after base VW vectorization has completed and its large image
# arrays have been released. This lowers Android peak RSS during the stage which
# was being killed on-device.
import gd_vw_tracer_v5 as v5

try:
    import cv2
    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
PROFILE_FILES = {
    'quality': ROOT / 'v5_18_quality_profile.json',
    'balanced': ROOT / 'v5_18_balanced_profile.json',
    'aggressive': ROOT / 'v5_18_aggressive_profile.json',
}


def _rss_mb():
    try:
        for line in Path('/proc/self/status').read_text(errors='ignore').splitlines():
            if line.startswith('VmRSS:'):
                return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return None


def _stage(work: Path, label: str):
    try:
        rss = _rss_mb()
        text = label if rss is None else f'{label} [RAM {rss} MB]'
        stage_file = work / 'crash_stage.txt'
        stage_file.write_text(text, encoding='utf8')
        # Let mobile_vectorize update the exact same persistent marker per region.
        os.environ['GDVW_STAGE_FILE'] = str(stage_file)
    except Exception:
        pass


def _call_main(module, *argv):
    old = sys.argv[:]
    sys.argv = [getattr(module, '__file__', module.__name__)] + [str(x) for x in argv]
    try:
        module.main()
    finally:
        sys.argv = old


def _build_base_scene(png: Path, work: Path):
    _stage(work, 'opening and resizing image')
    with Image.open(png) as opened:
        opened.thumbnail((512, 512), Image.Resampling.LANCZOS)
        source = opened.convert('RGBA')

    size = source.size

    _stage(work, 'building color palette')
    colors = v5.stable_palette(source, 25)

    _stage(work, 'segmenting image')
    labels, seeded = v5.edge_aware_labels(source, colors, 4)
    valid = labels[labels >= 0]
    if valid.size == 0:
        raise ValueError('The selected PNG has no visible pixels to trace.')
    background = int(np.bincount(valid, minlength=len(colors)).argmax())

    # The source bitmap, seeded mask and bincount selection are no longer needed
    # once labels are available. Drop them BEFORE Shapely vectorization starts.
    del valid, seeded, source
    gc.collect()

    _stage(work, 'vectorizing VW geometry')
    objects, stats = v5.vectorize(
        labels, colors, background,
        base_vw_area=46.0,
        rotation_threshold=30.0,
        safety_margin=0.03,
        workers=1,
    )

    scene = work / 'base_scene.json'
    cache = work / 'palette.npz'
    scene.write_text(json.dumps(objects), encoding='utf8')
    np.savez(cache,
             colors=np.asarray(colors, dtype=np.uint8),
             background=np.asarray([background], dtype=np.int16),
             size=np.asarray(size, dtype=np.int32))

    _stage(work, 'rendering base preview')
    base_preview = v5.render(size, colors, objects, background, antialias=2)
    base_preview.save(work / 'base_preview.png')
    base_count = len(objects)

    del base_preview, objects, labels
    gc.collect()
    return scene, cache, base_count, stats


def _quality(scene: Path, cache: Path, profile_path: Path, out: Path, work: Path):
    # Lazy import: none of these modules occupy memory during base vectorization.
    import optimize_v5_17
    import prune_v5_17
    import export_validate_v5_17
    import merge_v5_17_fallback

    cfg = json.loads(profile_path.read_text())
    cfg.setdefault('max_visible_error_per_merge', 4)
    cfg.setdefault('max_group_changed_fraction', .04)
    cfg.setdefault('min_group_changed_pixels', 4)
    cfg.setdefault('max_pair_distance', 8)
    cfg.setdefault('max_area_expansion_ratio', 100)
    cfg.setdefault('cross_layer_merges', True)
    cfg.setdefault('max_prune_error', 2)
    cfg.setdefault('max_total_prune_error', 60)
    cfg['protected_boxes_normalized'] = []
    effective = out / 'effective_profile.json'
    effective.write_text(json.dumps(cfg, indent=2))

    _stage(work, 'quality visibility optimization')
    visibility = out / 'visibility_objects.json'
    _call_main(optimize_v5_17, '--input', scene, '--reference', scene, '--cache', cache,
               '--profile', effective, '--output', visibility)
    protection = visibility.with_suffix('.protection.npy')

    _stage(work, 'quality merge pass')
    merged = out / 'merged_objects.json'
    from argparse import Namespace
    merge_v5_17_fallback.run(Namespace(
        protection=protection, input=visibility, cache=cache, reference=scene,
        profile=effective, output=merged))

    current = merged
    for iteration in range(1, 50):
        _stage(work, f'quality prune pass {iteration}')
        nxt = out / f'prune_{iteration}.json'
        _call_main(prune_v5_17, '--input', current, '--reference', scene, '--cache', cache,
                   '--profile', effective, '--protection', protection, '--output', nxt)
        before = len(json.loads(current.read_text()))
        after = len(json.loads(nxt.read_text()))
        current = nxt
        gc.collect()
        if before == after:
            break

    final = out / 'v5_17_final_objects.json'
    shutil.copyfile(current, final)
    _stage(work, 'quality export validation')
    _call_main(export_validate_v5_17, '--input', final, '--reference', scene, '--cache', cache,
               '--profile', effective, '--protection', protection, '--output-dir', out,
               '--name', 'v5_18_quality')
    return out / 'v5_18_quality.gmd', out / 'v5_18_quality_preview.png', out / 'v5_18_quality_report.json'


def _optimized(scene: Path, cache: Path, profile_path: Path, out: Path, tier: str, work: Path):
    # Lazy import: v5.18 optimizer is only loaded after the base scene exists.
    import optimize_v5_18
    import prune_v5_18
    import export_validate_v5_18

    cfg = json.loads(profile_path.read_text())
    fused = out / 'fused.json'

    _stage(work, f'{tier} fusion optimization')
    _call_main(optimize_v5_18, '--input', scene, '--reference', scene, '--cache', cache,
               '--profile', profile_path, '--output', fused)
    protection = fused.with_suffix('.protection.npy')
    importance = fused.with_suffix('.importance.npy')
    gc.collect()

    candidates = []
    for index, multiplier in enumerate(cfg.get('candidate_scales', [.8, 1, 1.2]), start=1):
        p = dict(cfg)
        p['normal_error_per_deletion'] = cfg.get('normal_error_per_deletion', 1.2) * multiplier
        folder = out / f'candidate_{index}'
        folder.mkdir(exist_ok=True)
        candidate_profile = folder / 'profile.json'
        candidate_profile.write_text(json.dumps(p, indent=2))
        current = fused

        for iteration in range(1, 50):
            _stage(work, f'{tier} candidate {index} prune pass {iteration}')
            nxt = folder / f'prune_{iteration}.json'
            _call_main(prune_v5_18, '--input', current, '--reference', scene, '--cache', cache,
                       '--profile', candidate_profile, '--protection', protection,
                       '--importance', importance, '--output', nxt)
            before = len(json.loads(current.read_text()))
            after = len(json.loads(nxt.read_text()))
            current = nxt
            gc.collect()
            if before == after:
                break

        name = 'v5_18_' + tier
        _stage(work, f'{tier} candidate {index} validation')
        _call_main(export_validate_v5_18, '--input', current, '--reference', scene, '--cache', cache,
                   '--profile', candidate_profile, '--protection', protection,
                   '--output-dir', folder, '--name', name)
        report = json.loads((folder / (name + '_report.json')).read_text())
        report.update(candidate=index, multiplier=multiplier, scene=str(current))
        candidates.append(report)
        gc.collect()

    passing = [r for r in candidates
               if r.get('passes_normal_threshold') and
               (r.get('protected_foreground_rgb_mae') or 0) < .5]
    selected = min(passing or candidates,
                   key=lambda r: (r['objects'], r['viewing_scale_metrics']['normal']['foreground_rgb_mae']))
    folder = out / f"candidate_{selected['candidate']}"
    name = 'v5_18_' + tier
    for suffix in ['.gmd', '_preview.png', '_report.json']:
        shutil.copyfile(folder / (name + suffix), out / (name + suffix))
    return out / (name + '.gmd'), out / (name + '_preview.png'), out / (name + '_report.json')


def trace(png_path: str, profile: str = 'balanced', work_dir: str | None = None):
    try:
        profile = profile.lower().strip()
        if profile not in PROFILE_FILES:
            raise ValueError('Unknown profile: ' + profile)

        work = Path(work_dir or (Path(png_path).parent / 'gd_vw_trace_work'))
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        os.environ['GDVW_STAGE_FILE'] = str(work / 'crash_stage.txt')
        _stage(work, 'starting trace')

        scene, cache, base_objects, stats = _build_base_scene(Path(png_path), work)
        gc.collect()

        out = work / 'result'
        out.mkdir(exist_ok=True)
        if profile == 'quality':
            gmd, preview, report_path = _quality(scene, cache, PROFILE_FILES[profile], out, work)
        else:
            gmd, preview, report_path = _optimized(scene, cache, PROFILE_FILES[profile], out, profile, work)

        _stage(work, 'reading final report')
        report = json.loads(report_path.read_text())
        result = {
            'ok': True,
            'profile': profile,
            'base_objects': base_objects,
            'objects': int(report['objects']),
            'reduction_percent': float(report.get('reduction_percent', 0.0)),
            'gmd': str(gmd),
            'preview': str(preview),
            'report': str(report_path),
            'regions': int(stats.get('regions', 0)),
        }
        _stage(work, 'COMPLETE')
        return json.dumps(result)
    except Exception as exc:
        try:
            work = Path(work_dir) if work_dir else Path(png_path).parent / 'gd_vw_trace_work'
            _stage(work, 'Python exception: ' + str(exc)[:160])
        except Exception:
            pass
        raise RuntimeError(str(exc) + '\n' + traceback.format_exc())

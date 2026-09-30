from __future__ import annotations
import json, shutil, sys, traceback
from pathlib import Path
import numpy as np
from PIL import Image

import gd_vw_tracer_v5 as v5
import optimize_v5_17, prune_v5_17, export_validate_v5_17
import merge_v5_17_fallback
import optimize_v5_18, prune_v5_18, export_validate_v5_18

ROOT = Path(__file__).resolve().parent
PROFILE_FILES = {
    'quality': ROOT / 'v5_18_quality_profile.json',
    'balanced': ROOT / 'v5_18_balanced_profile.json',
    'aggressive': ROOT / 'v5_18_aggressive_profile.json',
}

def _call_main(module, *argv):
    old = sys.argv[:]
    sys.argv = [getattr(module, '__file__', module.__name__)] + [str(x) for x in argv]
    try:
        module.main()
    finally:
        sys.argv = old

def _build_base_scene(png: Path, work: Path):
    source = Image.open(png).convert('RGBA')
    max_dimension = 512
    scale = min(1.0, max_dimension / max(source.size))
    size = tuple(max(1, int(round(v * scale))) for v in source.size)
    img = source.resize(size, Image.Resampling.LANCZOS) if size != source.size else source
    colors = v5.stable_palette(source, 25)
    labels, _seeded = v5.edge_aware_labels(img, colors, 4)
    valid = labels[labels >= 0]
    if valid.size == 0:
        raise ValueError('The selected PNG has no visible pixels to trace.')
    background = int(np.bincount(valid, minlength=len(colors)).argmax())
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
    base_preview = v5.render(size, colors, objects, background, antialias=2)
    base_preview.save(work / 'base_preview.png')
    return scene, cache, len(objects), stats

def _quality(scene: Path, cache: Path, profile_path: Path, out: Path):
    cfg = json.loads(profile_path.read_text())
    cfg.setdefault('max_visible_error_per_merge',4)
    cfg.setdefault('max_group_changed_fraction',.04)
    cfg.setdefault('min_group_changed_pixels',4)
    cfg.setdefault('max_pair_distance',8)
    cfg.setdefault('max_area_expansion_ratio',100)
    cfg.setdefault('cross_layer_merges',True)
    cfg.setdefault('max_prune_error',2)
    cfg.setdefault('max_total_prune_error',60)
    cfg['protected_boxes_normalized'] = []
    effective = out / 'effective_profile.json'
    effective.write_text(json.dumps(cfg, indent=2))

    visibility = out / 'visibility_objects.json'
    _call_main(optimize_v5_17, '--input',scene,'--reference',scene,'--cache',cache,'--profile',effective,'--output',visibility)
    protection = visibility.with_suffix('.protection.npy')
    merged = out / 'merged_objects.json'
    from argparse import Namespace
    merge_v5_17_fallback.run(Namespace(
        protection=protection, input=visibility, cache=cache, reference=scene,
        profile=effective, output=merged))
    current = merged
    for iteration in range(1, 50):
        nxt = out / f'prune_{iteration}.json'
        _call_main(prune_v5_17,'--input',current,'--reference',scene,'--cache',cache,
                   '--profile',effective,'--protection',protection,'--output',nxt)
        before = len(json.loads(current.read_text())); after = len(json.loads(nxt.read_text()))
        current = nxt
        if before == after:
            break
    final = out / 'v5_17_final_objects.json'
    shutil.copyfile(current, final)
    _call_main(export_validate_v5_17,'--input',final,'--reference',scene,'--cache',cache,
               '--profile',effective,'--protection',protection,'--output-dir',out,'--name','v5_18_quality')
    return out / 'v5_18_quality.gmd', out / 'v5_18_quality_preview.png', out / 'v5_18_quality_report.json'

def _optimized(scene: Path, cache: Path, profile_path: Path, out: Path, tier: str):
    cfg = json.loads(profile_path.read_text())
    fused = out / 'fused.json'
    _call_main(optimize_v5_18,'--input',scene,'--reference',scene,'--cache',cache,
               '--profile',profile_path,'--output',fused)
    protection = fused.with_suffix('.protection.npy')
    importance = fused.with_suffix('.importance.npy')
    candidates = []
    for index, multiplier in enumerate(cfg.get('candidate_scales',[.8,1,1.2]), start=1):
        p = dict(cfg)
        p['normal_error_per_deletion'] = cfg.get('normal_error_per_deletion',1.2) * multiplier
        folder = out / f'candidate_{index}'
        folder.mkdir(exist_ok=True)
        profile = folder / 'profile.json'
        profile.write_text(json.dumps(p, indent=2))
        current = fused
        for iteration in range(1, 50):
            nxt = folder / f'prune_{iteration}.json'
            _call_main(prune_v5_18,'--input',current,'--reference',scene,'--cache',cache,
                       '--profile',profile,'--protection',protection,'--importance',importance,'--output',nxt)
            before = len(json.loads(current.read_text())); after = len(json.loads(nxt.read_text()))
            current = nxt
            if before == after:
                break
        name = 'v5_18_' + tier
        _call_main(export_validate_v5_18,'--input',current,'--reference',scene,'--cache',cache,
                   '--profile',profile,'--protection',protection,'--output-dir',folder,'--name',name)
        report = json.loads((folder/(name+'_report.json')).read_text())
        report.update(candidate=index, multiplier=multiplier, scene=str(current))
        candidates.append(report)
    passing = [r for r in candidates if r.get('passes_normal_threshold') and (r.get('protected_foreground_rgb_mae') or 0) < .5]
    selected = min(passing or candidates, key=lambda r:(r['objects'],r['viewing_scale_metrics']['normal']['foreground_rgb_mae']))
    folder = out / f"candidate_{selected['candidate']}"
    name = 'v5_18_' + tier
    for suffix in ['.gmd','_preview.png','_report.json']:
        shutil.copyfile(folder/(name+suffix), out/(name+suffix))
    return out/(name+'.gmd'), out/(name+'_preview.png'), out/(name+'_report.json')

def trace(png_path: str, profile: str='balanced', work_dir: str|None=None):
    try:
        profile = profile.lower().strip()
        if profile not in PROFILE_FILES:
            raise ValueError('Unknown profile: ' + profile)
        work = Path(work_dir or (Path(png_path).parent / 'gd_vw_trace_work'))
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        scene, cache, base_objects, stats = _build_base_scene(Path(png_path), work)
        out = work / 'result'
        out.mkdir(exist_ok=True)
        if profile == 'quality':
            gmd, preview, report_path = _quality(scene, cache, PROFILE_FILES[profile], out)
        else:
            gmd, preview, report_path = _optimized(scene, cache, PROFILE_FILES[profile], out, profile)
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
        return json.dumps(result)
    except Exception as exc:
        raise RuntimeError(str(exc) + '\n' + traceback.format_exc())

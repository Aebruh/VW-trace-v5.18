from pathlib import Path
_parts = Path(__file__).resolve().parent / "gdvw_core_parts"
_src = "".join((_parts / f"part{i:02d}.txt").read_text(encoding="utf8") for i in range(5))
exec(compile(_src, str(_parts / "gd_vw_tracer_v5_core.py"), "exec"), globals(), globals())

# Android-only low-allocation adapter. This keeps the original v5.18 search,
# sizes, angles and scoring, while avoiding unnecessary temporary geometries.
try:
    from mobile_geometry_low_alloc import install_android_geometry
    install_android_geometry(globals())
except Exception:
    pass

# Keep connected-region masks bit-packed on Android and reconstruct one at a
# time. Desktop/multiworker calls still use the original vectorizer.
try:
    from mobile_vectorize import install_android_vectorize
    install_android_vectorize(globals())
except Exception:
    pass

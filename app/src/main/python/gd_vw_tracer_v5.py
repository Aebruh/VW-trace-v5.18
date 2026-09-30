from pathlib import Path
_parts = Path(__file__).resolve().parent / "gdvw_core_parts"
_src = "".join((_parts / f"part{i:02d}.txt").read_text(encoding="utf8") for i in range(5))
exec(compile(_src, str(_parts / "gd_vw_tracer_v5_core.py"), "exec"), globals(), globals())

# Android-only implementation adapters. The original v5 core above remains
# byte-for-byte unchanged. These replace only allocation-heavy plumbing while
# preserving the same VW search, rectangle sizes/angles and scoring.
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

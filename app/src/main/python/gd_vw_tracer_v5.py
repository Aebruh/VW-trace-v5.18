from pathlib import Path
_parts = Path(__file__).resolve().parent / "gdvw_core_parts"
_src = "".join((_parts / f"part{i:02d}.txt").read_text(encoding="utf8") for i in range(5))
exec(compile(_src, str(_parts / "gd_vw_tracer_v5_core.py"), "exec"), globals(), globals())

# Android memory adapter: keep the original v5 core byte-for-byte, but replace
# only the workers=1 vectorize entry with a streamed equivalent which packs
# connected-region masks and reconstructs them one at a time. Desktop/multiworker
# calls still use the original vectorizer.
try:
    from mobile_vectorize import install_android_vectorize
    install_android_vectorize(globals())
except Exception:
    pass

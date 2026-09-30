from pathlib import Path
_parts = Path(__file__).resolve().parent / "gdvw_core_parts"
_src = "".join((_parts / f"part{i:02d}.txt").read_text(encoding="utf8") for i in range(5))
exec(compile(_src, str(_parts / "gd_vw_tracer_v5_core.py"), "exec"), globals(), globals())

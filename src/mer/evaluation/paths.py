"""Find recorded run folders without changing historical artifacts."""
import json
from pathlib import Path

def locate_run(project_root, kind):
    """Prefer neutral folders; discover existing runs by their metadata."""
    if kind not in {"v1","multiseed_v3"}:
        raise ValueError("Unknown experiment kind")
    root=Path(project_root)
    preferred=root/"experiments"/kind
    marker="frozen_protocol.json" if kind=="v1" else "experiment_plan.json"
    def matches(folder):
        try:
            value=json.loads((folder/marker).read_text(encoding="utf-8"))
            if kind=="v1":
                return ("best_new" in value and "ensemble_runs" in value
                        and (folder/"deberta_s42/config.json").is_file())
            return value.get("seeds")==[42,123,2025] and value.get("new_seeds")==[123,2025]
        except (OSError,ValueError,TypeError):
            return False
    if matches(preferred):
        return preferred
    found=sorted({path.parent for path in root.glob("*/*/"+marker) if matches(path.parent)})
    if len(found)>1:
        raise ValueError("Multiple recorded runs found; provide an explicit run path")
    return found[0] if found else preferred

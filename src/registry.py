from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Language:
    name: str
    code: str
    input_hands: int
    classes: list[str]
    triton_model_name: str
    references_dir: Path
    notes: dict


def load_registry(root: Path = Path("languages")) -> dict[str, Language]:
    registry = {}
    for cfg_path in root.glob("*/config.yaml"):
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        lang_dir = cfg_path.parent
        registry[data["code"]] = Language(
            name=data["name"],
            code=data["code"],
            input_hands=data["input_hands"],
            classes=data["classes"],
            triton_model_name=data["triton_model_name"],
            references_dir=lang_dir / data["references_dir"],
            notes=data.get("notes", {}),
        )
    return registry

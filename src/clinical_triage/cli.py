"""Command-line entry point: the state of the project at a glance.

``uv run clinical-triage`` checks that the package is correctly installed and prints what is
available locally: configuration, dataset, trained models. It is the first command to run on a
fresh machine to know where things stand.
"""

from __future__ import annotations

import json

from clinical_triage import __version__
from clinical_triage.config import DATA, MODEL, PATHS, TRAINING, TRIAGE


def _state(path) -> str:
    """Say whether a pipeline artefact is on disk."""
    return "present" if path.exists() else "absent"


def _lora() -> str:
    """Describe the LoRA setting, the adapter's own if one exists.

    The values in ``config.py`` are only a starting point: fine-tuning follows the variant the
    tuning run kept. Printing the default here while an adapter sits on disk would describe
    something that was never trained.
    """
    default = f"r={TRAINING.lora_r}, alpha={TRAINING.lora_alpha}, lr={TRAINING.sft_lr} (default)"
    configuration = PATHS.sft_adapter / "adapter_config.json"
    if not configuration.exists():
        return default
    adapter = json.loads(configuration.read_text(encoding="utf-8"))
    return f"r={adapter['r']}, alpha={adapter['lora_alpha']} (supervised adapter on disk)"


def main() -> None:
    """Print a summary of the configuration and of the pipeline state."""
    lines = [
        ("Project version", __version__),
        ("Base model", MODEL.base_model),
        ("Triage levels", " · ".join(TRIAGE.levels)),
        (
            "Supervised set target",
            f"{DATA.target_sft_pairs} pairs, {DATA.french_share:.0%} in French",
        ),
        ("Preference set target", f"{DATA.target_dpo_pairs} pairs"),
        ("LoRA", _lora()),
        ("Project root", str(PATHS.root)),
    ]
    width = max(len(label) for label, _ in lines)
    print(f"Emergency triage assistant v{__version__}\n")
    for label, value in lines:
        print(f"  {label.ljust(width)} : {value}")

    print("\n  Pipeline state")
    metadata = PATHS.data_processed / "metadata.json"
    stages = [
        ("Dataset", metadata),
        ("SFT adapter", PATHS.sft_adapter),
        ("Merged SFT model", PATHS.sft_merged),
        ("DPO adapter", PATHS.dpo_adapter),
        ("Final merged model", PATHS.dpo_merged),
    ]
    stage_width = max(len(label) for label, _ in stages)
    for label, path in stages:
        print(f"    {label.ljust(stage_width)} : {_state(path)}")

    if metadata.exists():
        statistics = json.loads(metadata.read_text(encoding="utf-8"))["statistics"]
        print(
            f"\n  Dataset: {statistics['sft_total']} supervised examples, "
            f"{statistics['dpo_total']} preference pairs, "
            f"{statistics['clinical_evaluation']} clinical evaluation cases."
        )


if __name__ == "__main__":
    main()

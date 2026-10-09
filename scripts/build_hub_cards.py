"""Render the two Hugging Face Hub cards from their templates and the committed results.

=================================  ==================================================
``infra/hub/model-card.in.md``     ``infra/hub/model-card.md``
``infra/hub/dataset-card.in.md``   ``infra/hub/dataset-card.md``
=================================  ==================================================

A card is the first page anyone opening the weights reads, and it is published byte for byte:
what this script writes is what the Hub shows. No figure in it is typed by hand. Each one is a
placeholder that names the file it comes from and the key it reads::

    {{evaluation:clinical_set.models.dpo-merged.accuracy|f3}}

``evaluation`` designates ``reports/evaluation_results.json``; the path after the colon walks the
JSON, and the format after the bar writes the value (see ``FORMATS``). Paths joined by ``+`` add
up their values. A placeholder whose value
is missing stops the rendering: a card with a hole in it is not written.

Two placeholders are not figures. ``{{release:…}}`` gives the version and the addresses: the tag
the serving command pins and the repository the card points to, from ``pyproject.toml``, and the
two Hub repositories, from the project configuration.
``{{include:data/README.md}}`` inserts the data card, whose relative links are rewritten as
absolute addresses at the release tag: on the Hub, a link relative to a GitHub folder opens
nothing.

Usage::

    uv run python scripts/build_hub_cards.py           # render both cards
    uv run python scripts/build_hub_cards.py --check   # fail if a card is out of date
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json
import posixpath
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

from clinical_triage.config import MODEL, PATHS

CARDS = {
    Path("infra/hub/model-card.in.md"): Path("infra/hub/model-card.md"),
    Path("infra/hub/dataset-card.in.md"): Path("infra/hub/dataset-card.md"),
}

# The files a placeholder may read, by the name it uses for them.
SOURCES = {
    "evaluation": Path("reports/evaluation_results.json"),
    "seeds": Path("reports/seed_study.json"),
    "before_after": Path("reports/before_after.json"),
    "sft": Path("reports/training/sft.json"),
    "dpo": Path("reports/training/dpo.json"),
    "tuning": Path("reports/training/hyperparameter_comparison.json"),
    "data": Path("data/processed/metadata.json"),
}

PLACEHOLDER = re.compile(r"\{\{(\w+):([^|}]+)(?:\|(\w+))?\}\}")
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n+", re.DOTALL)
# A Markdown link or image whose target is a path in the repository, not an address.
RELATIVE_LINK = re.compile(r"(!?)\[([^\]]*)\]\((?!https?://|#|mailto:)([^)\s]+)\)")


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%".replace(".0%", "%")


def _p_value(value: float) -> str:
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def _size_category(value: int) -> str:
    """The Hub's size bucket for a number of rows."""
    for bound, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if value < bound:
            return label
    return "100K<n<1M"


FORMATS: dict[str, Callable] = {
    "raw": lambda value: json.dumps(value),
    "int": lambda value: f"{value:,}",
    "f2": lambda value: f"{value:.2f}",
    "f3": lambda value: f"{value:.3f}",
    "pct": _percent,
    "ci": lambda value: f"[{value[0]:.2f} – {value[1]:.2f}]",
    "ci_pct": lambda value: f"[{_percent(value[0])} – {_percent(value[1])}]",
    "p": _p_value,
    "minutes": lambda value: f"{value / 60:.0f} min",
    "gb": lambda value: f"{value:.1f} GB",
    "size": _size_category,
    "digests": lambda value: "\n".join(
        f"| `{name}` | `{digest}` |" for name, digest in value.items()
    ),
}


class MissingValue(LookupError):
    """A placeholder points to a value the results do not hold."""


def release(root: Path) -> dict[str, str]:
    """The version and the addresses every card derives from the project's configuration."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    repository = project["urls"]["Repository"].rstrip("/")
    return {
        "version": version,
        "tag": f"v{version}",
        "model_tag": f"model-v{version}",
        "repository": repository,
        "model_id": MODEL.hub_model_id,
        "dataset_id": MODEL.hub_dataset_id,
    }


def _walk(document, path: str):
    if "+" in path:
        return sum(_walk(document, part) for part in path.split("+"))
    value = document
    for key in path.split("."):
        if isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        elif isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise MissingValue(path)
    return value


def absolute_links(text: str, folder: str, release_: dict[str, str]) -> str:
    """Rewrite the links of a document of ``folder`` as addresses at the release tag.

    An image needs the file itself, a link the page that shows it: the first goes to the raw
    file, the second to GitHub's view of it.
    """
    owner_and_name = release_["repository"].removeprefix("https://github.com/")

    def rewrite(match: re.Match) -> str:
        bang, label, target = match.groups()
        path, _, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(folder, path))
        if bang:
            address = (
                f"https://raw.githubusercontent.com/{owner_and_name}/{release_['tag']}/{resolved}"
            )
        else:
            address = f"{release_['repository']}/blob/{release_['tag']}/{resolved}"
        return f"{bang}[{label}]({address}{'#' + anchor if anchor else ''})"

    return RELATIVE_LINK.sub(rewrite, text)


def render(template: str, root: Path) -> str:
    """Fill a template's placeholders; raise ``MissingValue`` on the first that has no value."""
    documents: dict[str, object] = {}
    release_ = release(root)

    def fill(match: re.Match) -> str:
        source, path, form = match.groups()
        if source == "release":
            if path not in release_:
                raise MissingValue(f"release:{path}")
            return release_[path]
        if source == "include":
            text = FRONT_MATTER.sub("", (root / path).read_text(encoding="utf-8"), count=1)
            return absolute_links(text.rstrip("\n"), posixpath.dirname(path), release_)
        if source not in SOURCES:
            raise MissingValue(f"{source}: no such source")
        if source not in documents:
            file = root / SOURCES[source]
            if not file.exists():
                raise MissingValue(f"{SOURCES[source].as_posix()} absent")
            documents[source] = json.loads(file.read_text(encoding="utf-8"))
        value = _walk(documents[source], path)
        return FORMATS[form or "raw"](value) if form or not isinstance(value, str) else value

    return PLACEHOLDER.sub(fill, template)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if a card is out of date")
    args = parser.parse_args()

    stale = []
    for source, target in CARDS.items():
        template = (PATHS.root / source).read_text(encoding="utf-8")
        try:
            card = render(template, PATHS.root)
        except MissingValue as missing:
            print(f"{source.as_posix()}: no value for {missing}; card not written.")
            return 1
        written = PATHS.root / target
        if args.check:
            current = written.read_text(encoding="utf-8") if written.exists() else None
            if current != card:
                stale.append(target.as_posix())
            continue
        written.write_text(card, encoding="utf-8", newline="\n")
        print(f"{target.as_posix()} rendered.")
    if stale:
        print(
            "Out of date: " + ", ".join(stale) + ". Run `python scripts/build_hub_cards.py` "
            "and commit what it writes."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

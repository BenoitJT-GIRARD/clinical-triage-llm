"""Take this repository's screenshots, and write down what each one shows.

A screenshot is the only image of a repository that cannot be regenerated from data. What
makes it checkable is the sentence next to it: which build was running, what had already
happened to the application, where the displayed data came from. This script takes the picture
and writes that sentence into ``docs/images/MANIFEST.json`` in the same gesture, because a
manifest filled in afterwards is filled in from memory.

The repository fills in :data:`CAPTURES` and :func:`prepare`, and nothing else. Each entry says
what the image must *prove*: « the API answers a prediction » names a surface, « a request with
a missing feature is refused with the field named » names a behaviour; :func:`prepare` holds the
commands that put the product into the state being photographed.

    uv run python scripts/capture.py                 # every capture
    uv run python scripts/capture.py --only api-docs
    uv run python scripts/capture.py --check         # take nothing, report what is stale

Two engines. Chrome headless is enough for a page that renders server-side or in one pass —
Swagger, Airflow, a static report — as long as ``--virtual-time-budget`` is given, without
which the picture is taken before the JavaScript has drawn anything. It is **not** enough for
a page whose content arrives over a websocket: a Streamlit page loads an empty skeleton and
fills it afterwards, and the virtual clock advances timers without waiting for that round
trip, so the capture comes out black however long the budget. Those pages go through
``playwright``, which waits for a selector that only exists once the content is there.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess  # nosec B404 - starts the product and the browser, both named below
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from clinical_triage.config import PATHS

#: This project answers the question of where things live in one module, and it is
#: ``config.py``: a second one would be a second answer.
ROOT_DIR = PATHS.root
IMAGES_DIR = PATHS.images

#: Logical size of every capture, and the density it is rendered at. One size for all of
#: them, so that two screenshots of this product can be read side by side.
VIEWPORT = (1280, 1000)
DEVICE_SCALE_FACTOR = 2

MANIFEST = IMAGES_DIR / "MANIFEST.json"
SOURCE = "scripts/capture.py"


@dataclass(frozen=True)
class Capture:
    """One image, and everything a reader needs to believe it.

    ``app_state`` is described precisely enough to be reproduced: « after 300 scoring
    requests, two of them refused », and never « with data ». ``data_source`` names where what
    is displayed comes from; a capture never redistributes someone else's work, so a
    third-party source is refused outright.
    """

    name: str
    route: str
    app_state: str
    demonstrates_behaviour: str
    data_source: str
    engine: str = "chrome"
    #: A selector that exists only once the content has arrived. Required by the playwright
    #: engine, ignored by Chrome.
    ready_selector: str | None = None
    #: What to do before the picture, in order. Each step is ``(action, target, value)``:
    #: ``("click", role, accessible name)``, ``("open", css selector, "")``,
    #: ``("fill", css selector, text)``,
    #: ``("scroll", css selector, "")``. A capture of an answer needs the three: click the
    #: control, type a real question, bring the response into the frame.
    steps: tuple[tuple[str, str, str], ...] = ()
    #: The twin image, when the product has a nominal and a degraded behaviour. A refusal
    #: shown alone reads as a failure; shown next to the nominal answer it reads as a design.
    paired_with: str | None = None
    depends_on: tuple[str, ...] = ()
    #: A surface of this repository that is NOT the one ``SERVE_COMMAND`` starts — an
    #: orchestrator's web interface, a database console, a second service of the same
    #: compose file. ``served_by`` is the command a reader runs to bring it up, and the
    #: capture is skipped, loudly, when nothing answers there: a picture is worth taking
    #: only of something that is running.
    base_url: str | None = None
    served_by: str | None = None

    @property
    def target(self) -> str:
        return f"{self.base_url or BASE_URL}{self.route}"

    @property
    def is_second_surface(self) -> bool:
        return self.base_url is not None

    @property
    def path(self) -> Path:
        return IMAGES_DIR / f"{self.name}.png"


#: Where the service listens once started, and the command that starts it. Both are quoted in
#: the README's « Running it » section: a capture taken against a service started some other
#: way proves something about that other way.
#: A port of the capture's own, and not the 8000 of the local stack. Whatever is already
#: listening there gets photographed in place of the product, and on a workstation that runs
#: a container relay something usually is.
CAPTURE_PORT = "8010"
BASE_URL = f"http://127.0.0.1:{CAPTURE_PORT}"
SERVE_COMMAND: tuple[str, ...] = (
    sys.executable,
    "-m",
    "uvicorn",
    "clinical_triage.serving.api:app",
    "--host",
    "127.0.0.1",
    "--port",
    CAPTURE_PORT,
    "--log-level",
    "warning",
)
#: The route that answers once the product is ready. ``None`` when there is nothing to start
#: and nothing to wait for — a page served from the filesystem.
#:
#: The contract, and not `/health`: this service's probe questions the inference engine
#: before answering, so it measures the engine's readiness and not the gateway's. Waiting on
#: it would wait for something no capture needs.
HEALTH_ROUTE: str | None = "/openapi.json"

#: The key the captured service demands. It is a capture key and nothing else: the service
#: refuses to start without one, and photographing it is the point — the reader sees the
#: header a caller has to present.
CAPTURE_KEY = "capture-key"

#: The complaint typed into the questionnaire, chosen because the explicit rule answers it on
#: its own: no model, no GPU, and an answer that is entirely this repository's logic.
COMPLAINT = "gêne dans la poitrine à l'effort"

#: The repository's captures. Nothing else in this file changes from one repository to
#: the next.
CAPTURES: tuple[Capture, ...] = (
    Capture(
        name="api-contract",
        route="/docs",
        app_state=(
            "the gateway started by SERVE_COMMAND with a capture key, and no inference engine "
            "behind it: the page is the contract, which does not depend on the engine"
        ),
        demonstrates_behaviour=(
            "the three routes and the X-API-Key scheme are generated from the Pydantic models, "
            "so the page a caller reads cannot drift from what the service enforces"
        ),
        data_source="none: the page is generated from the Pydantic models",
        depends_on=(
            "src/clinical_triage/serving/api.py",
            "src/clinical_triage/serving/schemas.py",
        ),
    ),
    Capture(
        name="triage-contract",
        route="/docs#/default/triage_triage_post",
        engine="playwright",
        steps=(
            ("open", "#operations-default-triage_triage_post", ""),
            ("click", "button", "Schema"),
            ("scroll", "#operations-default-triage_triage_post", ""),
        ),
        app_state="the same gateway, with the triage operation expanded on its reply schema",
        demonstrates_behaviour=(
            "the reply carries `rule_level`, `rule_reasons` and `agreement` beside the "
            "model's own answer: a caller cannot read the decision without seeing whether "
            "the explicit rule agreed with it"
        ),
        data_source="none: the schema is generated from the Pydantic models",
        depends_on=("src/clinical_triage/serving/schemas.py",),
    ),
    Capture(
        name="questionnaire-answer",
        route="/docs#/default/questionnaire_next_questionnaire_next_post",
        engine="playwright",
        steps=(
            # The modal's own buttons, named by CSS: with the dialog open, « Authorize »
            # matches twice — the page's button, behind the backdrop, and the modal's submit.
            ("click", "button", "Authorize"),
            ("fill", ".auth-container input[type='text']", CAPTURE_KEY),
            ("open", ".auth-btn-wrapper button.authorize", ""),
            ("open", ".auth-btn-wrapper button.btn-done", ""),
            ("open", "#operations-default-questionnaire_next_questionnaire_next_post", ""),
            ("click", "button", "Try it out"),
            (
                "fill",
                "#operations-default-questionnaire_next_questionnaire_next_post textarea",
                json.dumps(
                    {
                        "chief_complaint": COMPLAINT,
                        "answers": {
                            "conscience": "oui",
                            "respiration": "non",
                            "saignement": "non",
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
            ("click", "button", "Execute"),
            (
                "scroll",
                (
                    "#operations-default-questionnaire_next_questionnaire_next_post "
                    ".responses-wrapper"
                ),
                "",
            ),
        ),
        ready_selector=".responses-table .response-col_description pre",
        paired_with="triage-contract",
        app_state=(
            "the same gateway, authorised with the capture key, after one call to "
            "POST /questionnaire/next carrying a chest-discomfort complaint and the three "
            "red-flag answers"
        ),
        demonstrates_behaviour=(
            "the questionnaire adapts to the complaint instead of reading a fixed list: the "
            "chest theme is detected and the next question asked is the one about radiating "
            "pain. Nothing here goes through the model — the answer is the explicit rule's, "
            "computed in the gateway"
        ),
        data_source=(
            "the questionnaire plan of src/clinical_triage/serving/questionnaire.py, written "
            "for this project"
        ),
        depends_on=(
            "src/clinical_triage/serving/questionnaire.py",
            "src/clinical_triage/data/triage_rules.py",
        ),
    ),
)


def prepare() -> None:
    """Put the product into the state the captures need, before it is started.

    A picture of an empty product proves nothing, and a state built by hand in a terminal is
    a state nobody can reproduce. Whatever a capture depends on — a seeded database, a built
    index, a run of the pipeline — is commanded here, so that the image and the state behind
    it are written down in the same file.

    Here it is the service's own configuration. The gateway refuses to start without a key,
    and it writes an audit line for every triage: both are set to values that belong to the
    capture and to nothing else, so that no key of the author's and no log of theirs can end
    up in an image. The engine is left unreachable on purpose — none of the three captures
    needs it, and pointing at a live one would photograph a machine nobody else has.
    """
    os.environ.setdefault("TRIAGE_BACKEND", "vllm")
    os.environ["TRIAGE_API_KEY"] = CAPTURE_KEY
    os.environ["TRIAGE_VLLM_URL"] = "http://127.0.0.1:1"
    os.environ["TRIAGE_AUDIT_LOG"] = str(ROOT_DIR / "var" / "logs" / "capture.jsonl")


# --- Starting the product, and knowing when it is up ------------------------


def _answers(url: str) -> bool:
    """Whether something is already serving that URL, right now."""
    try:
        # nosec B310 - the URL is composed from BASE_URL and a route of this file
        with urllib.request.urlopen(url, timeout=1) as answer:  # nosec B310
            return answer.status < 500
    except (urllib.error.URLError, OSError):
        return False


def wait_until_healthy(url: str, *, timeout: float = 90.0) -> None:
    """Poll until the service answers. Never sleep a fixed number of seconds.

    A fixed sleep is either too short on a cold start, and the capture photographs a
    connection error, or wasted on every run afterwards.
    """
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            # nosec B310 - same URL, composed from this file's own constants
            with urllib.request.urlopen(url, timeout=2) as answer:  # nosec B310
                if answer.status < 500:
                    return
        except (urllib.error.URLError, OSError) as exc:  # not up yet
            last = exc
        time.sleep(0.25)
    raise TimeoutError(f"{url} never answered in {timeout:.0f}s ({last})")


class Serving:
    """Start the product, wait for it, capture, stop it — even when a capture raises."""

    def __init__(self, command: tuple[str, ...], health: str | None):
        self.command = command
        self.health = health
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> Self:
        if self.health is None:
            return self
        if not self.command:
            wait_until_healthy(self.health, timeout=5)
            return self
        # Something already answering on that port gets photographed in place of the
        # product: a server left over from an earlier run serves an older build, and its
        # picture is indistinguishable from a fresh one.
        if _answers(self.health):
            raise RuntimeError(
                f"{self.health} already answers: stop what is listening before capturing, "
                "or the picture will be of that and not of this build"
            )
        self.process = subprocess.Popen(  # nosec B603 - SERVE_COMMAND, a literal of this file
            list(self.command), cwd=ROOT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        wait_until_healthy(self.health)
        return self

    def __exit__(self, *_exception) -> None:
        """Stop the whole tree. `uv run uvicorn` is two processes, and killing the first
        leaves the second holding the port for the next run."""
        if self.process is None:
            return
        if os.name == "nt":
            subprocess.run(  # nosec B603 B607 - taskkill from the PATH, on our own child
                ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()


# --- The two engines --------------------------------------------------------


def _chrome_binary() -> str:
    """The browser on this machine, named by the environment and never guessed.

    A path hard-coded here would be one machine's installation shipped inside a published
    repository; ``CHROME_PATH`` keeps that constraint where it belongs.
    """
    explicit = os.environ.get("CHROME_PATH")
    if explicit:
        return explicit
    for name in ("chrome", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "no Chrome on PATH: set CHROME_PATH to the browser's executable, or run the "
        "capture with --engine playwright"
    )


def by_chrome(capture: Capture) -> None:
    """One pass, headless. `--virtual-time-budget` is what makes a JavaScript page render."""
    width, height = VIEWPORT
    subprocess.run(  # nosec B603 - the browser named by CHROME_PATH, flags from this file
        [
            _chrome_binary(),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--virtual-time-budget=8000",
            f"--window-size={width},{height}",
            f"--force-device-scale-factor={DEVICE_SCALE_FACTOR}",
            f"--screenshot={capture.path}",
            capture.target,
        ],
        check=True,
        cwd=ROOT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def by_playwright(capture: Capture) -> None:
    """For a page whose content arrives over a websocket, and that Chrome photographs black.

    ``channel="chrome"`` reuses the system browser: no download, and the picture is taken by
    the same engine a reader would open the page with.
    """
    from playwright.sync_api import sync_playwright  # installed in the capture environment

    width, height = VIEWPORT
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        page = browser.new_page(
            viewport={"width": width, "height": height},
            device_scale_factor=DEVICE_SCALE_FACTOR,
        )
        page.goto(capture.target, wait_until="networkidle", timeout=90_000)
        for action, target, value in capture.steps:
            if action == "click":
                # `.first`: a Swagger operation carries a title button and an arrow button
                # under the same accessible name, and the first in document order is the one
                # a reader sees and clicks.
                page.get_by_role(target, name=value).first.click()
            elif action == "open":
                # A control named by a CSS selector rather than by an accessible name. A
                # Swagger operation reached through a deep link is not always expanded by the
                # time the page settles, and its « Try it out » button is not in the DOM
                # until it is: clicking the operation's own header is what puts it there.
                page.locator(target).first.click()
            elif action == "fill":
                page.locator(target).first.fill(value)
            elif action == "scroll":
                page.locator(target).first.scroll_into_view_if_needed()
            else:
                raise ValueError(f"{capture.name}: unknown capture step « {action} »")
            page.wait_for_timeout(300)
        if capture.ready_selector:
            page.wait_for_selector(capture.ready_selector, timeout=180_000)
        page.wait_for_timeout(4000)  # let the animations settle
        page.screenshot(path=str(capture.path))
        browser.close()


ENGINES = {"chrome": by_chrome, "playwright": by_playwright}


# --- Writing down what was photographed -------------------------------------


def _git_revision() -> str | None:
    try:
        done = subprocess.run(  # nosec B603 B607 - git from the PATH, one read-only command
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


def record(capture: Capture) -> dict:
    """The manifest entry for an image that has just been written."""
    if capture.data_source == "third_party":
        raise ValueError(
            f"{capture.name}: a capture does not redistribute someone else's work. "
            "Photograph the product against data this repository may publish."
        )
    entry = {
        "sha256": hashlib.sha256(capture.path.read_bytes()).hexdigest(),
        "written": datetime.now(tz=UTC).date().isoformat(),
        "source": SOURCE,
        "command": f"uv run python {SOURCE} --only {capture.name}",
        "target": capture.target,
        "viewport": list(VIEWPORT),
        "device_scale_factor": DEVICE_SCALE_FACTOR,
        "app_state": capture.app_state,
        "data_source": capture.data_source,
        "demonstrates_behaviour": capture.demonstrates_behaviour,
    }
    revision = _git_revision()
    if revision:
        entry["git_revision"] = revision
    if capture.paired_with:
        entry["paired_with"] = f"{capture.paired_with}.png"
    if capture.depends_on:
        entry["depends_on"] = list(capture.depends_on)
    return entry


def write_manifest(entries: dict[str, dict]) -> None:
    """Merge into the manifest. An image nobody re-took keeps the entry it had."""
    payload: dict = {"schema": "image-manifest/1", "images": {}}
    if MANIFEST.exists():
        with contextlib.suppress(json.JSONDecodeError):
            payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload.setdefault("schema", "image-manifest/1")
    payload.setdefault("images", {})
    payload["images"].update(entries)
    payload["images"] = dict(sorted(payload["images"].items()))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline=""
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="a single capture, by name")
    parser.add_argument(
        "--check", action="store_true", help="take nothing; report what the manifest is missing"
    )
    arguments = parser.parse_args(argv)

    wanted = [c for c in CAPTURES if not arguments.only or c.name == arguments.only]
    if not wanted:
        print("no capture selected", file=sys.stderr)
        return 1

    if arguments.check:
        known = {}
        if MANIFEST.exists():
            known = json.loads(MANIFEST.read_text(encoding="utf-8")).get("images", {})
        stale = [
            c.name
            for c in wanted
            if not c.path.exists()
            or known.get(c.path.name, {}).get("sha256")
            != hashlib.sha256(c.path.read_bytes()).hexdigest()
        ]
        for name in stale:
            print(f"  {name} is missing or does not match its manifest entry")
        return 1 if stale else 0

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    prepare()
    entries: dict[str, dict] = {}
    health = f"{BASE_URL}{HEALTH_ROUTE}" if HEALTH_ROUTE else None
    with Serving(SERVE_COMMAND, health):
        for capture in wanted:
            if capture.is_second_surface and not _answers(capture.target):
                print(
                    f"{capture.name:24} skipped: nothing answers {capture.target}. "
                    f"Start it with: {capture.served_by}",
                    file=sys.stderr,
                )
                continue
            ENGINES[capture.engine](capture)
            entries[capture.path.name] = record(capture)
            print(f"{capture.name:24} {capture.path.relative_to(ROOT_DIR)}")
    write_manifest(entries)
    print(f"{len(entries)} capture(s), {MANIFEST.relative_to(ROOT_DIR)} updated")
    print("Read every image before committing it: no key, no token, no address on screen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

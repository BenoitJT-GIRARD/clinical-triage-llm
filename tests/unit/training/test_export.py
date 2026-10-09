"""Tests of the export check the merge and the publication share.

The weight files are replaced by their index: what is checked is the reading of the
configuration and of the tensor names, not a real model.
"""

from __future__ import annotations

import hashlib
import json

from clinical_triage.training.export import export_problems, weight_digests


def _export(folder, tied: bool = False, tensors=("model.embed_tokens.weight", "lm_head.weight")):
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"tie_word_embeddings": tied}), encoding="utf-8")
    index = {"weight_map": dict.fromkeys(tensors, "model.safetensors")}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return folder


def test_an_untied_export_with_its_output_head_passes(tmp_path):
    assert export_problems(_export(tmp_path / "merged")) == []


def test_an_export_whose_weights_are_tied_again_is_refused(tmp_path):
    problems = export_problems(_export(tmp_path / "merged", tied=True))
    assert any("tied" in problem for problem in problems)


def test_an_export_without_its_output_head_is_refused(tmp_path):
    problems = export_problems(_export(tmp_path / "merged", tensors=("model.embed_tokens.weight",)))
    assert any("lm_head.weight" in problem for problem in problems)


def test_a_folder_with_no_weights_is_refused(tmp_path):
    folder = tmp_path / "merged"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert export_problems(folder) == ["merged: no weight file"]


def test_the_digests_cover_every_weight_file_and_nothing_else(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert weight_digests(tmp_path) == {
        "model-00001-of-00002.safetensors": hashlib.sha256(b"first").hexdigest(),
        "model-00002-of-00002.safetensors": hashlib.sha256(b"second").hexdigest(),
    }

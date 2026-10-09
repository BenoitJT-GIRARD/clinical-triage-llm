"""The files the dataset build writes, byte for byte.

They go to the Hub as they are on disk: the same build must give the same bytes on every platform.
"""

from __future__ import annotations

from clinical_triage.data.dataset_io import read_jsonl, write_jsonl, write_metadata


def test_the_data_files_end_their_lines_in_lf_on_every_platform(tmp_path):
    rows = [{"level": "URGENCE_VITALE", "lang": "fr"}, {"level": "URGENCE_MODEREE", "lang": "en"}]
    write_jsonl(rows, tmp_path / "set.jsonl")
    write_metadata({"statistics": {"n": 2}}, tmp_path / "metadata.json")
    for name in ("set.jsonl", "metadata.json"):
        written = (tmp_path / name).read_bytes()
        assert b"\r\n" not in written, name
        assert written.endswith(b"\n"), name
    assert read_jsonl(tmp_path / "set.jsonl") == rows

"""Tests for NUS-48E label parsing and chunking."""
import numpy as np
import pytest
from autolyrics.data.nus48e import parse_nus_label, chunk_by_silence, build_lyrics_map


def _make_segments():
    """Minimal phone sequence: sil, 3 phones, sp, 3 phones, sil."""
    return [
        (0.0, 0.5, "sil"),
        (0.5, 1.0, "AH"),
        (1.0, 1.5, "N"),
        (1.5, 2.0, "D"),
        (2.0, 2.1, "sp"),
        (2.1, 2.6, "AY"),
        (2.6, 3.0, "AE"),
        (3.0, 3.5, "M"),
        (3.5, 4.0, "sil"),
    ]


def test_parse_nus_label_roundtrip(tmp_path):
    label_file = tmp_path / "test.txt"
    label_file.write_text("0.0\t0.5\tsil\n0.5\t1.0\tAH\n1.0\t1.5\tN\n")
    result = parse_nus_label(label_file)
    assert len(result) == 3
    assert result[0] == (0.0, 0.5, "sil")
    assert result[2] == (1.0, 1.5, "N")


def test_parse_nus_label_skips_malformed(tmp_path):
    label_file = tmp_path / "bad.txt"
    label_file.write_text("0.0\t0.5\tsil\nbad_line\n1.0\t1.5\tAH\n")
    result = parse_nus_label(label_file)
    assert len(result) == 2


def test_chunk_by_silence_basic():
    """Short audio should produce chunks above min length."""
    sr = 16_000
    duration_s = 30.0
    audio = np.zeros(int(duration_s * sr), dtype="float32")
    segments = _make_segments()
    lyric_words = ["and", "i", "am"]

    chunks = chunk_by_silence(
        audio, segments, lyric_words, sr=sr,
        target_s=2.0, max_s=5.0, min_s=0.5,
    )
    assert isinstance(chunks, list)
    for ch in chunks:
        assert "audio" in ch and "text" in ch
        assert len(ch["audio"]) / sr >= 0.5


def test_chunk_determinism():
    """Same inputs must produce identical chunks every run."""
    sr = 16_000
    audio = np.random.default_rng(0).random(sr * 30).astype("float32")
    segments = _make_segments()
    lyric_words = ["hello", "world", "test"]

    c1 = chunk_by_silence(audio, segments, lyric_words, sr=sr)
    c2 = chunk_by_silence(audio, segments, lyric_words, sr=sr)
    assert len(c1) == len(c2)
    for a, b in zip(c1, c2):
        assert a["text"] == b["text"]
        np.testing.assert_array_equal(a["audio"], b["audio"])


def test_build_lyrics_map(tmp_path):
    (tmp_path / "01.txt").write_text("hello world foo bar")
    (tmp_path / "02.txt").write_text("one two three")
    m = build_lyrics_map(tmp_path)
    assert m["01"] == ["hello", "world", "foo", "bar"]
    assert m["02"] == ["one", "two", "three"]

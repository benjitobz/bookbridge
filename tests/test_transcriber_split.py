"""AudioTranscriber.split_audio_file: chunk content and seek cost.

The split placed `-ss` after `-i` (output seeking), so ffmpeg decoded the whole
WAV up to each chunk's start: on a 22h book the per-chunk time grew linearly
from 0.7s to 16.7s, 4.4 minutes for 30 chunks. Input seeking is sample-exact on
PCM WAV, so the chunks must stay identical -- Whisper's timestamps are offset by
each chunk's start, and a shifted chunk would shift every transcript segment.
"""
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils.polisher import Polisher
from src.utils.transcriber import AudioTranscriber


def _make_wav(path: Path, seconds: float) -> None:
    """A 16 kHz mono PCM WAV whose samples are all distinct over short spans
    (a sine sweep), so a chunk shifted by even one sample would not match."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-f", "lavfi", "-i", f"aevalsrc=sin(2*PI*(200+300*t)*t):s=16000:d={seconds}",
            "-ac", "1", "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
    )


def _frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "ffmpeg/ffprobe not on PATH -- these split real audio",
)
class TestSplitAudioFile(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.transcriber = AudioTranscriber(self.tmp, MagicMock(), Polisher())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_chunks_concatenate_to_the_original_samples(self) -> None:
        source = self.tmp / "book_normalized.wav"
        _make_wav(source, 9.0)
        original = _frames(source)

        chunks = self.transcriber.split_audio_file(source, target_max_duration_sec=4)

        self.assertEqual(len(chunks), 3)
        joined = b"".join(_frames(chunk) for chunk in chunks)
        # Sample-exact: same length and same bytes, chunk seams included.
        self.assertEqual(len(joined), len(original))
        self.assertEqual(joined, original)

    def test_each_chunk_seeks_the_input_instead_of_decoding_up_to_it(self) -> None:
        source = self.tmp / "book_normalized.wav"
        _make_wav(source, 9.0)
        commands = []
        real_run = subprocess.run

        def record(cmd, *args, **kwargs):
            commands.append(cmd)
            return real_run(cmd, *args, **kwargs)

        with patch("src.utils.transcriber.subprocess.run", side_effect=record):
            self.transcriber.split_audio_file(source, target_max_duration_sec=4)

        split_commands = [c for c in commands if "-ss" in c]
        self.assertEqual(len(split_commands), 3)
        for cmd in split_commands:
            self.assertLess(cmd.index("-ss"), cmd.index("-i"), cmd)


if __name__ == "__main__":
    unittest.main()

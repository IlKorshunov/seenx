import io
import select
import subprocess as sp
import sys
from pydub import AudioSegment
from ...utils.logger import Logger
from .consts import *


logger = Logger(show=True).get_logger()


def mp4_to_wav(input_path, output_path):
    sp.run(["ffmpeg", "-y", "-i", str(input_path), "-ab", "160k", "-ac", "2", "-ar", str(DEMUCS_AUDIO_SR), "-vn", str(output_path)], check=True)


def _pipe_streams(process):
    def raw(s):
        assert s is not None
        return s.raw if isinstance(s, io.BufferedIOBase) else s

    fds = {raw(process.stdout).fileno(): (raw(process.stdout), sys.stdout), raw(process.stderr).fileno(): (raw(process.stderr), sys.stderr)}
    active = list(fds.keys())
    while active:
        for fd in select.select(active, [], [])[0]:
            buf = fds[fd][0].read(2**16)
            if not buf:
                active.remove(fd)
                continue
            fds[fd][1].write(buf.decode())
            fds[fd][1].flush()


def separate(files, outp, device=None, segment=None):
    cmd = ["python3", "-m", "demucs.separate", "-o", str(outp), "-n", DEMUCS_MODEL]
    if device:
        cmd += ["-d", device]
    if segment is not None:
        try:
            segment_value = float(segment)
        except (TypeError, ValueError):
            segment_value = 0.0
        if segment_value > 0:
            if segment_value > DEMUCS_MAX_SEGMENT:
                logger.warning("demucs_segment=%.3f is too high for %s; clamping to %.1f", segment_value, DEMUCS_MODEL, DEMUCS_MAX_SEGMENT)
                segment_value = DEMUCS_MAX_SEGMENT
            cmd += ["--segment", f"{segment_value:.1f}"]
    if DEMUCS_MP3:
        cmd += ["--mp3", f"--mp3-bitrate={DEMUCS_MP3_RATE}"]
    if DEMUCS_FLOAT32:
        cmd += ["--float32"]
    if DEMUCS_INT24:
        cmd += ["--int24"]
    if DEMUCS_TWO_STEMS is not None:
        cmd += [f"--two-stems={DEMUCS_TWO_STEMS}"]
    logger.info("With command: %s", " ".join(cmd + files))
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    _pipe_streams(p)
    p.wait()
    if p.returncode != 0:
        logger.warning("Demucs failed (exit %d). Likely CUDA OOM on long audio. Try demucs_device: 'cpu' or smaller demucs_segment in config.", p.returncode)
        return False
    return True


def combine(folder):
    vocal_path = f"{folder}/{STEM_VOCALS}"
    audio1 = AudioSegment.from_mp3(f"{folder}/{STEM_OTHER}")
    audio2 = AudioSegment.from_mp3(f"{folder}/{STEM_DRUMS}")
    audio3 = AudioSegment.from_mp3(f"{folder}/{STEM_BASS}")
    audio = audio1.overlay(audio2).overlay(audio3)
    music_path = f"{folder}/{STEM_MIXED}"
    audio.export(music_path, format="mp3")
    return music_path, vocal_path

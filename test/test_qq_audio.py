"""QQ 通话音频分句与格式转换测试。"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from ..voice_call.qq_audio import PCMUtteranceSegmenter, encode_wav


def make_pcm(milliseconds: int, amplitude: int = 0) -> bytes:
    """生成固定采样率的受控 PCM。"""
    return np.full(milliseconds * 16, amplitude, dtype="<i2").tobytes()


def test_silence_is_not_an_utterance() -> None:
    """连续静音不产生任何可提交语句。"""
    segmenter = PCMUtteranceSegmenter(0.008, 100, 1000, 40)

    assert segmenter.feed(make_pcm(1200)) == []
    assert segmenter.finish() == b""


def test_voice_ends_at_silence_and_retains_bytes() -> None:
    """语音后的静音端点产生完整音频，不丢弃起始声音。"""
    segmenter = PCMUtteranceSegmenter(0.008, 100, 1000, 40)
    speech = make_pcm(200, 4096)

    assert segmenter.feed(make_pcm(200)) == []
    assert segmenter.feed(speech) == []
    utterances = segmenter.feed(make_pcm(160))

    assert len(utterances) == 1
    assert speech in utterances[0]
    assert segmenter.finish() == b""


def test_split_samples_and_max_duration() -> None:
    """任意分块可跨采样边界，最长句限制不丢失字节。"""
    segmenter = PCMUtteranceSegmenter(0.008, 100, 200, 0)
    speech = make_pcm(300, 4096)
    assert segmenter.feed(speech[:17]) == []
    utterances = segmenter.feed(speech[17:])

    assert len(utterances) == 1
    assert utterances[0] + segmenter.finish() == speech


def test_wav_wrap_preserves_pcm() -> None:
    """WAV 只封装 PCM，不改变采样格式或字节。"""
    pcm = make_pcm(100, 4096)
    with wave.open(io.BytesIO(encode_wav(pcm)), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        assert audio.readframes(audio.getnframes()) == pcm

    with pytest.raises(ValueError, match="采样"):
        encode_wav(b"x")


@pytest.mark.parametrize("activity_ms", [20, 40, 60, 80, 100])
def test_short_threshold_crossing_is_not_submitted(activity_ms: int) -> None:
    """短促音量脉冲不生成输入，后续正常短句仍能起句。"""
    segmenter = PCMUtteranceSegmenter(0.008, 900, 30000, 500)
    preroll = make_pcm(500)
    trigger = make_pcm(activity_ms, 4096)
    silence = make_pcm(900)
    assert segmenter.feed(preroll) == []
    assert segmenter.feed(trigger) == []
    assert segmenter.feed(silence) == []
    assert segmenter.finish() == b""
    speech = make_pcm(160, 300)
    assert segmenter.feed(speech) == []
    assert segmenter.feed(silence) == [speech + silence]


def test_light_speech_and_soft_edges_keep_preroll_and_continuation() -> None:
    """持续轻声和带低能量句首句尾的短句不会按原阈值截断。"""
    segmenter = PCMUtteranceSegmenter(0.008, 900, 30000, 500)
    pcm = (make_pcm(300) + make_pcm(200, 180) + make_pcm(40, 300)
           + make_pcm(300, 180) + make_pcm(900))
    utterances: list[bytes] = []
    for offset in range(0, len(pcm), 1018):
        utterances.extend(segmenter.feed(pcm[offset:offset + 1018]))
    assert utterances == [pcm]
    assert segmenter.finish() == b""


def test_sparse_pulses_cannot_accumulate_into_sustained_speech() -> None:
    """相互隔开的噪声脉冲不能靠累计过阈值时长成为有效发言。"""
    segmenter = PCMUtteranceSegmenter(0.008, 900, 30000, 500)
    pcm = (make_pcm(20, 4096) + make_pcm(200)) * 8 + make_pcm(900)
    assert segmenter.feed(pcm) == []
    assert segmenter.finish() == b""


def test_network_boundaries_and_short_pauses_preserve_one_complete_utterance() -> None:
    """不足静音端点的句中停顿跨任意块只产生一次无损完整发言。"""
    segmenter = PCMUtteranceSegmenter(0.008, 900, 30000, 500)
    pcm = (make_pcm(500) + make_pcm(1400, 4096) + make_pcm(600)
           + make_pcm(1800, 8192) + make_pcm(900))
    utterances: list[bytes] = []
    for offset in range(0, len(pcm), 1018):
        utterances.extend(segmenter.feed(pcm[offset:offset + 1018]))
    assert utterances == [pcm]
    assert segmenter.finish() == b""
    with wave.open(io.BytesIO(encode_wav(utterances[0])), "rb") as audio:
        assert audio.readframes(audio.getnframes()) == pcm


def test_max_duration_and_capture_eof_have_distinct_boundary_reasons() -> None:
    """安全时长限制和采音正常结束不伪装成静音端点。"""
    segmenter = PCMUtteranceSegmenter(0.008, 900, 200, 0)
    pcm = make_pcm(300, 4096)
    utterances = segmenter.feed(pcm)
    assert segmenter.completed_stats[0]["end_reason"] == "max_duration"
    tail = segmenter.finish()
    assert segmenter.completed_stats[0]["end_reason"] == "capture_eof"
    assert b"".join(utterances) + tail == pcm


def test_max_duration_continuation_does_not_confirm_a_later_noise_pulse() -> None:
    """时长切分保留轻声延续，静音后短脉冲仍必须重新通过起句判定。"""
    segmenter = PCMUtteranceSegmenter(0.008, 100, 200, 0)
    assert segmenter.feed(make_pcm(200, 4096)) == [make_pcm(200, 4096)]
    assert segmenter.feed(make_pcm(100, 180) + make_pcm(100)) == [make_pcm(100, 180) + make_pcm(100)]
    assert segmenter.feed(make_pcm(100)) == []
    assert segmenter.feed(make_pcm(20, 4096) + make_pcm(100)) == []
    assert segmenter.finish() == b""
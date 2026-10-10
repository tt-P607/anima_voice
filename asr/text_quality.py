"""ASR 结果文本质量判断。"""

from __future__ import annotations

from dataclasses import dataclass

_COMMON_CJK_CHARS = set(
    "的一是在不了有人和国中大为上个年我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"
)


@dataclass(frozen=True)
class TextQualityDecision:
    """文本质量判断结果。"""

    accepted: bool
    reason: str = ""


def is_likely_normal_utterance(
    text: str,
    *,
    min_chars: int = 2,
    max_ascii_ratio: float = 0.2,
    min_cjk_ratio: float = 0.65,
    min_common_cjk_ratio: float = 0.45,
) -> TextQualityDecision:
    """按长度、字符比例和重复模式判断中文识别文本。"""

    normalized = "".join(char for char in text.strip() if not char.isspace())
    if len(normalized) < min_chars:
        return TextQualityDecision(False, "too_short")
    content_chars = [char for char in normalized if not _is_punctuation(char)]
    if len(content_chars) < min_chars:
        return TextQualityDecision(False, "too_short")
    total = len(content_chars)
    ascii_letters = sum(char.isascii() and char.isalpha() for char in content_chars)
    cjk_chars = [char for char in content_chars if _is_cjk(char)]
    if ascii_letters / total > max_ascii_ratio:
        return TextQualityDecision(False, "too_much_ascii")
    if len(cjk_chars) / total < min_cjk_ratio:
        return TextQualityDecision(False, "too_little_chinese")
    if len(cjk_chars) >= 4:
        common_count = sum(char in _COMMON_CJK_CHARS for char in cjk_chars)
        if common_count / len(cjk_chars) < min_common_cjk_ratio:
            return TextQualityDecision(False, "too_many_uncommon_chars")
    if _has_excessive_repetition(content_chars):
        return TextQualityDecision(False, "excessive_repetition")
    return TextQualityDecision(True)


def is_obviously_invalid_asr_text(
    text: str, *, min_chars: int = 2
) -> TextQualityDecision:
    """识别明显无效、标点或无语言字符的输出。"""

    normalized = "".join(char for char in text.strip() if not char.isspace())
    if len(normalized) < min_chars:
        return TextQualityDecision(True, "too_short")
    content_chars = [char for char in normalized if not _is_punctuation(char)]
    if len(content_chars) < min_chars:
        return TextQualityDecision(True, "punctuation_only")
    if _has_excessive_repetition(list(normalized)):
        return TextQualityDecision(True, "excessive_repetition")
    cjk_count = sum(_is_cjk(char) for char in content_chars)
    ascii_letters = sum(char.isascii() and char.isalpha() for char in content_chars)
    if cjk_count == 0 and ascii_letters == 0:
        return TextQualityDecision(True, "no_language_chars")
    return TextQualityDecision(False)


def _is_cjk(char: str) -> bool:
    """判断字符是否位于 CJK 统一表意文字区。"""

    return "\u4e00" <= char <= "\u9fff"


def _is_punctuation(char: str) -> bool:
    """判断字符是否为常见中英文标点。"""

    return char in "，。！？、；：,.!?;:()（）[]【】'\"“”‘’《》<>-—_~`!@#$%^&*+=|\\/"


def _has_excessive_repetition(chars: list[str]) -> bool:
    """检测连续重复字符异常。"""

    if len(chars) < 5:
        return False
    previous = ""
    run_length = 0
    for char in chars:
        if char == previous:
            run_length += 1
            if run_length >= 4:
                return True
        else:
            previous = char
            run_length = 1
    return False


__all__ = [
    "TextQualityDecision",
    "is_likely_normal_utterance",
    "is_obviously_invalid_asr_text",
]
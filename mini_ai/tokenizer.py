"""한국어/영어 혼용 문장을 검색용 토큰으로 쪼개는 모듈.

형태소 분석기 없이 동작해야 하므로, 규칙 기반 어미/조사 제거와
음절 n-gram을 섞어서 오타나 활용형에 어느 정도 견디도록 만들었다.
"""

import re
import unicodedata

_TOKEN_RE = re.compile(r"[가-힣]+|[a-zA-Z]+|[0-9]+")
_HANGUL_RE = re.compile(r"^[가-힣]+$")

# 긴 것부터 지워야 "에서는"이 "는"으로 잘못 잘리지 않는다.
_SUFFIXES = sorted(
    (
        # 조사
        "으로써", "으로서", "이라고", "에서는", "에게서", "한테서", "으로", "이라", "라고",
        "에서", "에게", "한테", "까지", "부터", "보다", "처럼", "마다", "조차", "밖에",
        "이랑", "하고", "이나", "이야", "께서", "은", "는", "이", "가", "을", "를",
        "의", "에", "와", "과", "도", "만", "로", "야", "랑", "께",
        # 어미
        "습니까", "습니다", "였습니다", "했습니다", "이에요", "예요", "인가요", "일까요",
        "했어요", "하세요", "해줘요", "인가", "일까", "을까", "ㄹ까", "어요", "아요",
        "네요", "거야", "건가", "해요", "해줘", "해봐", "하지", "한다", "했다", "이다",
        "입니다", "니다", "는데", "니야", "냐", "지", "죠", "고", "해", "함", "임",
    ),
    key=len,
    reverse=True,
)


def normalize(text):
    """유니코드 정규화 + 소문자 + 공백 정리."""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def words(text):
    """문장에서 한글/영문/숫자 덩어리만 뽑아낸다."""
    return _TOKEN_RE.findall(normalize(text))


def stem(word):
    """가장 긴 조사/어미 하나를 떼어낸 어간 근사값."""
    if not _HANGUL_RE.match(word) or len(word) < 3:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[: -len(suffix)]
    return word


def bigrams(word):
    """한글 단어의 음절 2-gram. '반갑습니다' -> ['반갑','갑습','습니','니다']"""
    if not _HANGUL_RE.match(word) or len(word) < 2:
        return []
    return [word[i : i + 2] for i in range(len(word) - 1)]


def features(text):
    """검색에 쓸 최종 토큰 목록.

    원형 단어 + 어간 + 음절 bigram을 모두 넣어서,
    '안녕하세요'와 '안녕'이 서로를 어느 정도 알아보게 만든다.
    """
    out = []
    for word in words(text):
        out.append(word)
        root = stem(word)
        if root != word:
            out.append(root)
        out.extend(bigrams(word))
    return out

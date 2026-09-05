"""검색보다 먼저 시도하는 규칙 기반 능력들.

각 스킬은 (text, bot) -> str | None 형태이고, None을 돌려주면
다음 스킬 또는 TF-IDF 검색으로 넘어간다.
"""

import ast
import datetime
import operator
import random
import re

from .tokenizer import normalize

# ---------------------------------------------------------------- 계산기

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_MATH_CHARS = re.compile(r"^[\d\s+\-*/().%^]+$")
_KOREAN_OPS = [
    ("더하기", "+"), ("빼기", "-"), ("곱하기", "*"), ("나누기", "/"),
    ("플러스", "+"), ("마이너스", "-"), ("나머지", "%"), ("제곱", "**"),
]


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("숫자가 아님")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("허용되지 않은 식")


def safe_eval(expression):
    """eval() 대신 AST를 직접 걸어서 사칙연산만 허용한다."""
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    if isinstance(result, float):
        return round(result, 10)
    return result


def calculator(text, bot):
    expression = normalize(text)
    for korean, symbol in _KOREAN_OPS:
        expression = expression.replace(korean, symbol)
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    expression = re.sub(r"(는|은)?\s*(뭐야|몇이야|얼마야|계산해줘|계산해|얼마|몇)\s*[?？]?$", "", expression)
    expression = expression.strip()

    if not expression or not _MATH_CHARS.match(expression.replace("**", "^")):
        return None
    if not re.search(r"[+\-*/%]", expression) or not re.search(r"\d", expression):
        return None

    try:
        return "{} = {}".format(expression, safe_eval(expression))
    except ZeroDivisionError:
        return "0으로는 못 나눠. 그건 나도 안 돼."
    except (SyntaxError, ValueError):
        return None


# ---------------------------------------------------------------- 시간/날짜

_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


def clock(text, bot):
    query = normalize(text)
    now = datetime.datetime.now()
    if re.search(r"(몇\s*시|시간\s*(알려|뭐|몇))", query):
        return now.strftime("지금 %H시 %M분이야.")
    # 그냥 "오늘"만 보고 반응하면 "오늘 뭐 먹지"까지 날짜로 답해버린다.
    if re.search(r"(며칠|몇\s*일|무슨\s*요일|요일이야|날짜)", query):
        return "오늘은 {}년 {}월 {}일 {}요일이야.".format(
            now.year, now.month, now.day, _WEEKDAYS[now.weekday()]
        )
    return None


# ---------------------------------------------------------------- 기억

# 이름 뒤에 붙는 서술 어미는 이름의 일부가 아니므로 따로 떼어낸다.
_NAME_TAIL = re.compile(r"(?:이야|이에요|입니다|예요|라고\s*해|이라고\s*해|야|임|다)$")
_NAME_PATTERNS = [
    re.compile(r"내\s*이름은\s*([가-힣a-zA-Z]{1,20})"),
    re.compile(r"나는\s*([가-힣a-zA-Z]{1,20})(?:야|이야|입니다|이라고\s*해|라고\s*해)"),
    re.compile(r"저는\s*([가-힣a-zA-Z]{1,20})(?:예요|이에요|입니다)"),
]


def _clean_name(raw):
    name = _NAME_TAIL.sub("", raw.strip())
    # 어미를 떼고 한 글자도 안 남으면 원본이 곧 이름이다.
    return name if len(name) >= 1 else raw.strip()


def memory(text, bot):
    query = normalize(text)

    for pattern in _NAME_PATTERNS:
        match = pattern.search(query)
        if match:
            name = _clean_name(match.group(1))
            bot.remember("이름", name)
            return "반가워 {0}! 이제 {0}라고 부를게.".format(name)

    if re.search(r"내\s*이름", query):
        name = bot.recall("이름")
        return "너는 {}(이)잖아.".format(name) if name else "아직 이름 안 알려줬는데? '내 이름은 ○○' 이렇게 말해줘."

    match = re.search(r"내\s*([가-힣a-zA-Z ]{1,15}?)(?:은|는)\s*(.+?)(?:야|이야|입니다|예요)$", query)
    if match:
        key, value = match.group(1).strip(), match.group(2).strip()
        if key and value:
            bot.remember(key, value)
            return "{} = {} 라고 기억해뒀어.".format(key, value)

    match = re.search(r"내\s*([가-힣a-zA-Z ]{1,15}?)\s*(?:이|가)?\s*뭐(?:야|였지|지)", query)
    if match:
        key = match.group(1).strip()
        value = bot.recall(key)
        if value:
            return "네 {}은(는) {}(이)야.".format(key, value)
    return None


# ---------------------------------------------------------------- 잡기술

_DICE_RE = re.compile(r"(?:(\d{1,2})\s*[dD]\s*)?(\d{1,3})\s*면체|주사위")


def dice(text, bot):
    query = normalize(text)
    if "동전" in query and re.search(r"(던져|뒤집|튕겨)", query):
        return "결과는 **{}**!".format(bot.rng.choice(["앞면", "뒷면"]))
    if "주사위" in query:
        return "주사위 굴린 결과: {}".format(bot.rng.randint(1, 6))
    match = re.search(r"(\d+)\s*(?:부터|~|-)\s*(\d+)\s*(?:사이|중).*(?:랜덤|아무|뽑아|골라)", query)
    if match:
        low, high = sorted((int(match.group(1)), int(match.group(2))))
        return "내가 고른 숫자는 {}!".format(bot.rng.randint(low, high))
    return None


def coin_choice(text, bot):
    """'A랑 B 중에 뭐가 나아?' 같은 선택 질문."""
    query = normalize(text)
    if not re.search(r"(중에|중에서).*(뭐|어느|어떤|골라|추천)", query):
        return None
    head = re.split(r"중에|중에서", query)[0]
    options = [o.strip() for o in re.split(r"(?:랑|이랑|하고|,|와|과|或)", head) if o.strip()]
    options = [o for o in options if 0 < len(o) <= 20]
    if len(options) < 2:
        return None
    return "고민되면 {} 쪽에 한 표.".format(bot.rng.choice(options))


ALL_SKILLS = [memory, calculator, clock, dice, coin_choice]

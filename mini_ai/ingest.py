"""사용자가 던진 자유로운 정보를 구조화된 JSON 데이터셋으로 정리한다.

정리 결과는 dataset.json 에 쌓이고, Brain 이 그 파일을 읽어 학습한다.
사람이 직접 열어서 고칠 수 있도록 일부러 읽기 쉬운 형태로 저장한다.

인식하는 입력 형태
  Q: 질문 / A: 답변      명시적 문답
  질문 | 답변             한 줄 문답
  키: 값                  사실
  X는 Y이다 / X은 Y야     사실 + 자동 생성 문답
  그 외 문장              문장 생성용 말뭉치
"""

import json
import os
import re

_QUESTION_LINE = re.compile(r"^\s*(?:q|질문|물음)\s*[:：.]\s*(.+)$", re.IGNORECASE)
_ANSWER_LINE = re.compile(r"^\s*(?:a|답|답변)\s*[:：.]\s*(.+)$", re.IGNORECASE)
_KEY_VALUE = re.compile(r"^\s*([^:：|]{1,30}?)\s*[:：]\s*(.+?)\s*$")
# "X는 Y이다" 꼴. 서술문("오늘은 날씨가 좋았다")까지 사실로 삼지 않으려고
# 서술격 조사를 필수로 요구하고, 주어는 두 어절까지만 허용한다.
_DEFINITION = re.compile(
    r"^\s*(\S+(?:\s\S+)?)\s*(?:은|는|이란|란)\s+(.+?)"
    r"\s*(?:이다|이야|입니다|이에요|예요|라고\s*해)\s*[.!]?\s*$"
)
_QUESTION_MARK = re.compile(r"[?？]\s*$")


def has_batchim(word):
    """마지막 글자에 받침이 있는지. 조사를 자연스럽게 고르는 데 쓴다."""
    if not word:
        return False
    last = word.strip()[-1]
    if not "가" <= last <= "힣":
        return False
    return (ord(last) - 0xAC00) % 28 != 0


def _question_forms(subject):
    """사실 하나에서 물어볼 법한 질문들을 만들어 둔다."""
    subject = subject.strip()
    subject_marker = "이" if has_batchim(subject) else "가"
    return [
        "{} 뭐야".format(subject),
        "{}{} 뭐야".format(subject, subject_marker),
        "{} 알려줘".format(subject),
        "{}에 대해 설명해줘".format(subject),
    ]


def parse(raw_text):
    """자유 텍스트를 {facts, pairs, sentences} 로 정리한다.

    같은 내용을 여러 번 넣어도 중복이 쌓이지 않도록 키 기준으로 합친다.
    """
    facts = {}
    pairs = []
    sentences = []
    seen_questions = set()

    def add_pair(question, answer, origin):
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            return
        key = question.lower()
        if key in seen_questions:
            return
        seen_questions.add(key)
        pairs.append({"question": question, "answer": answer, "origin": origin})

    lines = [line.strip() for line in (raw_text or "").splitlines()]
    pending_question = None

    for line in lines:
        if not line or line.startswith("#"):
            continue

        match = _QUESTION_LINE.match(line)
        if match:
            pending_question = match.group(1).strip()
            continue

        match = _ANSWER_LINE.match(line)
        if match and pending_question:
            add_pair(pending_question, match.group(1), "qa")
            pending_question = None
            continue

        # 질문만 덩그러니 있고 다음 줄이 답인 경우
        if pending_question:
            add_pair(pending_question, line, "qa")
            pending_question = None
            continue

        if "|" in line:
            question, answer = line.split("|", 1)
            if question.strip() and answer.strip():
                add_pair(question, answer, "pipe")
                continue

        if _QUESTION_MARK.search(line):
            pending_question = line
            continue

        # 콜론은 가장 명시적인 표기이므로 정의문보다 먼저 해석한다.
        match = _KEY_VALUE.match(line)
        if match:
            key, value = match.group(1).strip(), match.group(2).strip()
            facts[key] = value
            for form in _question_forms(key):
                add_pair(form, value, "fact")
            continue

        match = _DEFINITION.match(line)
        if match:
            subject, description = match.group(1).strip(), match.group(2).strip()
            if subject and description and len(description) >= 2:
                facts[subject] = description
                for form in _question_forms(subject):
                    add_pair(form, line, "definition")
                sentences.append(line)
                continue

        sentences.append(line)

    if pending_question:  # 답을 못 받은 질문은 문장으로만 남긴다.
        sentences.append(pending_question)

    return {"facts": facts, "pairs": pairs, "sentences": sentences}


def load_dataset(path):
    """저장된 데이터셋을 읽는다. 없거나 깨졌으면 빈 데이터셋."""
    empty = {"facts": {}, "pairs": [], "sentences": []}
    if not os.path.exists(path):
        return empty
    try:
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
    except (ValueError, OSError):
        return empty
    if not isinstance(stored, dict):
        return empty
    return {
        "facts": dict(stored.get("facts") or {}),
        "pairs": [
            item for item in (stored.get("pairs") or [])
            if isinstance(item, dict) and item.get("question") and item.get("answer")
        ],
        "sentences": [s for s in (stored.get("sentences") or []) if isinstance(s, str)],
    }


def merge(dataset, parsed):
    """정리된 내용을 기존 데이터셋에 합치고, 새로 늘어난 개수를 돌려준다."""
    added = {"facts": 0, "pairs": 0, "sentences": 0}

    for key, value in parsed["facts"].items():
        if dataset["facts"].get(key) != value:
            added["facts"] += 1
        dataset["facts"][key] = value

    existing = {item["question"].lower() for item in dataset["pairs"]}
    for item in parsed["pairs"]:
        key = item["question"].lower()
        if key in existing:
            # 같은 질문이면 최신 답으로 갱신한다.
            for stored in dataset["pairs"]:
                if stored["question"].lower() == key:
                    stored["answer"] = item["answer"]
            continue
        existing.add(key)
        dataset["pairs"].append(item)
        added["pairs"] += 1

    known = set(dataset["sentences"])
    for sentence in parsed["sentences"]:
        if sentence not in known:
            known.add(sentence)
            dataset["sentences"].append(sentence)
            added["sentences"] += 1

    return added


def save_dataset(dataset, path):
    """원자적으로 저장한다. 도중에 죽어도 기존 파일은 남는다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def compile_source(source_path, dataset_path):
    """사용자가 편집하는 원본 파일을 읽어 dataset.json 으로 컴파일한다.

    원본은 자유 텍스트(.txt)여도 되고 이미 구조화된 JSON 이어도 된다.
    호출할 때마다 처음부터 다시 만들기 때문에, 원본에서 지운 내용은
    데이터셋에서도 사라진다 - 파일이 곧 유일한 진실이 된다.

    (dataset, 개수요약) 를 돌려준다. 원본이 없으면 빈 데이터셋.
    """
    dataset = {"facts": {}, "pairs": [], "sentences": []}
    if not source_path or not os.path.exists(source_path):
        return dataset, {"facts": 0, "pairs": 0, "sentences": 0}

    with open(source_path, encoding="utf-8") as handle:
        raw = handle.read()

    stripped = raw.lstrip()
    if stripped.startswith("{"):
        # 이미 정리된 JSON 을 그대로 준 경우
        try:
            loaded = json.loads(raw)
        except ValueError as error:
            raise ValueError("JSON 형식이 잘못됐어: {}".format(error))
        parsed = {
            "facts": dict(loaded.get("facts") or {}),
            "pairs": [
                dict(item) for item in (loaded.get("pairs") or [])
                if isinstance(item, dict) and item.get("question") and item.get("answer")
            ],
            "sentences": [s for s in (loaded.get("sentences") or []) if isinstance(s, str)],
        }
        for key, value in parsed["facts"].items():
            for form in _question_forms(key):
                if not any(p["question"].lower() == form.lower() for p in parsed["pairs"]):
                    parsed["pairs"].append(
                        {"question": form, "answer": value, "origin": "fact"}
                    )
    else:
        parsed = parse(raw)

    added = merge(dataset, parsed)
    save_dataset(dataset, dataset_path)
    return dataset, added

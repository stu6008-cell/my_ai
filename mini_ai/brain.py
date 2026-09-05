"""대화 엔진 본체.

응답 순서:
  1) 되묻기에 대한 확인 응답 처리
  2) TF-IDF 유사도 검색 - 내가 준 데이터가 최우선
  3) 규칙 기반 스킬 (계산, 시간 ...)
  4) 나이브 베이즈 분류기의 두 번째 의견
  5) 애매하면 되묻기, 전혀 모르면 원본 파일 안내

지식은 원본 파일에서만 온다. 대화는 기본적으로 읽기 전용이다.
"""

import json
import os
import random
import re
from collections import namedtuple

from . import ingest, skills
from .learner import NaiveBayesClassifier, ResponseRanker
from .markov import MarkovChain
from .tokenizer import features, normalize
from .vectorizer import TfidfIndex

Reply = namedtuple("Reply", "text source score")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_KNOWLEDGE = os.path.join(DATA_DIR, "knowledge.json")
DEFAULT_MEMORY = os.path.join(DATA_DIR, "memory.json")
DEFAULT_DATASET = os.path.join(DATA_DIR, "dataset.json")
DEFAULT_SOURCE = os.path.join(DATA_DIR, "source.txt")

# 이 아래면 "모르겠다", 이 위면 확신, 사이면 되묻는다.
CONFIDENT = 0.30
UNSURE = 0.13
# 검색이 애매할 때 분류기 단독으로 답하려면 이만큼은 확신해야 한다.
BAYES_CONFIDENT = 0.80

_YES = re.compile(r"^(응+|어+|웅|ㅇ+|네|넹|예|맞아요?|맞어|그래|그렇지|정답|yes|y|ok)[.!~]*$")
_NO = re.compile(r"^(아니+(야|요)?|아냐|ㄴ+|노|틀렸어|틀려|아닌데|no|n)[.!~]*$")

_FALLBACKS = [
    "음, 그건 아직 모르겠어. '/배워 {q} | 답변' 으로 알려주면 외울게.",
    "처음 듣는 얘기야. 뭐라고 답하면 좋을지 가르쳐줄래? (/배워 {q} | 답변)",
    "그 말은 내 지식에 없네. '/배워' 로 알려주면 다음엔 대답할 수 있어.",
]

_EMPTY_MEMORY = {
    "facts": {},       # 사용자에 대해 외운 사실
    "learned": [],     # 직접 가르친 문답
    "patterns": {},    # 대화하며 스스로 익힌 표현 (의도 이름 -> 문장들)
    "examples": [],    # 분류기 학습용 예문 (텍스트 + 라벨)
    "weights": {},     # 응답별 피드백 가중치
    "stats": {"turns": 0, "matched": 0, "unknown": 0, "taught": 0, "auto_learned": 0},
}


class Brain:
    def __init__(
        self,
        knowledge_path=DEFAULT_KNOWLEDGE,
        memory_path=DEFAULT_MEMORY,
        dataset_path=DEFAULT_DATASET,
        source_path=DEFAULT_SOURCE,
        strict=True,
        learn_from_chat=False,
        rng=None,
    ):
        """지식의 출처를 두 개의 스위치로 정한다.

        strict=True          내장 지식(knowledge.json)을 아예 읽지 않는다.
        learn_from_chat=False 대화로는 아무것도 배우지 않는다. 원본 파일이
                             유일한 학습 경로이고, 대화는 읽기 전용이다.

        계산기·시계는 기억이 아니라 계산이라 두 스위치와 무관하게 동작한다.
        """
        self.knowledge_path = knowledge_path
        self.memory_path = memory_path
        self.dataset_path = dataset_path
        self.source_path = source_path
        self.strict = strict
        self.learn_from_chat = learn_from_chat
        self.rng = rng or random.Random()
        self.dataset = ingest.load_dataset(dataset_path)

        self.intents = []
        self.memory = json.loads(json.dumps(_EMPTY_MEMORY))  # 깊은 복사
        self.index = TfidfIndex()
        self.markov = MarkovChain(order=2, rng=self.rng)
        self.classifier = NaiveBayesClassifier()
        self.ranker = ResponseRanker(rng=self.rng)

        self.last_user_message = ""
        self._last_reply = None
        self._last_label = None          # 피드백을 귀속시킬 대상
        self.awaiting_answer_for = None  # 대화형 학습 상태
        self.pending_confirm = None      # (원문, payload) - 되묻는 중

        self._load_knowledge()
        self._load_memory()
        self.reload_source()   # 원본 파일이 곧 지식이다

    def reload_source(self):
        """원본 파일을 다시 읽어 데이터셋과 색인을 통째로 갱신한다.

        컴파일이 실패해도 이전 데이터셋으로 계속 동작해야 하므로,
        예외는 호출한 쪽에 알리되 색인은 반드시 다시 만든다.
        """
        error = None
        if self.source_path and os.path.exists(self.source_path):
            try:
                self.dataset, added = ingest.compile_source(self.source_path, self.dataset_path)
            except (ValueError, OSError) as failure:
                error, added = failure, None
            else:
                self.reindex()
                return added, None
        else:
            added = None
        self.reindex()
        return added, error

    # ------------------------------------------------------------ 적재/저장

    def _load_knowledge(self):
        if self.strict:
            self.intents = []           # 내가 준 정보 말고는 아무것도 안다고 하지 않는다
            self._intent_positions = {}
            return
        with open(self.knowledge_path, encoding="utf-8") as handle:
            self.intents = json.load(handle).get("intents", [])
        self._intent_positions = {
            intent.get("name", str(i)): i for i, intent in enumerate(self.intents)
        }

    def _load_memory(self):
        if not os.path.exists(self.memory_path):
            return
        try:
            with open(self.memory_path, encoding="utf-8") as handle:
                stored = json.load(handle)
        except (ValueError, OSError):
            return  # 손상된 기억 파일 때문에 실행이 막히면 안 된다.
        if not isinstance(stored, dict):
            return

        self.memory["facts"] = dict(stored.get("facts", {}))
        self.memory["learned"] = [
            item for item in stored.get("learned", [])
            if isinstance(item, dict) and item.get("question") and item.get("answer")
        ]
        self.memory["patterns"] = {
            name: list(texts)
            for name, texts in (stored.get("patterns") or {}).items()
            if name in self._intent_positions and isinstance(texts, list)
        }
        self.memory["examples"] = [
            item for item in stored.get("examples", [])
            if isinstance(item, dict) and item.get("text") and item.get("label")
        ]
        self.memory["weights"] = {
            key: float(value)
            for key, value in (stored.get("weights") or {}).items()
            if isinstance(value, (int, float))
        }
        self.memory["stats"].update(
            {k: int(v) for k, v in (stored.get("stats") or {}).items() if isinstance(v, int)}
        )

    def save(self):
        os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
        self.memory["weights"] = self.ranker.weights
        temp_path = self.memory_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(self.memory, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.memory_path)  # 저장 중 중단돼도 기존 파일은 살아남는다.

    # ------------------------------------------------------------ 색인/학습

    def reindex(self):
        """검색 색인, 분류기, 생성기를 현재 지식 기준으로 다시 만든다.

        분류기를 통째로 다시 세우는 게 낭비처럼 보이지만, 예문을 원문 그대로
        보관하기 때문에 지식 파일을 사람이 고쳐도 상태가 어긋나지 않는다.
        """
        self.index = TfidfIndex()
        self.classifier = NaiveBayesClassifier()
        self.ranker = ResponseRanker(self.memory["weights"], rng=self.rng)
        self._label_answers = {}   # 분류기 라벨 -> 실제 답변
        corpus = []

        for position, intent in enumerate(self.intents):
            name = intent.get("name", str(position))
            for pattern in intent.get("patterns", []):
                self.index.add(pattern, ("intent", position))
                self.classifier.train(features(pattern), name)
            for pattern in self.memory["patterns"].get(name, []):
                self.index.add(pattern, ("intent", position))
                self.classifier.train(features(pattern), name)
                corpus.append(pattern)
            corpus.extend(intent.get("patterns", []))
            corpus.extend(intent.get("responses", []))

        # 같은 답을 가리키는 질문들을 한 라벨로 묶어 분류기에 먹인다.
        # "수도 뭐야 / 수도가 뭐야 / 수도 알려줘" 가 한 묶음이 되므로,
        # 파일에 없는 표현으로 물어도 같은 답에 닿을 수 있다.
        answer_labels = {}
        for position, item in enumerate(self.dataset["pairs"]):
            self.index.add(item["question"], ("dataset", position))
            label = answer_labels.get(item["answer"])
            if label is None:
                label = "answer:{}".format(len(answer_labels))
                answer_labels[item["answer"]] = label
                self._label_answers[label] = item["answer"]
            self.classifier.train(features(item["question"]), label)
            corpus.append(item["question"])
            corpus.append(item["answer"])
        corpus.extend(self.dataset["sentences"])

        for position, item in enumerate(self.memory["learned"]):
            self.index.add(item["question"], ("learned", position))
            corpus.append(item["question"])
            corpus.append(item["answer"])

        for item in self.memory["examples"]:
            if item["label"] in self._intent_positions:
                self.classifier.train(features(item["text"]), item["label"])

        self.index.build()
        self.markov = MarkovChain(order=2, rng=self.rng)
        self.markov.train(corpus)

    def absorb(self, text, label, as_pattern=True):
        """대화 중 얻은 문장을 사례로 흡수한다. learn_from_chat 이 꺼져 있으면 거부."""
        if not self.learn_from_chat:
            return False
        return self._absorb(text, label, as_pattern)

    def _absorb(self, text, label, as_pattern=True):
        """문장 하나를 특정 의도의 사례로 흡수한다.

        as_pattern=True 면 검색 색인에도 넣어 다음엔 유사도만으로 잡히게 하고,
        False 면 분류기 예문으로만 쓴다.
        """
        text = text.strip()
        if not text or label not in self._intent_positions:
            return False

        known = set(self.intents[self._intent_positions[label]].get("patterns", []))
        known.update(self.memory["patterns"].get(label, []))
        if text in known:
            return False

        self.memory["examples"].append({"text": text, "label": label})
        if as_pattern:
            self.memory["patterns"].setdefault(label, []).append(text)
            self.memory["stats"]["auto_learned"] += 1
        self.reindex()
        return True

    def train_from_file(self, path):
        """텍스트 파일로 한꺼번에 학습한다.

        '질문 | 답변' 줄은 문답으로, 나머지 줄은 문장 생성용 말뭉치로 쓴다.
        '#' 로 시작하는 줄은 주석.
        """
        with open(path, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]

        pairs, sentences = 0, []
        for line in lines:
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                question, answer = line.split("|", 1)
                if question.strip() and answer.strip():
                    self._learn_quiet(question, answer)
                    pairs += 1
                    continue
            sentences.append(line)

        self.reindex()
        self.markov.train(sentences)
        self.memory["stats"]["taught"] += pairs
        return pairs, len(sentences)

    def ingest_text(self, raw_text):
        """자유 텍스트를 구조화해서 dataset.json 에 저장하고 바로 학습한다.

        (정리 결과, 새로 늘어난 개수) 를 돌려준다.
        """
        parsed = ingest.parse(raw_text)
        added = ingest.merge(self.dataset, parsed)
        ingest.save_dataset(self.dataset, self.dataset_path)
        self.memory["stats"]["ingested"] = self.memory["stats"].get("ingested", 0) + added["pairs"]
        for key, value in parsed["facts"].items():
            self.remember(key, value)
        self.reindex()
        return parsed, added

    def ingest_file(self, path):
        with open(path, encoding="utf-8") as handle:
            return self.ingest_text(handle.read())

    def dataset_summary(self):
        facts, pairs, sentences = (
            self.dataset["facts"], self.dataset["pairs"], self.dataset["sentences"]
        )
        if not facts and not pairs and not sentences:
            return "데이터셋이 비어 있어. '/정보' 로 알려주면 정리해서 저장할게.\n({})".format(
                self.dataset_path
            )

        lines = ["파일: {}".format(self.dataset_path)]
        if facts:
            lines.append("[사실 {}개]".format(len(facts)))
            lines.extend("  {} = {}".format(k, v) for k, v in list(facts.items())[:20])
        if pairs:
            lines.append("[문답 {}개]".format(len(pairs)))
            lines.extend(
                "  {} -> {}".format(item["question"], item["answer"]) for item in pairs[:20]
            )
            if len(pairs) > 20:
                lines.append("  ... 외 {}개".format(len(pairs) - 20))
        if sentences:
            lines.append("[문장 {}개]".format(len(sentences)))
        return "\n".join(lines)

    # ------------------------------------------------------------ 기억 API

    def remember(self, key, value):
        self.memory["facts"][normalize(key)] = value

    def recall(self, key):
        return self.memory["facts"].get(normalize(key))

    def forget(self, key=None):
        if key is None:
            count = (
                len(self.memory["facts"])
                + len(self.memory["learned"])
                + sum(len(v) for v in self.memory["patterns"].values())
                + len(self.memory["examples"])
            )
            self.memory["facts"].clear()
            self.memory["learned"] = []
            self.memory["patterns"] = {}
            self.memory["examples"] = []
            self.memory["weights"] = {}
            self.reindex()
            return count

        key = normalize(key)
        if key in self.memory["facts"]:
            del self.memory["facts"][key]
            return 1

        removed = 0
        before = len(self.memory["learned"])
        self.memory["learned"] = [
            item for item in self.memory["learned"] if normalize(item["question"]) != key
        ]
        removed += before - len(self.memory["learned"])

        for name, texts in list(self.memory["patterns"].items()):
            kept = [t for t in texts if normalize(t) != key]
            removed += len(texts) - len(kept)
            if kept:
                self.memory["patterns"][name] = kept
            else:
                del self.memory["patterns"][name]

        before = len(self.memory["examples"])
        self.memory["examples"] = [
            item for item in self.memory["examples"] if normalize(item["text"]) != key
        ]
        removed += before - len(self.memory["examples"])

        if removed:
            self.reindex()
        return removed

    def _learn_quiet(self, question, answer):
        """reindex 없이 문답만 등록한다. 대량 학습용."""
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            raise ValueError("질문과 답변이 모두 필요해")
        key = normalize(question)
        for item in self.memory["learned"]:
            if normalize(item["question"]) == key:
                item["answer"] = answer
                return False
        self.memory["learned"].append({"question": question, "answer": answer})
        return True

    def learn(self, question, answer):
        if not self.learn_from_chat:
            raise PermissionError(
                "대화로는 안 배우도록 설정돼 있어. 원본 파일({})에 적고 '/새로고침' 해줘.".format(
                    self.source_path
                )
            )
        created = self._learn_quiet(question, answer)
        self.memory["stats"]["taught"] += 1
        self.reindex()
        return created

    # ------------------------------------------------------------ 피드백

    def feedback(self, positive):
        """직전 답변 평가. 대화 학습이 꺼져 있으면 아무것도 하지 않는다."""
        if not self.learn_from_chat:
            return None
        return self._feedback(positive)

    def _feedback(self, positive):
        """직전 답변에 대한 칭찬/지적을 반영한다.

        칭찬이면 그 답변의 가중치를 올리고, 사용자가 실제로 쓴 표현까지
        해당 의도의 사례로 흡수한다. 지적이면 가중치를 낮춘다.
        """
        if not self._last_label or not self._last_reply:
            return None
        label, response = self._last_label, self._last_reply
        if positive:
            weight = self.ranker.reward(label, response)
            learned = self._absorb(self.last_user_message, label)
            self.memory["weights"] = self.ranker.weights
            return ("up", label, weight, learned)
        weight = self.ranker.penalize(label, response)
        self.memory["weights"] = self.ranker.weights
        return ("down", label, weight, False)

    # ------------------------------------------------------------ 응답

    def _intent_name(self, position):
        return self.intents[position].get("name", str(position))

    def _payload_answer(self, payload):
        kind, position = payload
        if kind == "dataset":
            item = self.dataset["pairs"][position]
            self._last_label = "데이터:" + item["question"]
            return item["answer"]
        if kind == "learned":
            self._last_label = "학습:" + self.memory["learned"][position]["question"]
            return self.memory["learned"][position]["answer"]
        name = self._intent_name(position)
        self._last_label = name
        responses = self.intents[position].get("responses") or ["..."]
        return self.ranker.choose(name, responses, avoid=self._last_reply)

    def _finish(self, text, source, score):
        self._last_reply = text
        return Reply(text, source, score)

    def _handle_confirmation(self, text):
        """되묻기 직후의 응/아니 처리. 해당 없으면 None."""
        original, payload = self.pending_confirm
        stripped = normalize(text)

        if _NO.match(stripped):
            self.pending_confirm = None
            hint = (
                "그럼 원본 파일에 적어줘: {}".format(self.source_path)
                if not self.learn_from_chat else "그럼 뭐라고 답해야 해? (/배워 로 알려줘)"
            )
            return self._finish("아, 아니구나. " + hint, "confirm:no", 0.0)

        if not _YES.match(stripped):
            self.pending_confirm = None
            return None  # 확인이 아니라 새 이야기다.

        self.pending_confirm = None
        kind, position = payload
        if not self.learn_from_chat:
            # 되묻기는 하되, 그 대답으로 지식을 늘리지는 않는다.
            return self._finish(self._payload_answer(payload), "confirm:yes", 1.0)
        if kind == "intent":
            learned = self._absorb(original, self._intent_name(position))
            answer = self._payload_answer(payload)
            if learned:
                answer += "\n(그리고 '{}' 도 같은 뜻으로 외웠어.)".format(original)
        else:
            item = (
                self.dataset["pairs"][position] if kind == "dataset"
                else self.memory["learned"][position]
            )
            self._learn_quiet(original, item["answer"])
            self.memory["stats"]["auto_learned"] += 1
            self.reindex()
            answer = item["answer"] + "\n('{}' 도 같은 질문으로 외웠어.)".format(original)
        return self._finish(answer, "confirm:yes", 1.0)

    def respond(self, text):
        """사용자 발화 하나에 대한 Reply를 만든다."""
        text = (text or "").strip()
        if not text:
            return Reply("뭐라고? 아무것도 안 들렸어.", "empty", 0.0)

        # 대화형 학습 중이면 이번 입력을 답변으로 받는다. (기본값에서는 꺼져 있다)
        if self.learn_from_chat and self.awaiting_answer_for is not None:
            question, self.awaiting_answer_for = self.awaiting_answer_for, None
            self.learn(question, text)
            return self._finish(
                "외웠어. 이제 '{}' 라고 물으면 그렇게 답할게.".format(question), "learn", 1.0
            )

        if self.pending_confirm is not None:
            reply = self._handle_confirmation(text)
            if reply is not None:
                return reply

        self.last_user_message = text
        self.memory["stats"]["turns"] += 1

        # 내가 준 데이터가 내장 스킬보다 우선한다. 파일에 "출근 시간"을 적어놨는데
        # "출근 몇시에 해" 가 시계 스킬로 새면 안 되기 때문이다.
        results = self.index.search(text, top_k=3)
        top = results[0] if results else None

        if top and top[0] >= CONFIDENT:
            score, _, payload = top
            answer = self._payload_answer(payload)
            self.memory["stats"]["matched"] += 1
            # 확신한 매칭을 분류기 자료로 되먹이는 건 대화 학습이라 기본적으로 막혀 있다.
            if self.learn_from_chat and payload[0] == "intent":
                self.memory["examples"].append(
                    {"text": text, "label": self._intent_name(payload[1])}
                )
                self.classifier.train(features(text), self._intent_name(payload[1]))
            return self._finish(answer, "match:" + top[1], score)

        for skill in skills.ALL_SKILLS:
            answer = skill(text, self)
            if answer:
                self._last_label = "skill:" + skill.__name__
                self.memory["stats"]["matched"] += 1
                return self._finish(answer, "skill:" + skill.__name__, 1.0)

        # 검색이 흔들릴 때 분류기에게 물어본다.
        prediction = self.classifier.predict(features(text))
        if prediction and prediction[1] >= BAYES_CONFIDENT:
            label, probability = prediction
            if label in self._label_answers:
                answer = self._label_answers[label]
                self._last_label = label
            elif label in self._intent_positions:
                answer = self._payload_answer(("intent", self._intent_positions[label]))
            else:
                answer = None
            if answer:
                self.memory["stats"]["matched"] += 1
                return self._finish(answer, "bayes:" + label, probability)

        if top and top[0] >= UNSURE:
            self.pending_confirm = (text, top[2])
            self._last_label = None
            return self._finish(
                "혹시 '{}' 얘기야? 맞으면 '응' 이라고 해줘.".format(top[1]),
                "clarify:" + top[1],
                top[0],
            )

        self._last_label = None
        self.memory["stats"]["unknown"] += 1
        preview = text if len(text) <= 20 else text[:20] + "..."
        if not self.learn_from_chat:
            if not self.dataset["pairs"]:
                return self._finish(
                    "아직 배운 게 없어. 원본 파일에 정보를 적고 '/새로고침' 해줘.\n  {}".format(
                        self.source_path
                    ),
                    "fallback",
                    0.0,
                )
            return self._finish(
                "'{}' 은(는) 내 파일에 없는 내용이야. 원본 파일에 적어주면 배울게.\n  {}".format(
                    preview, self.source_path
                ),
                "fallback",
                0.0,
            )
        return self._finish(self.rng.choice(_FALLBACKS).format(q=preview), "fallback", 0.0)

    # ------------------------------------------------------------ 조회

    def imagine(self, seed=None):
        """마르코프 체인으로 아무 말이나 지어낸다."""
        for _ in range(6):
            sentence = self.markov.generate(seed=seed)
            if sentence and sentence != seed:
                return sentence
        return "아직 배운 문장이 부족해서 상상이 안 돼."

    def explain(self, text):
        """디버그용: 어떤 후보들이 몇 점이었는지."""
        rows = self.index.search(text, top_k=5)
        lines = []
        if rows:
            lines.append("[유사도 검색]")
            lines.extend(
                "  {:.3f}  {}  <- {}".format(score, document, payload[0])
                for score, document, payload in rows
            )
        else:
            lines.append("[유사도 검색] 후보 없음")

        prediction = self.classifier.predict(features(text))
        lines.append(
            "[분류기] {}".format(
                "{} ({:.0%})".format(*prediction) if prediction else "판단 보류 (근거 부족)"
            )
        )
        lines.append("[토큰] {}".format(", ".join(features(text)) or "없음"))
        return "\n".join(lines)

    def progress(self):
        """학습 현황 요약."""
        stats = self.memory["stats"]
        auto = sum(len(v) for v in self.memory["patterns"].values())
        answered = stats["matched"]
        turns = max(stats["turns"], 1)

        lines = [
            "[대화] 총 {}턴 · 응답 {}회 · 모름 {}회 · 응답률 {:.0%}".format(
                stats["turns"], answered, stats["unknown"], answered / turns
            ),
            "[출처] {} · 대화 학습 {}".format(
                "내가 준 파일만 (내장 지식 꺼짐)" if self.strict else "내장 지식 + 내 파일",
                "켜짐" if self.learn_from_chat else "꺼짐 (파일로만 학습)",
            ),
            "[원본] {}".format(self.source_path or "지정 안 됨"),
            "[데이터셋] 사실 {}개 · 문답 {}개 · 문장 {}개".format(
                len(self.dataset["facts"]),
                len(self.dataset["pairs"]),
                len(self.dataset["sentences"]),
            ),
            "[지식] 기본 의도 {}개 · 가르친 문답 {}개 · 스스로 익힌 표현 {}개".format(
                len(self.intents), len(self.memory["learned"]), auto
            ),
            "[분류기] 라벨 {}개 · 예문 {}개 · 어휘 {}개".format(
                len(self.classifier.classes),
                sum(self.classifier.class_counts.values()),
                len(self.classifier.vocabulary),
            ),
            "[생성기] 학습 문장 {}개".format(self.markov.trained),
        ]

        if self.memory["patterns"]:
            lines.append("[스스로 익힌 표현]")
            for name, texts in self.memory["patterns"].items():
                lines.append("  {} <- {}".format(name, ", ".join(texts)))

        top = self.ranker.top(3)
        if top:
            lines.append("[선호 응답]")
            lines.extend("  {:.2f}  {}".format(weight, pair[-1]) for pair, weight in top)
        return "\n".join(lines)

"""대화 엔진 본체.

응답 순서:
  1) 규칙 기반 스킬 (계산, 시간, 기억 ...)
  2) TF-IDF 유사도 검색 (기본 지식 + 사용자가 가르친 문장)
  3) 애매하면 되묻기, 전혀 모르면 학습 유도
"""

import json
import os
import random
from collections import namedtuple

from . import skills
from .markov import MarkovChain
from .tokenizer import features, normalize
from .vectorizer import TfidfIndex

Reply = namedtuple("Reply", "text source score")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_KNOWLEDGE = os.path.join(DATA_DIR, "knowledge.json")
DEFAULT_MEMORY = os.path.join(DATA_DIR, "memory.json")

# 이 아래면 "모르겠다", 이 위면 확신, 사이면 되묻는다.
CONFIDENT = 0.30
UNSURE = 0.13

_FALLBACKS = [
    "음, 그건 아직 모르겠어. '/배워 {q} | 답변' 으로 알려주면 외울게.",
    "처음 듣는 얘기야. 뭐라고 답하면 좋을지 가르쳐줄래? (/배워 {q} | 답변)",
    "그 말은 내 지식에 없네. '/배워' 로 알려주면 다음엔 대답할 수 있어.",
]


class Brain:
    def __init__(self, knowledge_path=DEFAULT_KNOWLEDGE, memory_path=DEFAULT_MEMORY, rng=None):
        self.knowledge_path = knowledge_path
        self.memory_path = memory_path
        self.rng = rng or random.Random()

        self.intents = []
        self.memory = {"facts": {}, "learned": []}
        self.index = TfidfIndex()
        self.markov = MarkovChain(order=2, rng=self.rng)

        self.last_user_message = ""
        self._last_reply = None
        self.awaiting_answer_for = None  # 대화형 학습 상태

        self._load_knowledge()
        self._load_memory()
        self.reindex()

    # ------------------------------------------------------------ 적재/저장

    def _load_knowledge(self):
        with open(self.knowledge_path, encoding="utf-8") as handle:
            self.intents = json.load(handle).get("intents", [])

    def _load_memory(self):
        if not os.path.exists(self.memory_path):
            return
        try:
            with open(self.memory_path, encoding="utf-8") as handle:
                stored = json.load(handle)
        except (ValueError, OSError):
            return  # 손상된 기억 파일 때문에 실행이 막히면 안 된다.
        self.memory["facts"] = dict(stored.get("facts", {}))
        self.memory["learned"] = [
            item for item in stored.get("learned", [])
            if isinstance(item, dict) and item.get("question") and item.get("answer")
        ]

    def save(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        temp_path = self.memory_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(self.memory, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.memory_path)  # 저장 중 중단돼도 기존 파일은 살아남는다.

    # ------------------------------------------------------------ 색인

    def reindex(self):
        self.index = TfidfIndex()
        corpus = []

        for position, intent in enumerate(self.intents):
            for pattern in intent.get("patterns", []):
                self.index.add(pattern, ("intent", position))
            corpus.extend(intent.get("patterns", []))
            corpus.extend(intent.get("responses", []))

        for position, item in enumerate(self.memory["learned"]):
            self.index.add(item["question"], ("learned", position))
            corpus.append(item["question"])
            corpus.append(item["answer"])

        self.index.build()
        self.markov = MarkovChain(order=2, rng=self.rng)
        self.markov.train(corpus)

    # ------------------------------------------------------------ 기억 API

    def remember(self, key, value):
        self.memory["facts"][normalize(key)] = value

    def recall(self, key):
        return self.memory["facts"].get(normalize(key))

    def forget(self, key=None):
        if key is None:
            count = len(self.memory["facts"]) + len(self.memory["learned"])
            self.memory["facts"].clear()
            self.memory["learned"].clear()
            self.reindex()
            return count
        key = normalize(key)
        if key in self.memory["facts"]:
            del self.memory["facts"][key]
            return 1
        before = len(self.memory["learned"])
        self.memory["learned"] = [
            item for item in self.memory["learned"] if normalize(item["question"]) != key
        ]
        removed = before - len(self.memory["learned"])
        if removed:
            self.reindex()
        return removed

    def learn(self, question, answer):
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            raise ValueError("질문과 답변이 모두 필요해")
        key = normalize(question)
        for item in self.memory["learned"]:
            if normalize(item["question"]) == key:
                item["answer"] = answer
                self.reindex()
                return False  # 갱신
        self.memory["learned"].append({"question": question, "answer": answer})
        self.reindex()
        return True  # 신규

    # ------------------------------------------------------------ 응답

    def _payload_answer(self, payload):
        kind, position = payload
        if kind == "learned":
            return self.memory["learned"][position]["answer"]
        responses = self.intents[position].get("responses") or ["..."]
        if len(responses) > 1 and self._last_reply in responses:
            responses = [r for r in responses if r != self._last_reply]
        return self.rng.choice(responses)

    def respond(self, text):
        """사용자 발화 하나에 대한 Reply를 만든다."""
        text = (text or "").strip()
        if not text:
            return Reply("뭐라고? 아무것도 안 들렸어.", "empty", 0.0)

        # 대화형 학습 중이면 이번 입력을 답변으로 받는다.
        if self.awaiting_answer_for is not None:
            question, self.awaiting_answer_for = self.awaiting_answer_for, None
            self.learn(question, text)
            return Reply("외웠어. 이제 '{}' 라고 물으면 그렇게 답할게.".format(question), "learn", 1.0)

        self.last_user_message = text

        for skill in skills.ALL_SKILLS:
            answer = skill(text, self)
            if answer:
                self._last_reply = answer
                return Reply(answer, "skill:" + skill.__name__, 1.0)

        results = self.index.search(text, top_k=3)
        if results:
            score, matched, payload = results[0]
            if score >= CONFIDENT:
                answer = self._payload_answer(payload)
                self._last_reply = answer
                return Reply(answer, "match:" + matched, score)
            if score >= UNSURE:
                self._last_reply = None
                return Reply(
                    "혹시 '{}' 얘기야? 맞으면 그대로 다시 말해줘.".format(matched),
                    "clarify:" + matched,
                    score,
                )

        self._last_reply = None
        preview = text if len(text) <= 20 else text[:20] + "..."
        return Reply(self.rng.choice(_FALLBACKS).format(q=preview), "fallback", 0.0)

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
        if not rows:
            return "매칭된 후보 없음 (토큰: {})".format(", ".join(features(text)) or "없음")
        return "\n".join(
            "  {:.3f}  {}  <- {}".format(score, document, payload[0])
            for score, document, payload in rows
        )

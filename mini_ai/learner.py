"""온라인 학습 부품들.

- NaiveBayesClassifier: 예문이 쌓일수록 좋아지는 의도 분류기.
  TF-IDF 검색이 애매할 때 두 번째 의견 역할을 한다.
- ResponseRanker: 사용자 피드백으로 응답별 가중치를 조절한다.
"""

import math
import random
from collections import Counter, defaultdict

from .tokenizer import features

SEPARATOR = "\x1f"

# ingest._question_forms 가 사실 하나를 질문 네 개로 부풀릴 때 갖다 붙이는
# 의문사 껍데기들. 이건 내가 만든 문법이지 사용자가 준 정보가 아니므로,
# 분류기가 근거로 삼으면 안 된다. "뭐야" 하나만 겹쳤다고 답을 내놓으면
# 파일에 없는 질문까지 아무 답에나 붙어버린다.
QUESTION_SCAFFOLD = frozenset(
    features("뭐야") + features("알려줘") + features("에 대해 설명해줘")
)


class NaiveBayesClassifier:
    """다항 나이브 베이즈. 라플라스 스무딩을 쓰고 전부 로그 공간에서 센다."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.class_counts = Counter()          # 라벨별 예문 수
        self.token_counts = defaultdict(Counter)  # 라벨 -> 토큰 -> 빈도
        self.class_totals = Counter()          # 라벨별 전체 토큰 수
        self.vocabulary = set()

    def train(self, tokens, label):
        tokens = [t for t in tokens if t]
        if not tokens:
            return
        self.class_counts[label] += 1
        counter = self.token_counts[label]
        for token in tokens:
            counter[token] += 1
            self.vocabulary.add(token)
        self.class_totals[label] += len(tokens)

    def train_many(self, examples):
        for tokens, label in examples:
            self.train(tokens, label)
        return self

    @property
    def classes(self):
        return list(self.class_counts)

    def predict(self, tokens, min_margin=0.08):
        """(라벨, 확신도)를 돌려준다. 근거가 부족하면 None.

        어휘에 없는 토큰은 점수 계산에서 아예 빼고, 남은 것 중
        의문사 껍데기가 아닌 토큰이 하나도 없으면 판단을 포기한다.
        """
        if len(self.class_counts) < 2:
            # 라벨이 하나뿐이면 정규화 확률이 무조건 1.0 이라 무엇을 물어도
            # 그 답을 내놓는다. 비교할 대상이 없으면 분류기는 침묵해야 한다.
            return None

        known = [t for t in tokens if t in self.vocabulary]
        # "사는 곳이 뭐야" 처럼 내용어는 전부 처음 보고 "뭐야" 만 겹치는 경우를
        # 걸러낸다. 껍데기를 뺀 실질 근거가 없으면 아는 척하지 않는다.
        if not {t for t in known if t not in QUESTION_SCAFFOLD}:
            return None

        total_docs = sum(self.class_counts.values())
        vocabulary_size = len(self.vocabulary)
        scores = {}
        for label, count in self.class_counts.items():
            score = math.log(count / total_docs)
            denominator = self.class_totals[label] + self.alpha * vocabulary_size
            counter = self.token_counts[label]
            for token in known:
                score += math.log((counter[token] + self.alpha) / denominator)
            scores[label] = score

        # log-sum-exp 로 정규화해서 확률처럼 읽을 수 있게 만든다.
        best = max(scores.values())
        exponentials = {label: math.exp(score - best) for label, score in scores.items()}
        total = sum(exponentials.values())
        probabilities = sorted(
            ((value / total, label) for label, value in exponentials.items()), reverse=True
        )

        top_probability, top_label = probabilities[0]
        runner_up = probabilities[1][0] if len(probabilities) > 1 else 0.0
        if top_probability - runner_up < min_margin:
            return None  # 1, 2위가 붙어 있으면 찍는 것과 다름없다.
        return top_label, top_probability


class ResponseRanker:
    """응답마다 가중치를 두고, 칭찬받은 답은 더 자주 고른다."""

    UP = 1.6
    DOWN = 0.35
    CEILING = 8.0
    FLOOR = 0.05

    def __init__(self, weights=None, rng=None):
        self.weights = dict(weights or {})
        self.rng = rng or random.Random()

    @staticmethod
    def key(label, response):
        return "{}{}{}".format(label, SEPARATOR, response)

    def weight(self, label, response):
        return self.weights.get(self.key(label, response), 1.0)

    def reward(self, label, response):
        key = self.key(label, response)
        self.weights[key] = min(self.CEILING, self.weights.get(key, 1.0) * self.UP)
        return self.weights[key]

    def penalize(self, label, response):
        key = self.key(label, response)
        self.weights[key] = max(self.FLOOR, self.weights.get(key, 1.0) * self.DOWN)
        return self.weights[key]

    def choose(self, label, responses, avoid=None):
        """가중 무작위 선택. 직전 답변은 가능하면 피한다."""
        pool = [r for r in responses if r != avoid] or list(responses)
        weights = [self.weight(label, response) for response in pool]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(pool)
        threshold = self.rng.uniform(0, total)
        running = 0.0
        for response, weight in zip(pool, weights):
            running += weight
            if running >= threshold:
                return response
        return pool[-1]

    def top(self, limit=5):
        rows = sorted(self.weights.items(), key=lambda row: row[1], reverse=True)
        return [(key.split(SEPARATOR), weight) for key, weight in rows[:limit]]

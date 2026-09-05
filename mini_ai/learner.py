"""온라인 학습 부품들.

- NaiveBayesClassifier: 예문이 쌓일수록 좋아지는 의도 분류기.
  TF-IDF 검색이 애매할 때 두 번째 의견 역할을 한다.
- ResponseRanker: 사용자 피드백으로 응답별 가중치를 조절한다.
"""

import math
import random
from collections import Counter, defaultdict

SEPARATOR = "\x1f"


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

    def predict(self, tokens, min_known=2, min_margin=0.08):
        """(라벨, 확신도)를 돌려준다. 근거가 부족하면 None.

        어휘에 없는 토큰은 아예 세지 않는다. 대신 아는 토큰이
        min_known개 미만이면 판단을 포기한다 - 모르는 문장에
        억지로 라벨을 붙이는 걸 막는 장치다.
        """
        if not self.class_counts:
            return None
        known = [t for t in tokens if t in self.vocabulary]
        if len(known) < min_known:
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

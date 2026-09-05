"""학습한 문장들로 새 문장을 지어내는 마르코프 체인 생성기.

검색이 실패했을 때 억지로 쓰기보다는 '/상상' 명령의 놀이용으로 쓴다.
"""

import random
from collections import defaultdict

_START = "\x02"
_END = "\x03"


class MarkovChain:
    def __init__(self, order=2, rng=None):
        self.order = order
        self.rng = rng or random.Random()
        self._table = defaultdict(list)
        self._trained = 0

    def train(self, sentences):
        for sentence in sentences:
            tokens = sentence.split()
            if len(tokens) < 2:
                continue
            padded = [_START] * self.order + tokens + [_END]
            for i in range(len(padded) - self.order):
                state = tuple(padded[i : i + self.order])
                self._table[state].append(padded[i + self.order])
                # 낮은 차수 상태도 같이 저장해두면 막다른 길에서 빠져나올 수 있다.
                if self.order > 1:
                    self._table[state[1:]].append(padded[i + self.order])
            self._trained += 1
        return self

    @property
    def trained(self):
        return self._trained

    def _next(self, state):
        choices = self._table.get(state)
        if not choices and len(state) > 1:
            choices = self._table.get(state[1:])
        if not choices:
            return None
        return self.rng.choice(choices)

    def generate(self, seed=None, max_words=24):
        """seed 단어로 시작하는 문장을 만든다. 못 만들면 None."""
        if not self._table:
            return None

        if seed:
            seed_tokens = seed.split()[-self.order :]
            state = tuple([_START] * (self.order - len(seed_tokens)) + seed_tokens)
            output = list(seed_tokens)
            if state not in self._table and state[1:] not in self._table:
                state = tuple([_START] * self.order)
                output = []
        else:
            state = tuple([_START] * self.order)
            output = []

        for _ in range(max_words):
            token = self._next(state)
            if token is None or token == _END:
                break
            output.append(token)
            state = tuple(list(state)[1:] + [token])

        sentence = " ".join(t for t in output if t not in (_START, _END)).strip()
        return sentence or None

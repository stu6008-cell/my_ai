"""외부 라이브러리 없이 만든 TF-IDF + 코사인 유사도 검색기."""

import math
from collections import Counter

from .tokenizer import features

# 커버리지 벌점의 세기. 라벨링한 문장으로 0 / 0.25 / 0.5 를 비교해서 고른 값이다.
# 0 이면 "주식 전망 알려줘"가 파이썬 설명에 붙고, 0.5 면 정상적인 의역까지 떨어진다.
COVERAGE_EXPONENT = 0.25


class TfidfIndex:
    """문서 목록을 받아 질의와 가장 비슷한 문서를 찾아준다."""

    def __init__(self):
        self.documents = []      # 원본 문자열
        self.payloads = []       # 문서에 딸린 임의의 값 (의도 id 등)
        self._vectors = []       # 정규화된 tf-idf 벡터
        self._idf = {}
        self._default_idf = 1.0
        self._doc_terms = []

    def __len__(self):
        return len(self.documents)

    def add(self, text, payload=None):
        """문서를 색인에 넣는다. 넣은 뒤에는 build()를 호출해야 반영된다."""
        self.documents.append(text)
        self.payloads.append(payload)
        self._doc_terms.append(Counter(features(text)))

    def build(self):
        """IDF를 계산하고 모든 문서 벡터를 만든다."""
        total = len(self._doc_terms)
        document_frequency = Counter()
        for terms in self._doc_terms:
            document_frequency.update(terms.keys())

        # smoothed idf: 문서가 하나뿐일 때도 0으로 죽지 않는다.
        self._idf = {
            term: math.log((total + 1) / (count + 1)) + 1.0
            for term, count in document_frequency.items()
        }
        # 색인에 없는 단어는 "아주 드문 단어"로 취급한다. 그냥 버리면
        # 모르는 질문일수록 남은 흔한 단어끼리 100% 일치해버린다.
        self._default_idf = math.log(total + 1) + 1.0
        self._vectors = [self._to_vector(terms) for terms in self._doc_terms]

    def _to_vector(self, terms):
        if not terms:
            return {}
        max_count = max(terms.values())
        vector = {}
        for term, count in terms.items():
            idf = self._idf.get(term, self._default_idf)
            # 서브리니어 TF 대신 정규화 TF: 짧은 문장이 많아 이쪽이 안정적이다.
            vector[term] = (0.5 + 0.5 * count / max_count) * idf
        norm = math.sqrt(sum(w * w for w in vector.values()))
        if norm == 0:
            return {}
        return {term: w / norm for term, w in vector.items()}

    def search(self, query, top_k=3):
        """(점수, 문서, payload) 목록을 점수 내림차순으로 돌려준다."""
        if not self._vectors:
            return []
        query_vector = self._to_vector(Counter(features(query)))
        if not query_vector:
            return []

        scored = []
        for index, vector in enumerate(self._vectors):
            if len(query_vector) > len(vector):
                shorter, longer = vector, query_vector
            else:
                shorter, longer = query_vector, vector

            cosine = 0.0
            for term, weight in shorter.items():
                other = longer.get(term)
                if other:
                    cosine += weight * other
            if cosine <= 0:
                continue

            # 코사인만 쓰면 짧은 문서가 유리하다. "양자역학이 뭐야"가 "너는 뭐야"에
            # 걸리는 걸 막으려고, 질의 쪽이 얼마나 설명됐는지를 함께 곱한다.
            covered = sum(
                query_vector[term] ** 2 for term in query_vector if term in vector
            )
            score = cosine * (covered ** COVERAGE_EXPONENT)
            scored.append((score, self.documents[index], self.payloads[index]))

        scored.sort(key=lambda row: row[0], reverse=True)
        return scored[:top_k]

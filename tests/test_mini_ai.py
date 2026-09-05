"""표준 unittest만 사용하는 테스트. python -m unittest discover -s tests 로 실행."""

import json
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_ai import ingest
from mini_ai.brain import CONFIDENT, Brain
from mini_ai.cli import handle_command
from mini_ai.learner import NaiveBayesClassifier, ResponseRanker
from mini_ai.markov import MarkovChain
from mini_ai.skills import safe_eval
from mini_ai.tokenizer import bigrams, features, stem, words
from mini_ai.vectorizer import TfidfIndex

SAMPLE_SOURCE = """\
# 주석은 무시된다
좋아하는 음식: 마라탕
회사 출근 시간: 오전 9시 30분
Q: 우리 강아지 이름이 뭐야
A: 콩이야
주말에 뭐 할까 | 한강 가서 자전거 타자
파이썬은 배우기 쉬운 프로그래밍 언어이다
오늘은 비가 와서 집에 있었다
"""


class TokenizerTest(unittest.TestCase):
    def test_words_splits_scripts(self):
        self.assertEqual(words("안녕! python 3 이야"), ["안녕", "python", "3", "이야"])

    def test_stem_strips_one_suffix(self):
        self.assertEqual(stem("반갑습니다"), "반갑")
        self.assertEqual(stem("학교에서"), "학교")
        self.assertEqual(stem("대단한"), "대단")

    def test_stem_keeps_short_words(self):
        self.assertEqual(stem("안녕"), "안녕")
        self.assertEqual(stem("hello"), "hello")

    def test_bigrams(self):
        self.assertEqual(bigrams("반갑다"), ["반갑", "갑다"])
        self.assertEqual(bigrams("a"), [])

    def test_features_include_stem_and_bigrams(self):
        result = features("반갑습니다")
        self.assertIn("반갑습니다", result)
        self.assertIn("반갑", result)
        self.assertIn("습니", result)


class VectorizerTest(unittest.TestCase):
    def setUp(self):
        self.index = TfidfIndex()
        for text, tag in [("안녕하세요", "인사"), ("배고파 밥 먹자", "식사"), ("파이썬 공부 중이야", "공부")]:
            self.index.add(text, tag)
        self.index.build()

    def test_exact_match_scores_highest(self):
        score, _, payload = self.index.search("안녕하세요")[0]
        self.assertEqual(payload, "인사")
        self.assertGreater(score, 0.9)

    def test_partial_match_still_found(self):
        self.assertEqual(self.index.search("밥 먹자")[0][2], "식사")

    def test_unknown_query_scores_low(self):
        results = self.index.search("양자역학 논문 리뷰")
        self.assertTrue(not results or results[0][0] < CONFIDENT)

    def test_empty_index_returns_nothing(self):
        self.assertEqual(TfidfIndex().search("아무거나"), [])


class SafeEvalTest(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(safe_eval("2 + 3 * 4"), 14)
        self.assertEqual(safe_eval("(1+1) ** 10"), 1024)
        self.assertEqual(safe_eval("7 / 2"), 3.5)

    def test_rejects_code_execution(self):
        for bad in ["__import__('os').system('ls')", "open('x')", "1 if True else 2"]:
            with self.assertRaises((ValueError, SyntaxError)):
                safe_eval(bad)


class MarkovTest(unittest.TestCase):
    def test_generates_from_training_data(self):
        chain = MarkovChain(order=2, rng=random.Random(0))
        chain.train(["나는 오늘 밥을 먹었다", "나는 내일 책을 읽는다"] * 5)
        self.assertEqual(chain.trained, 10)
        self.assertTrue(chain.generate())

    def test_untrained_returns_none(self):
        self.assertIsNone(MarkovChain().generate())


class LearnerTest(unittest.TestCase):
    def setUp(self):
        self.classifier = NaiveBayesClassifier()
        self.classifier.train_many([
            (["안녕", "하이", "인사"], "인사"),
            (["안녕", "반가워"], "인사"),
            (["배고파", "밥", "먹자"], "식사"),
            (["점심", "밥", "뭐"], "식사"),
        ])

    def test_predicts_trained_class(self):
        label, probability = self.classifier.predict(["안녕", "반가워"])
        self.assertEqual(label, "인사")
        self.assertGreater(probability, 0.5)

    def test_abstains_without_enough_known_tokens(self):
        self.assertIsNone(self.classifier.predict(["양자", "역학"]))
        self.assertIsNone(self.classifier.predict(["안녕"]))

    def test_untrained_classifier_abstains(self):
        self.assertIsNone(NaiveBayesClassifier().predict(["아무", "말"]))

    def test_ranker_reward_and_penalty(self):
        ranker = ResponseRanker(rng=random.Random(0))
        self.assertEqual(ranker.weight("a", "x"), 1.0)
        self.assertGreater(ranker.reward("a", "x"), 1.0)
        self.assertLess(ranker.penalize("a", "y"), 1.0)

    def test_ranker_weights_are_bounded(self):
        ranker = ResponseRanker(rng=random.Random(0))
        for _ in range(50):
            ranker.reward("a", "x")
            ranker.penalize("a", "y")
        self.assertLessEqual(ranker.weight("a", "x"), ResponseRanker.CEILING)
        self.assertGreaterEqual(ranker.weight("a", "y"), ResponseRanker.FLOOR)


class IngestTest(unittest.TestCase):
    def test_key_value_becomes_fact_and_questions(self):
        parsed = ingest.parse("좋아하는 음식: 마라탕")
        self.assertEqual(parsed["facts"]["좋아하는 음식"], "마라탕")
        questions = [item["question"] for item in parsed["pairs"]]
        self.assertIn("좋아하는 음식이 뭐야", questions)
        self.assertTrue(all(item["answer"] == "마라탕" for item in parsed["pairs"]))

    def test_colon_wins_over_definition(self):
        # "좋아하는 음식" 의 '는' 때문에 정의문으로 잘못 읽히면 안 된다.
        parsed = ingest.parse("좋아하는 음식: 마라탕")
        self.assertNotIn("좋아하", parsed["facts"])

    def test_question_answer_block(self):
        parsed = ingest.parse("Q: 강아지 이름\nA: 콩이야")
        self.assertEqual(parsed["pairs"][0]["answer"], "콩이야")

    def test_pipe_line(self):
        parsed = ingest.parse("주말에 뭐 할까 | 한강 가자")
        self.assertEqual(parsed["pairs"][0]["question"], "주말에 뭐 할까")

    def test_definition_requires_copula(self):
        # 서술문은 사실이 아니라 문장으로만 남아야 한다.
        parsed = ingest.parse("오늘은 날씨가 좋았다")
        self.assertEqual(parsed["facts"], {})
        self.assertIn("오늘은 날씨가 좋았다", parsed["sentences"])

    def test_definition_with_copula(self):
        parsed = ingest.parse("파이썬은 쉬운 언어이다")
        self.assertEqual(parsed["facts"]["파이썬"], "쉬운 언어")

    def test_josa_follows_batchim(self):
        self.assertIn("수도가 뭐야", [p["question"] for p in ingest.parse("수도: 서울")["pairs"]])
        self.assertIn("이름이 뭐야", [p["question"] for p in ingest.parse("이름: 콩이")["pairs"]])

    def test_comments_ignored(self):
        self.assertEqual(ingest.parse("# 주석\n\n"), {"facts": {}, "pairs": [], "sentences": []})

    def test_merge_updates_existing_answer(self):
        dataset = {"facts": {}, "pairs": [], "sentences": []}
        ingest.merge(dataset, ingest.parse("Q: 색\nA: 빨강"))
        added = ingest.merge(dataset, ingest.parse("Q: 색\nA: 파랑"))
        self.assertEqual(added["pairs"], 0)
        self.assertEqual(dataset["pairs"][0]["answer"], "파랑")


class SourceCompileTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.source = os.path.join(self.tempdir.name, "source.txt")
        self.dataset = os.path.join(self.tempdir.name, "dataset.json")

    def test_compiles_text_to_json(self):
        with open(self.source, "w", encoding="utf-8") as handle:
            handle.write(SAMPLE_SOURCE)
        dataset, added = ingest.compile_source(self.source, self.dataset)
        self.assertTrue(os.path.exists(self.dataset))
        self.assertGreater(added["pairs"], 0)
        with open(self.dataset, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["facts"]["좋아하는 음식"], "마라탕")

    def test_accepts_json_source(self):
        with open(self.source, "w", encoding="utf-8") as handle:
            json.dump({"facts": {"수도": "서울"}, "pairs": []}, handle, ensure_ascii=False)
        dataset, _ = ingest.compile_source(self.source, self.dataset)
        self.assertIn("수도가 뭐야", [p["question"] for p in dataset["pairs"]])

    def test_bad_json_raises(self):
        with open(self.source, "w", encoding="utf-8") as handle:
            handle.write("{ broken json")
        with self.assertRaises(ValueError):
            ingest.compile_source(self.source, self.dataset)

    def test_removed_lines_disappear(self):
        # 원본이 유일한 진실이므로, 지운 내용은 다음 컴파일에서 사라져야 한다.
        with open(self.source, "w", encoding="utf-8") as handle:
            handle.write("색: 파랑\n맛: 단맛\n")
        ingest.compile_source(self.source, self.dataset)
        with open(self.source, "w", encoding="utf-8") as handle:
            handle.write("색: 파랑\n")
        dataset, _ = ingest.compile_source(self.source, self.dataset)
        self.assertNotIn("맛", dataset["facts"])

    def test_missing_source_is_empty(self):
        dataset, added = ingest.compile_source(self.source, self.dataset)
        self.assertEqual(dataset["pairs"], [])
        self.assertEqual(added["pairs"], 0)


class FileOnlyBrainTest(unittest.TestCase):
    """기본 정책: 내장 지식 없음, 대화 학습 없음, 원본 파일만."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.source = os.path.join(self.tempdir.name, "source.txt")
        with open(self.source, "w", encoding="utf-8") as handle:
            handle.write(SAMPLE_SOURCE)
        self.bot = self._brain()

    def _brain(self):
        return Brain(
            memory_path=os.path.join(self.tempdir.name, "memory.json"),
            dataset_path=os.path.join(self.tempdir.name, "dataset.json"),
            source_path=self.source,
            rng=random.Random(1234),
        )

    def test_answers_from_file(self):
        self.assertEqual(self.bot.respond("좋아하는 음식이 뭐야").text, "마라탕")
        self.assertEqual(self.bot.respond("우리 강아지 이름이 뭐야").text, "콩이야")
        self.assertEqual(self.bot.respond("주말에 뭐 할까").text, "한강 가서 자전거 타자")

    def test_generalizes_to_unseen_phrasing(self):
        reply = self.bot.respond("출근 몇시에 해")
        self.assertEqual(reply.text, "오전 9시 30분")

    def test_no_builtin_knowledge(self):
        self.assertEqual(self.bot.intents, [])
        self.assertEqual(self.bot.respond("안녕하세요").source, "fallback")

    def test_user_data_outranks_builtin_skill(self):
        # 파일에 출근 시간이 있으면 시계 스킬이 가로채면 안 된다.
        self.assertNotIn("분이야", self.bot.respond("출근 몇시에 해").text)

    def test_skills_still_compute(self):
        self.assertIn("= 84", self.bot.respond("12 * 7 은 뭐야").text)
        self.assertIn("분이야", self.bot.respond("지금 몇시").text)

    def test_chat_teaching_is_refused(self):
        with self.assertRaises(PermissionError):
            self.bot.learn("질문", "답")

    def test_chat_facts_are_refused(self):
        self.bot.respond("내 이름은 지훈이야")
        self.assertIsNone(self.bot.recall("이름"))

    def test_feedback_is_inert(self):
        self.bot.respond("좋아하는 음식이 뭐야")
        self.assertIsNone(self.bot.feedback(True))

    def test_absorb_is_refused(self):
        self.assertFalse(self.bot.absorb("아무 말", "아무 라벨"))

    def test_confirmation_does_not_teach(self):
        before = len(self.bot.dataset["pairs"])
        self.bot.pending_confirm = ("새로운 표현", ("dataset", 0))
        self.bot.respond("응")
        self.assertEqual(len(self.bot.dataset["pairs"]), before)
        self.assertEqual(self.bot.memory["learned"], [])

    def test_reload_picks_up_edits(self):
        self.assertEqual(self.bot.respond("취미가 뭐야").source, "fallback")
        with open(self.source, "a", encoding="utf-8") as handle:
            handle.write("취미: 등산\n")
        added, error = self.bot.reload_source()
        self.assertIsNone(error)
        self.assertEqual(self.bot.respond("취미가 뭐야").text, "등산")

    def test_reload_reports_bad_json(self):
        with open(self.source, "w", encoding="utf-8") as handle:
            handle.write("{ broken")
        _, error = self.bot.reload_source()
        self.assertIsNotNone(error)

    def test_dataset_persists_across_restart(self):
        self.assertEqual(self._brain().respond("좋아하는 음식이 뭐야").text, "마라탕")

    def test_empty_input(self):
        self.assertEqual(self.bot.respond("   ").source, "empty")

    def test_corrupt_memory_file_does_not_crash(self):
        with open(self.bot.memory_path, "w", encoding="utf-8") as handle:
            handle.write("{ this is not json")
        self.assertEqual(self._brain().respond("좋아하는 음식이 뭐야").text, "마라탕")


class ChatLearningBrainTest(unittest.TestCase):
    """--chat-learning 을 켰을 때만 동작해야 하는 경로들."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.bot = Brain(
            memory_path=os.path.join(self.tempdir.name, "memory.json"),
            dataset_path=os.path.join(self.tempdir.name, "dataset.json"),
            source_path=os.path.join(self.tempdir.name, "missing.txt"),
            strict=False,
            learn_from_chat=True,
            rng=random.Random(1234),
        )

    def test_builtin_knowledge_available(self):
        reply = self.bot.respond("안녕하세요")
        self.assertGreaterEqual(reply.score, CONFIDENT)

    def test_teach_and_answer(self):
        self.bot.learn("좋아하는 색이 뭐야", "파란색")
        self.assertEqual(self.bot.respond("좋아하는 색이 뭐야").text, "파란색")

    def test_relearning_replaces_answer(self):
        self.assertTrue(self.bot.learn("취미가 뭐야", "코딩"))
        self.assertFalse(self.bot.learn("취미가 뭐야", "등산"))
        self.assertEqual(self.bot.respond("취미가 뭐야").text, "등산")

    def test_name_memory(self):
        self.bot.respond("내 이름은 지훈이야")
        self.assertEqual(self.bot.recall("이름"), "지훈")

    def test_confirmation_learns_new_phrasing(self):
        self.bot.pending_confirm = ("잘 지내셨어요", ("intent", 0))
        self.bot.respond("응")
        self.assertIn("잘 지내셨어요", self.bot.memory["patterns"].get("인사", []))

    def test_feedback_rewards_response(self):
        self.bot.respond("안녕하세요")
        result = self.bot.feedback(True)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "up")
        self.assertGreater(result[2], 1.0)


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.source = os.path.join(self.tempdir.name, "source.txt")
        with open(self.source, "w", encoding="utf-8") as handle:
            handle.write(SAMPLE_SOURCE)
        self.bot = Brain(
            memory_path=os.path.join(self.tempdir.name, "memory.json"),
            dataset_path=os.path.join(self.tempdir.name, "dataset.json"),
            source_path=self.source,
            rng=random.Random(7),
        )

    def test_plain_text_is_not_a_command(self):
        self.assertIsNone(handle_command(self.bot, "안녕"))

    def test_unknown_command(self):
        self.assertIn("그런 명령은 없어", handle_command(self.bot, "/없는명령"))

    def test_reload_command(self):
        self.assertIn("다시 학습했어", handle_command(self.bot, "/새로고침"))

    def test_data_command_lists_learned_content(self):
        self.assertIn("마라탕", handle_command(self.bot, "/데이터"))

    def test_source_command_shows_path(self):
        self.assertIn(self.source, handle_command(self.bot, "/원본"))

    def test_train_command_switches_source(self):
        other = os.path.join(self.tempdir.name, "other.txt")
        with open(other, "w", encoding="utf-8") as handle:
            handle.write("수도: 서울\n")
        handle_command(self.bot, "/학습 " + other)
        self.assertEqual(self.bot.respond("수도가 뭐야").text, "서울")

    def test_train_command_rejects_missing_file(self):
        self.assertIn("그런 파일이 없어", handle_command(self.bot, "/학습 /없는/경로.txt"))

    def test_learn_command_blocked_by_policy(self):
        self.assertIn("대화로는 안 배우도록", handle_command(self.bot, "/배워 질문 | 답변"))
        self.assertEqual(self.bot.memory["learned"], [])

    def test_feedback_blocked_by_policy(self):
        self.assertIn("대화로는 안 배우도록", handle_command(self.bot, "/좋아"))

    def test_stats_command(self):
        self.assertIn("대화 학습 꺼짐", handle_command(self.bot, "/학습현황"))

    def test_debug_command_requires_argument(self):
        self.assertIn("확인할 문장", handle_command(self.bot, "/디버그"))


if __name__ == "__main__":
    unittest.main()

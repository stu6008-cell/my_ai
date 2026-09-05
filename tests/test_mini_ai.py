"""표준 unittest만 사용하는 테스트. python -m unittest discover 로 실행."""

import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_ai.brain import CONFIDENT, Brain
from mini_ai.cli import handle_command
from mini_ai.markov import MarkovChain
from mini_ai.skills import safe_eval
from mini_ai.tokenizer import bigrams, features, stem, words
from mini_ai.vectorizer import TfidfIndex


class TokenizerTest(unittest.TestCase):
    def test_words_splits_scripts(self):
        self.assertEqual(words("안녕! python 3 이야"), ["안녕", "python", "3", "이야"])

    def test_stem_strips_one_suffix(self):
        self.assertEqual(stem("반갑습니다"), "반갑")
        self.assertEqual(stem("학교에서"), "학교")

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
        score, document, payload = self.index.search("안녕하세요")[0]
        self.assertEqual(payload, "인사")
        self.assertGreater(score, 0.9)

    def test_partial_match_still_found(self):
        self.assertEqual(self.index.search("밥 먹자")[0][2], "식사")

    def test_unknown_query_scores_low(self):
        results = self.index.search("양자역학 논문 리뷰")
        self.assertTrue(not results or results[0][0] < 0.3)

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


class BrainTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.memory_path = os.path.join(self.tempdir.name, "memory.json")
        self.bot = Brain(memory_path=self.memory_path, rng=random.Random(1234))

    def test_greeting_is_confident(self):
        reply = self.bot.respond("안녕하세요")
        self.assertGreaterEqual(reply.score, CONFIDENT)
        self.assertTrue(reply.source.startswith("match:"))

    def test_calculator_skill(self):
        self.assertIn("= 84", self.bot.respond("12 * 7 은 뭐야").text)

    def test_korean_operator_words(self):
        self.assertIn("= 350", self.bot.respond("100 더하기 250").text)

    def test_name_is_stored_without_verb_ending(self):
        self.bot.respond("내 이름은 지훈이야")
        self.assertEqual(self.bot.recall("이름"), "지훈")
        self.assertIn("지훈", self.bot.respond("내 이름 뭐야").text)

    def test_unknown_question_falls_back(self):
        self.assertEqual(self.bot.respond("초끈이론 좀 요약해줘").source, "fallback")

    def test_learning_then_answering(self):
        self.bot.learn("좋아하는 색이 뭐야", "파란색")
        reply = self.bot.respond("좋아하는 색이 뭐야")
        self.assertEqual(reply.text, "파란색")

    def test_relearning_replaces_answer(self):
        self.assertTrue(self.bot.learn("취미가 뭐야", "코딩"))
        self.assertFalse(self.bot.learn("취미가 뭐야", "등산"))
        self.assertEqual(len(self.bot.memory["learned"]), 1)
        self.assertEqual(self.bot.respond("취미가 뭐야").text, "등산")

    def test_learn_requires_both_sides(self):
        with self.assertRaises(ValueError):
            self.bot.learn("   ", "답")

    def test_forget_specific_and_all(self):
        self.bot.learn("질문", "답")
        self.bot.remember("생일", "1월 1일")
        self.assertEqual(self.bot.forget("생일"), 1)
        self.assertEqual(self.bot.forget(), 1)
        self.assertEqual(self.bot.memory["learned"], [])

    def test_memory_survives_restart(self):
        self.bot.learn("비밀번호 힌트가 뭐야", "네 강아지 이름")
        self.bot.remember("이름", "지훈")
        self.bot.save()

        reborn = Brain(memory_path=self.memory_path, rng=random.Random(1))
        self.assertEqual(reborn.recall("이름"), "지훈")
        self.assertEqual(reborn.respond("비밀번호 힌트가 뭐야").text, "네 강아지 이름")

    def test_corrupt_memory_file_does_not_crash(self):
        with open(self.memory_path, "w", encoding="utf-8") as handle:
            handle.write("{ this is not json")
        bot = Brain(memory_path=self.memory_path)
        self.assertEqual(bot.memory["learned"], [])

    def test_empty_input(self):
        self.assertEqual(self.bot.respond("   ").source, "empty")


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.bot = Brain(
            memory_path=os.path.join(self.tempdir.name, "memory.json"),
            rng=random.Random(7),
        )

    def test_plain_text_is_not_a_command(self):
        self.assertIsNone(handle_command(self.bot, "안녕"))

    def test_unknown_command(self):
        self.assertIn("그런 명령은 없어", handle_command(self.bot, "/없는명령"))

    def test_learn_command(self):
        handle_command(self.bot, "/배워 수도가 어디야 | 서울")
        self.assertEqual(self.bot.respond("수도가 어디야").text, "서울")

    def test_learn_command_requires_pipe(self):
        self.assertIn("형식이 달라", handle_command(self.bot, "/배워 파이프없음"))

    def test_interactive_learning_flow(self):
        self.bot.respond("좋아하는 영화 있어")
        self.assertIn("뭐라고 답하면", handle_command(self.bot, "/배워"))
        self.bot.respond("인터스텔라!")
        self.assertEqual(self.bot.respond("좋아하는 영화 있어").text, "인터스텔라!")

    def test_memory_command(self):
        self.assertIn("아직 외운 게", handle_command(self.bot, "/기억"))
        self.bot.remember("이름", "지훈")
        self.assertIn("지훈", handle_command(self.bot, "/기억"))

    def test_save_command_writes_file(self):
        handle_command(self.bot, "/저장")
        self.assertTrue(os.path.exists(self.bot.memory_path))


if __name__ == "__main__":
    unittest.main()

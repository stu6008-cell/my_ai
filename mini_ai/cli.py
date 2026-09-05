"""터미널 대화 루프와 슬래시 명령어."""

import argparse
import sys

from .brain import Brain

BANNER = """\
┌──────────────────────────────────────────────┐
│  미니 AI - 순수 파이썬 대화 프로그램          │
│  외부 API·인터넷·설치 패키지 전부 없음        │
│  '/도움말' 로 명령어 보기, '/종료' 로 나가기  │
└──────────────────────────────────────────────┘"""

HELP = """\
사용할 수 있는 명령어
  /도움말              이 목록 보기
  /배워 질문 | 답변    새로운 문답 가르치기
  /배워                방금 내가 한 말에 대한 답을 가르치기
  /기억                지금까지 외운 것 모두 보기
  /잊어 [키워드]       특정 기억 삭제 (키워드 없으면 전부)
  /저장                기억을 파일에 저장
  /상상 [단어]         배운 문장들로 새 문장 지어내기
  /디버그 문장         그 문장의 유사도 후보와 점수 보기
  /종료                대화 끝내기"""


def _cmd_learn(bot, argument):
    if not argument:
        if not bot.last_user_message:
            return "먼저 뭔가 말을 걸어줘. 그 말에 대한 답을 가르칠 수 있어."
        bot.awaiting_answer_for = bot.last_user_message
        return "'{}' 에는 뭐라고 답하면 될까? 다음 줄에 적어줘.".format(bot.last_user_message)

    if "|" not in argument:
        return "형식이 달라. '/배워 질문 | 답변' 처럼 세로줄로 나눠줘."

    question, answer = argument.split("|", 1)
    try:
        created = bot.learn(question, answer)
    except ValueError as error:
        return str(error)
    return "새로 배웠어!" if created else "기존 답변을 새 걸로 바꿨어."


def _cmd_memory(bot, argument):
    facts, learned = bot.memory["facts"], bot.memory["learned"]
    if not facts and not learned:
        return "아직 외운 게 하나도 없어."

    lines = []
    if facts:
        lines.append("[사실]")
        lines.extend("  {} = {}".format(key, value) for key, value in facts.items())
    if learned:
        lines.append("[배운 문답] {}개".format(len(learned)))
        lines.extend("  {} -> {}".format(item["question"], item["answer"]) for item in learned)
    return "\n".join(lines)


def _cmd_forget(bot, argument):
    removed = bot.forget(argument or None)
    if not removed:
        return "'{}' 에 해당하는 기억이 없어.".format(argument)
    return "{}개 지웠어.".format(removed)


def _cmd_save(bot, argument):
    bot.save()
    return "{} 에 저장했어.".format(bot.memory_path)


def _cmd_debug(bot, argument):
    if not argument:
        return "'/디버그 안녕' 처럼 확인할 문장을 같이 적어줘."
    return bot.explain(argument)


COMMANDS = {
    "도움말": lambda bot, arg: HELP,
    "help": lambda bot, arg: HELP,
    "배워": _cmd_learn,
    "learn": _cmd_learn,
    "기억": _cmd_memory,
    "memory": _cmd_memory,
    "잊어": _cmd_forget,
    "forget": _cmd_forget,
    "저장": _cmd_save,
    "save": _cmd_save,
    "상상": lambda bot, arg: bot.imagine(arg or None),
    "dream": lambda bot, arg: bot.imagine(arg or None),
    "디버그": _cmd_debug,
    "debug": _cmd_debug,
}

EXIT_WORDS = {"/종료", "/exit", "/quit", "/q"}


def handle_command(bot, line):
    """'/'로 시작하는 입력을 처리한다. 명령이 아니면 None."""
    if not line.startswith("/"):
        return None
    name, _, argument = line[1:].partition(" ")
    handler = COMMANDS.get(name.strip().lower())
    if handler is None:
        return "그런 명령은 없어. '/도움말' 을 봐."
    return handler(bot, argument.strip())


def run(bot=None, show_score=False):
    bot = bot or Brain()
    print(BANNER)
    print("(기본 지식 {}개 · 배운 문답 {}개)\n".format(len(bot.intents), len(bot.memory["learned"])))

    while True:
        try:
            line = input("너 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            line = "/종료"

        if line in EXIT_WORDS:
            bot.save()
            print("미니 > 기억 저장 완료. 잘 가!")
            return 0

        if not line:
            continue

        answer = handle_command(bot, line)
        if answer is None:
            reply = bot.respond(line)
            answer = reply.text
            if show_score:
                answer += "   [{} {:.2f}]".format(reply.source, reply.score)

        print("미니 > {}\n".format(answer))


def main(argv=None):
    parser = argparse.ArgumentParser(description="순수 파이썬으로 만든 미니 대화 AI")
    parser.add_argument("-m", "--memory", help="기억 파일 경로")
    parser.add_argument("-k", "--knowledge", help="지식 파일 경로")
    parser.add_argument("-s", "--score", action="store_true", help="응답 근거와 점수를 함께 표시")
    parser.add_argument("--say", help="대화 루프 없이 한 마디만 주고받기")
    args = parser.parse_args(argv)

    kwargs = {}
    if args.memory:
        kwargs["memory_path"] = args.memory
    if args.knowledge:
        kwargs["knowledge_path"] = args.knowledge
    bot = Brain(**kwargs)

    if args.say:
        reply = bot.respond(args.say)
        print(reply.text)
        return 0

    return run(bot, show_score=args.score)


if __name__ == "__main__":
    sys.exit(main())

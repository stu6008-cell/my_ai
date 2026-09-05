"""터미널 대화 루프와 슬래시 명령어."""

import argparse
import os
import sys

from .brain import Brain

BANNER = """\
┌──────────────────────────────────────────────┐
│  미니 AI - 순수 파이썬 대화 프로그램          │
│  외부 API·인터넷·설치 패키지 전부 없음        │
│  내가 파일에 넣은 정보만 학습한다             │
│  '/도움말' 로 명령어 보기, '/종료' 로 나가기  │
└──────────────────────────────────────────────┘"""

HELP = """\
사용할 수 있는 명령어
  /도움말              이 목록 보기
  /새로고침            원본 파일을 다시 읽어서 학습 (파일 고친 뒤에 실행)
  /데이터              지금 학습한 내용 전부 보기
  /원본                원본 파일 경로와 작성법 보기
  /학습 파일경로       다른 파일을 원본으로 삼아 학습
  /학습현황            무엇을 얼마나 배웠는지 통계
  /상상 [단어]         배운 문장들로 새 문장 지어내기
  /디버그 문장         그 문장의 유사도 후보와 점수 보기
  /종료                대화 끝내기

대화 학습이 켜져 있을 때만 쓸 수 있는 명령 (--chat-learning)
  /배워 질문 | 답변    대화로 새 문답 가르치기
  /좋아  /별로         직전 답변 평가"""

SOURCE_GUIDE = """\
원본 파일에 이렇게 적으면 돼. 저장한 뒤 '/새로고침'.

  # 이렇게 시작하는 줄은 주석
  좋아하는 음식: 마라탕          <- 키: 값  (사실)
  회사 출근 시간: 오전 9시

  Q: 우리 강아지 이름이 뭐야     <- 명시적 문답
  A: 콩이야

  주말에 뭐 할까 | 한강 가자     <- 한 줄 문답

  파이썬은 배우기 쉬운 언어이다   <- X는 Y이다 (사실 + 자동 문답)

  오늘은 비가 왔다               <- 그 외 문장은 말투 학습용

이미 정리된 JSON 을 그대로 넣어도 된다 ({"facts": {...}, "pairs": [...]}).
원본 파일이 유일한 진실이라, 파일에서 지운 내용은 다음 새로고침에 사라진다."""

_CHAT_LEARNING_OFF = (
    "대화로는 안 배우도록 설정돼 있어. 원본 파일에 적고 '/새로고침' 해줘.\n  {}"
)


def _cmd_reload(bot, argument):
    added, error = bot.reload_source()
    if error is not None:
        return "원본을 못 읽었어: {}".format(error)
    if added is None:
        return "원본 파일이 없어: {}\n'/원본' 으로 작성법을 볼 수 있어.".format(bot.source_path)
    return "다시 학습했어. 사실 {}개 · 문답 {}개 · 문장 {}개".format(
        added["facts"], added["pairs"], added["sentences"]
    )


def _cmd_source(bot, argument):
    exists = os.path.exists(bot.source_path) if bot.source_path else False
    header = "원본 파일: {} ({})".format(bot.source_path, "있음" if exists else "없음")
    return header + "\n\n" + SOURCE_GUIDE


def _cmd_train(bot, argument):
    if not argument:
        return "'/학습 내정보.txt' 처럼 파일 경로를 같이 적어줘."
    if not os.path.exists(argument):
        return "그런 파일이 없어: {}".format(argument)
    bot.source_path = argument
    return _cmd_reload(bot, None)


def _cmd_learn(bot, argument):
    if not bot.learn_from_chat:
        return _CHAT_LEARNING_OFF.format(bot.source_path)
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
    except (ValueError, PermissionError) as error:
        return str(error)
    return "새로 배웠어!" if created else "기존 답변을 새 걸로 바꿨어."


def _cmd_feedback(bot, argument, positive):
    if not bot.learn_from_chat:
        return _CHAT_LEARNING_OFF.format(bot.source_path)
    result = bot.feedback(positive)
    if result is None:
        return "아직 평가할 답변이 없어. 먼저 말을 걸어줘."
    direction, label, weight, learned = result
    if direction == "up":
        message = "고마워! '{}' 쪽 답변을 더 자주 쓸게. (가중치 {:.2f})".format(label, weight)
        if learned:
            message += "\n네 표현 '{}' 도 같은 뜻으로 외웠어.".format(bot.last_user_message)
        return message
    return "알겠어, 그 답변은 앞으로 덜 쓸게. (가중치 {:.2f})".format(weight)


def _cmd_debug(bot, argument):
    if not argument:
        return "'/디버그 안녕' 처럼 확인할 문장을 같이 적어줘."
    return bot.explain(argument)


COMMANDS = {
    "도움말": lambda bot, arg: HELP,
    "help": lambda bot, arg: HELP,
    "새로고침": _cmd_reload,
    "reload": _cmd_reload,
    "데이터": lambda bot, arg: bot.dataset_summary(),
    "data": lambda bot, arg: bot.dataset_summary(),
    "원본": _cmd_source,
    "source": _cmd_source,
    "학습": _cmd_train,
    "train": _cmd_train,
    "학습현황": lambda bot, arg: bot.progress(),
    "stats": lambda bot, arg: bot.progress(),
    "상상": lambda bot, arg: bot.imagine(arg or None),
    "dream": lambda bot, arg: bot.imagine(arg or None),
    "디버그": _cmd_debug,
    "debug": _cmd_debug,
    "배워": _cmd_learn,
    "learn": _cmd_learn,
    "좋아": lambda bot, arg: _cmd_feedback(bot, arg, True),
    "good": lambda bot, arg: _cmd_feedback(bot, arg, True),
    "별로": lambda bot, arg: _cmd_feedback(bot, arg, False),
    "bad": lambda bot, arg: _cmd_feedback(bot, arg, False),
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
    print("(원본 {} · 사실 {}개 · 문답 {}개)\n".format(
        os.path.basename(bot.source_path) if bot.source_path else "없음",
        len(bot.dataset["facts"]),
        len(bot.dataset["pairs"]),
    ))
    if not bot.dataset["pairs"]:
        print("아직 아는 게 없어. '/원본' 을 쳐서 파일 작성법을 확인해줘.\n")

    while True:
        try:
            line = input("너 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            line = "/종료"

        if line in EXIT_WORDS:
            bot.save()
            print("미니 > 잘 가!")
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
    parser = argparse.ArgumentParser(description="내가 준 파일로만 배우는 미니 대화 AI")
    parser.add_argument("-f", "--source", help="학습할 원본 파일 (기본: data/source.txt)")
    parser.add_argument("-d", "--dataset", help="정리된 데이터셋 JSON 경로")
    parser.add_argument("-m", "--memory", help="기억 파일 경로")
    parser.add_argument("-s", "--score", action="store_true", help="응답 근거와 점수를 함께 표시")
    parser.add_argument("--say", help="대화 루프 없이 한 마디만 주고받기")
    parser.add_argument(
        "--with-defaults", action="store_true", help="내장 잡담 지식도 함께 사용"
    )
    parser.add_argument(
        "--chat-learning", action="store_true", help="대화로도 배우게 허용 (기본은 금지)"
    )
    args = parser.parse_args(argv)

    kwargs = {"strict": not args.with_defaults, "learn_from_chat": args.chat_learning}
    for name, value in (
        ("source_path", args.source),
        ("dataset_path", args.dataset),
        ("memory_path", args.memory),
    ):
        if value:
            kwargs[name] = value
    bot = Brain(**kwargs)

    if args.say:
        print(bot.respond(args.say).text)
        return 0

    return run(bot, show_score=args.score)


if __name__ == "__main__":
    sys.exit(main())

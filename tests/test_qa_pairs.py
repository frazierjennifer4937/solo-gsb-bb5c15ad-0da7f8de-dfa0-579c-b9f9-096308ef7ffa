"""访谈问答索引核心逻辑单元测试 + 暴力对拍（纯标准库）。

配对部分用暴力 DFS 枚举全部保序一对一配对，按
(配对数 desc, 回答字符总数 desc, (问题首词, 回答首词) 序列) 取最优，
与 build_qa_pairs 的重建结果对照；另用小例子手工校验轮次合并、问号识别、
说话人与间隔约束、未配对问题回报与输入校验复用。
"""

import random
import unittest
from types import SimpleNamespace

from app.errors import DomainError
from app.qa_pairs import build_qa_pairs


def W(start, end, text, speaker="A"):
    return SimpleNamespace(start_ms=start, end_ms=end, text=text, speaker=speaker)


def split_turns(words):
    """与核心逻辑一致的轮次合并（测试侧独立实现，便于对照）。"""
    turns = []
    run_start = 0
    for i in range(1, len(words) + 1):
        if i == len(words) or words[i].speaker != words[run_start].speaker:
            text = "".join(w.text for w in words[run_start:i])
            turns.append(
                {
                    "start_word": run_start,
                    "end_word": i - 1,
                    "start_ms": words[run_start].start_ms,
                    "end_ms": words[i - 1].end_ms,
                    "speaker": words[run_start].speaker,
                    "text": text,
                    "is_question": text.rstrip().endswith(("？", "?")),
                }
            )
            run_start = i
    return turns


def brute_force_best(words, max_gap_ms):
    """枚举全部保序一对一配对，返回最优 (问题首词, 回答首词) 元组序列。"""
    turns = split_turns(words)
    questions = [t for t in turns if t["is_question"]]
    answers = [t for t in turns if not t["is_question"]]
    compatible = [
        [
            a["speaker"] != q["speaker"]
            and 0 <= a["start_ms"] - q["end_ms"] <= max_gap_ms
            for a in answers
        ]
        for q in questions
    ]
    best = None  # ((-配对数, -回答字符总数), 配对元组)

    def dfs(i, j, chosen, chars):
        nonlocal best
        if i == len(questions) or j == len(answers):
            candidate = ((-len(chosen), -chars), tuple(chosen))
            if best is None or candidate < best:
                best = candidate
            return
        dfs(i + 1, j, chosen, chars)
        dfs(i, j + 1, chosen, chars)
        if compatible[i][j]:
            dfs(i + 1, j + 1, chosen + [(i, j)], chars + len(answers[j]["text"]))

    dfs(0, 0, [], 0)
    return tuple(
        (questions[i]["start_word"], answers[j]["start_word"]) for i, j in best[1]
    )


def pair_word_sequence(pairs):
    return tuple((p["question"]["start_word"], p["answer"]["start_word"]) for p in pairs)


class TurnAndQuestionTests(unittest.TestCase):
    def test_adjacent_same_speaker_merges_into_one_turn(self):
        words = [
            W(0, 500, "你今天", "A"),
            W(500, 1000, "怎么样？", "A"),  # 与上词同说话人，合并为一个提问轮次
            W(1200, 1800, "挺好的。", "B"),
        ]
        pairs, unpaired = build_qa_pairs(words, 300)
        self.assertEqual(unpaired, [])
        self.assertEqual(len(pairs), 1)
        question = pairs[0]["question"]
        self.assertEqual((question["start_word"], question["end_word"]), (0, 1))
        self.assertEqual((question["start_ms"], question["end_ms"]), (0, 1000))
        self.assertEqual(question["text"], "你今天怎么样？")
        self.assertEqual(question["speaker"], "A")

    def test_question_mark_must_be_at_turn_end(self):
        # 轮次中间出现问号不算提问：整轮文本不以问号结尾
        words = [
            W(0, 500, "什么？", "A"),
            W(500, 1000, "不是的。", "A"),  # 合并后文本“什么？不是的。”
            W(1000, 1500, "真的吗？", "B"),
        ]
        pairs, unpaired = build_qa_pairs(words, 10_000)
        # 唯一提问是 B 的“真的吗？”，其后没有轮次可回答
        self.assertEqual(pairs, [])
        self.assertEqual(unpaired, [2])

    def test_trailing_whitespace_stripped_before_mark_check(self):
        words = [
            W(0, 500, "好吗？ ", "A"),   # 尾部空白后仍以问号结尾
            W(500, 1000, "好的。", "B"),
            W(1000, 1500, "走吗?", "A"),  # 半角问号
            W(1500, 2000, "走。", "B"),
        ]
        pairs, unpaired = build_qa_pairs(words, 10_000)
        self.assertEqual(unpaired, [])
        self.assertEqual(pair_word_sequence(pairs), ((0, 1), (2, 3)))

    def test_no_question_returns_empty_pairs(self):
        words = [W(0, 500, "今天很好。", "A"), W(500, 1000, "是的。", "B")]
        pairs, unpaired = build_qa_pairs(words, 10_000)
        self.assertEqual(pairs, [])
        self.assertEqual(unpaired, [])


class CandidateConstraintTests(unittest.TestCase):
    def test_answer_must_come_from_other_speaker(self):
        words = [
            W(0, 1000, "听懂了吗？", "A"),
            W(1000, 1500, "嗯。", "B"),
            W(1500, 2000, "我懂了。", "A"),  # 与提问同说话人，不可作回答
        ]
        pairs, unpaired = build_qa_pairs(words, 10_000)
        self.assertEqual(pair_word_sequence(pairs), ((0, 1),))
        self.assertEqual(unpaired, [])

    def test_question_turn_cannot_be_answer(self):
        words = [
            W(0, 1000, "第一个问题？", "A"),
            W(1000, 2000, "第二个问题？", "B"),  # 提问轮次，不可作回答
            W(2000, 3000, "回答。", "A"),
        ]
        pairs, unpaired = build_qa_pairs(words, 10_000)
        # 问一（A）的唯一后续非提问轮次“回答。”同为 A，不可配对；
        # 问二（B）自身是提问，但仍可被“回答。”回答
        self.assertEqual(pair_word_sequence(pairs), ((1, 2),))
        self.assertEqual(unpaired, [0])

    def test_answer_must_start_after_question_ends_within_gap(self):
        words = [
            W(0, 1000, "怎么样？", "A"),
            W(1500, 2000, "挺好的。", "B"),  # 间隔 500ms
        ]
        pairs, unpaired = build_qa_pairs(words, 500)  # 间隔恰好等于上限
        self.assertEqual(pair_word_sequence(pairs), ((0, 1),))
        self.assertEqual(unpaired, [])

        pairs, unpaired = build_qa_pairs(words, 499)  # 间隔超限，无候选
        self.assertEqual(pairs, [])
        self.assertEqual(unpaired, [0])

    def test_zero_gap_requires_touching_turns(self):
        words = [
            W(0, 1000, "走吗？", "A"),
            W(1000, 1500, "走。", "B"),   # 端点相接，间隔 0
            W(2000, 2500, "急吗？", "A"),
            W(2600, 3000, "不急。", "B"),  # 间隔 100，max_gap_ms=0 下不可用
        ]
        pairs, unpaired = build_qa_pairs(words, 0)
        self.assertEqual(pair_word_sequence(pairs), ((0, 1),))
        self.assertEqual(unpaired, [2])

    def test_unpaired_questions_reported_in_order(self):
        words = [
            W(0, 500, "问一？", "A"),
            W(500, 1000, "答一。", "B"),
            W(1000, 1500, "问二？", "A"),
            W(1500, 2000, "问三？", "B"),  # 提问不能回答提问
        ]
        pairs, unpaired = build_qa_pairs(words, 10_000)
        self.assertEqual(pair_word_sequence(pairs), ((0, 1),))
        self.assertEqual(unpaired, [2, 3])


class OptimizationTests(unittest.TestCase):
    def test_pair_count_maximized_first(self):
        # 若只允许一对，(问一, 答长) 字符更多；但保序两对 (问一,答一)(问二,答二) 优先
        words = [
            W(0, 100, "问一？", "A"),
            W(100, 200, "答一。", "B"),
            W(200, 300, "问二？", "A"),
            W(300, 400, "答二很长很长。", "B"),
        ]
        pairs, unpaired = build_qa_pairs(words, 10_000)
        self.assertEqual(pair_word_sequence(pairs), ((0, 1), (2, 3)))
        self.assertEqual(unpaired, [])

    def test_answer_chars_maximized_over_lex(self):
        # 同一提问的两个候选回答：字典序较小者字符更少，应取字符更多者
        words = [
            W(0, 100, "问？", "A"),
            W(100, 200, "短。", "B"),
            W(200, 300, "长长的回答。", "C"),
        ]
        pairs, _ = build_qa_pairs(words, 200)
        self.assertEqual(pair_word_sequence(pairs), ((0, 2),))
        self.assertEqual(pairs[0]["answer"]["text"], "长长的回答。")

    def test_lexicographic_tie_break(self):
        # 问二只能配答丙；问一可配答甲或答乙，字符数相同，取首词序号较小的答甲
        words = [
            W(0, 100, "问一？", "A"),
            W(100, 200, "答甲。", "B"),
            W(200, 300, "答乙。", "C"),
            W(300, 400, "问二？", "A"),
            W(400, 500, "答丙。", "B"),
        ]
        pairs, unpaired = build_qa_pairs(words, 400)
        self.assertEqual(pair_word_sequence(pairs), ((0, 1), (3, 4)))
        self.assertEqual(unpaired, [])

    def test_order_preserving_matching(self):
        # 交叉配对 (问一,答二)+(问二,答一) 不允许；保序下问二无处可配
        words = [
            W(0, 100, "问一？", "A"),
            W(100, 200, "问二？", "B"),
            W(200, 300, "答一。", "C"),
            W(300, 400, "答二。", "A"),
        ]
        pairs, unpaired = build_qa_pairs(words, 10_000)
        # 两问都兼容两答；保序最优为 (问一,答一)+(问二,答二)
        self.assertEqual(pair_word_sequence(pairs), ((0, 2), (1, 3)))
        self.assertEqual(unpaired, [])

    def test_validation_reused(self):
        with self.assertRaises(DomainError) as ctx:
            build_qa_pairs([W(1000, 2000, "甲？", "A"), W(900, 1500, "乙。", "B")], 0)
        self.assertEqual(ctx.exception.code, "time_out_of_order")
        with self.assertRaises(DomainError) as ctx:
            build_qa_pairs([W(0, 1000, "甲？", "A"), W(500, 1500, "乙。", "B")], 0)
        self.assertEqual(ctx.exception.code, "overlapping_time_range")


class DifferentialTests(unittest.TestCase):
    def test_matching_matches_brute_force(self):
        random.seed(20260925)
        texts = ["好吗？", "嗯。", "走吗?", "对", "啊？", "哈哈。", "x", "嗯？ "]
        speakers = ["A", "B", "C"]
        for trial in range(500):
            n = random.randint(1, 9)
            words = []
            t = 0
            for _ in range(n):
                t += random.choice([0, 100, 500])
                dur = random.choice([50, 200])
                words.append(
                    W(t, t + dur, random.choice(texts), random.choice(speakers))
                )
                t += dur
            max_gap_ms = random.choice([0, 100, 500, 10_000])
            pairs, unpaired = build_qa_pairs(words, max_gap_ms)

            expected = brute_force_best(words, max_gap_ms)
            self.assertEqual(
                pair_word_sequence(pairs),
                expected,
                f"trial {trial}: {[(w.start_ms, w.end_ms, w.text, w.speaker) for w in words]} gap={max_gap_ms}",
            )

            # 自洽校验：配对保序、回答满足说话人与间隔约束、未配对问题齐全
            turns = split_turns(words)
            questions = [t_ for t_ in turns if t_["is_question"]]
            used = set()
            prev_q = prev_a = -1
            for p in pairs:
                q, a = p["question"], p["answer"]
                self.assertNotEqual(q["speaker"], a["speaker"])
                self.assertGreaterEqual(a["start_ms"], q["end_ms"])
                self.assertLessEqual(a["start_ms"] - q["end_ms"], max_gap_ms)
                self.assertGreater(q["start_word"], prev_q)
                self.assertGreater(a["start_word"], prev_a)
                prev_q, prev_a = q["start_word"], a["start_word"]
                used.add(q["start_word"])
            self.assertEqual(
                unpaired,
                [t_["start_word"] for t_ in questions if t_["start_word"] not in used],
            )


if __name__ == "__main__":
    unittest.main()

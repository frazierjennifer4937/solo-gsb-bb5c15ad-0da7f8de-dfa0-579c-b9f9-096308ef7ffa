"""访谈问答索引核心逻辑（纯标准库，不依赖 FastAPI，便于单测）。

把相邻同一说话人的词合并为发言轮次；轮次文本（词文本原样拼接）去掉尾部
空白后以 "？" 或 "?" 结尾的视为**提问**，其余轮次可作**回答**。回答须
来自其他说话人、开始于问题结束后（回答首词 start_ms 不早于问题末词
end_ms），且该间隔不超过 max_gap_ms。对全部候选作**保序一对一配对**：

* 目标一：配对数最大；
* 目标二：同配对数时，回答文本字符总数（Unicode code point）最大；
* 目标三：同分时取（问题首词序号, 回答首词序号）序列字典序最小者。

算法（后向 DP + 贪心重建），O(Q·A) 时间/空间（Q、A 为提问/候选回答
轮次数，均不超过轮次总数 ≤ 300）：

* ``dp[i][j]`` 为提问后缀 [i, Q) 与回答后缀 [j, A) 的最优
  ``(配对数, 回答字符总数)``，按字典序比较（先配对数后字符数）；
  转移为跳过提问 i、跳过回答 j，或（候选边存在时）配对 (i, j)；
* 重建时在当前子矩形 [i0, Q)×[j0, A) 内按提问升序、同提问按回答升序，
  取第一个满足“配对 (i, j) + 后缀最优 == 当前最优”的候选边——轮次首词
  序号随轮次序号递增，故枚举顺序即取（问题首词, 回答首词）字典序最小的
  可行首对，归纳可得整条序列字典序最小；每次取对后子矩形严格缩小，
  累计扫描不超过 Q·A 格。
"""

from .segmentation import validate_words

QUESTION_MARKS = ("？", "?")  # 认可的提问结尾（全角/半角问号）


def build_qa_pairs(words, max_gap_ms):
    """为访谈转写生成问答索引，返回 (pairs, unpaired_question_start_words)。

    参数 words 中每个对象需具备属性 start_ms / end_ms / text / speaker
    （沿用同一套词序与时间校验：按时间排列、不重叠）；max_gap_ms 为回答
    起点与问题终点允许的最大间隔（毫秒，非负整数）。

    pairs 按时间升序，每项为 dict：index / question / answer；question 与
    answer 均为轮次 dict：start_word / end_word（0-based、含端点）/
    start_ms / end_ms / speaker / text。unpaired_question_start_words 为
    未配对提问的首词序号（0-based，升序）。无候选时 pairs 为空数组。
    """
    validate_words(words)
    n = len(words)

    # ---- 合并相邻同一说话人的词为发言轮次 ----
    turns = []
    run_start = 0
    for i in range(1, n + 1):
        if i == n or words[i].speaker != words[run_start].speaker:
            turns.append(
                {
                    "start_word": run_start,
                    "end_word": i - 1,  # 词序号均为 0-based、含端点
                    "start_ms": words[run_start].start_ms,
                    "end_ms": words[i - 1].end_ms,
                    "speaker": words[run_start].speaker,
                    "text": "".join(w.text for w in words[run_start:i]),
                }
            )
            run_start = i

    # ---- 划分提问轮次与候选回答轮次（去尾部空白后以问号结尾为提问）----
    questions = []
    answers = []
    for turn in turns:
        if turn["text"].rstrip().endswith(QUESTION_MARKS):
            questions.append(turn)
        else:
            answers.append(turn)
    q_count = len(questions)
    a_count = len(answers)
    answer_chars = [len(a["text"]) for a in answers]  # Unicode code points

    # ---- 候选边：回答来自其他说话人、开始于问题结束后且间隔不超 max_gap_ms ----
    compatible = [[False] * a_count for _ in range(q_count)]
    for qi, question in enumerate(questions):
        q_end = question["end_ms"]
        q_speaker = question["speaker"]
        for ai, answer in enumerate(answers):
            if answer["speaker"] == q_speaker:
                continue
            gap = answer["start_ms"] - q_end
            if 0 <= gap <= max_gap_ms:
                compatible[qi][ai] = True

    # ---- 后向 DP：dp[i][j] = 后缀 [i,Q)×[j,A) 的最优 (配对数, 回答字符总数) ----
    dp = [[(0, 0)] * (a_count + 1) for _ in range(q_count + 1)]
    for i in range(q_count - 1, -1, -1):
        row = dp[i]
        next_row = dp[i + 1]
        compat_row = compatible[i]
        for j in range(a_count - 1, -1, -1):
            best = next_row[j] if next_row[j] >= row[j + 1] else row[j + 1]
            if compat_row[j]:
                taken = (next_row[j + 1][0] + 1, next_row[j + 1][1] + answer_chars[j])
                if taken > best:
                    best = taken
            row[j] = best

    # ---- 贪心重建字典序最小的最优配对序列 ----
    matched = []  # (提问序号, 回答序号)
    i0 = j0 = 0
    target = dp[0][0]
    while target[0] > 0:
        found = None
        # 在当前子矩形 [i0,Q)×[j0,A) 内按提问升序、同提问按回答升序找第一条
        # “配对后其后缀还能补满 target”的候选边：枚举顺序即取（问题首词,
        # 回答首词）字典序最小的可行首对。
        for i in range(i0, q_count):
            next_row = dp[i + 1]
            compat_row = compatible[i]
            for j in range(j0, a_count):
                if not compat_row[j]:
                    continue
                suffix = next_row[j + 1]
                if (suffix[0] + 1, suffix[1] + answer_chars[j]) == target:
                    found = (i, j)
                    target = suffix
                    break
            if found is not None:
                break
        i, j = found  # target[0] > 0 时必有可行对
        matched.append((i, j))
        i0, j0 = i + 1, j + 1  # 后续扫描矩形严格缩小，全程累计访问 O(Q·A)

    pairs = []
    for index, (qi, ai) in enumerate(matched):
        pairs.append(
            {
                "index": index,
                "question": dict(questions[qi]),
                "answer": dict(answers[ai]),
            }
        )
    used = {qi for qi, _ in matched}
    unpaired = [
        question["start_word"]
        for qi, question in enumerate(questions)
        if qi not in used
    ]
    return pairs, unpaired

"""Generate synthetic reasoning-style SFT data.

The output is tl-corpus-v1 JSONL. User segments are context-only and assistant
segments are trainable by default.

This is intentionally deterministic for a fixed seed, so regenerated data is
stable unless the task templates change.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Callable

try:
    from .corpus import TrainingSegment, make_record
except ImportError:
    from corpus import TrainingSegment, make_record


CHINESE_NAMES = ["小明", "小红", "小刚", "小丽", "阿宁", "小周", "小林", "小夏"]
ITEMS = ["苹果", "梨", "石子", "书", "金币", "面包", "水晶", "零件"]
COLORS = ["红", "蓝", "绿", "黄", "紫", "黑", "白"]
ANIMALS = ["猫", "狗", "鸟", "马", "鱼", "兔"]
PLACES = ["仓库", "教室", "城堡", "花园", "码头", "图书馆"]


@dataclass(frozen=True)
class Example:
    task: str
    input: str
    reasoning: str
    answer: str

    @property
    def output(self) -> str:
        return f"思考：{self.reasoning}\n答案：{self.answer}"

    def to_json(self, index: int) -> dict:
        return make_record(
            f"reasoning-{index:06d}",
            "synthetic_reasoning",
            [
                TrainingSegment("user", self.input, False),
                TrainingSegment("assistant", self.output, True),
            ],
            {
                "task": self.task,
                "reasoning": self.reasoning,
                "answer": self.answer,
            },
        )


def make_arithmetic(rng: random.Random) -> Example:
    op = rng.choice(["+", "-", "*", "mixed"])
    if op == "+":
        a, b, c = rng.randint(10, 999), rng.randint(10, 999), rng.randint(1, 99)
        answer = a + b + c
        prompt = f"计算：{a} + {b} + {c} = ?"
        reasoning = f"先算 {a}+{b}={a+b}，再加 {c}，得到 {answer}。"
    elif op == "-":
        a, b, c = rng.randint(200, 1500), rng.randint(1, 199), rng.randint(1, 99)
        answer = a - b - c
        prompt = f"计算：{a} - {b} - {c} = ?"
        reasoning = f"先算 {a}-{b}={a-b}，再减 {c}，得到 {answer}。"
    elif op == "*":
        a, b, c = rng.randint(2, 30), rng.randint(2, 20), rng.randint(1, 99)
        answer = a * b + c
        prompt = f"计算：{a} * {b} + {c} = ?"
        reasoning = f"先乘法：{a}*{b}={a*b}，再加 {c}，得到 {answer}。"
    else:
        a, b, c = rng.randint(10, 99), rng.randint(2, 20), rng.randint(1, 50)
        answer = (a + b) * c
        prompt = f"计算：({a} + {b}) * {c} = ?"
        reasoning = f"括号优先：{a}+{b}={a+b}，再乘 {c}，得到 {answer}。"
    return Example("arithmetic", prompt, reasoning, str(answer))


def make_word_problem(rng: random.Random) -> Example:
    name = rng.choice(CHINESE_NAMES)
    item = rng.choice(ITEMS)
    start = rng.randint(3, 80)
    gain = rng.randint(2, 50)
    lose = rng.randint(1, min(start + gain - 1, 40))
    answer = start + gain - lose
    prompt = f"{name}原来有{start}个{item}，后来得到{gain}个，又用掉{lose}个。现在还有多少个{item}？"
    reasoning = f"先把得到的加上：{start}+{gain}={start+gain}；再减去用掉的 {lose}，所以剩下 {answer}。"
    return Example("word_problem", prompt, reasoning, f"{answer}个{item}")


def make_compare(rng: random.Random) -> Example:
    nums = [rng.randint(1, 999) for _ in range(rng.randint(4, 7))]
    target = rng.choice(["最大", "最小", "第二大"])
    if target == "最大":
        answer_num = max(nums)
        reasoning = f"比较这些数，最大的数是 {answer_num}。"
    elif target == "最小":
        answer_num = min(nums)
        reasoning = f"比较这些数，最小的数是 {answer_num}。"
    else:
        answer_num = sorted(nums, reverse=True)[1]
        reasoning = f"先从大到小排序为 {sorted(nums, reverse=True)}，第二大是 {answer_num}。"
    prompt = f"数字列表：{nums}。请找出{target}的数。"
    return Example("compare", prompt, reasoning, str(answer_num))


def make_sort(rng: random.Random) -> Example:
    nums = rng.sample(range(1, 200), rng.randint(5, 9))
    ascending = rng.choice([True, False])
    sorted_nums = sorted(nums, reverse=not ascending)
    direction = "从小到大" if ascending else "从大到小"
    prompt = f"把这些数字{direction}排序：{nums}"
    reasoning = f"按{direction}逐个排列，得到 {sorted_nums}。"
    answer = " ".join(str(n) for n in sorted_nums)
    return Example("sort", prompt, reasoning, answer)


def make_rule_transform(rng: random.Random) -> Example:
    nums = [rng.randint(1, 30) for _ in range(rng.randint(4, 7))]
    rule = rng.choice(["double_plus_one", "square_minus_one", "parity_label"])
    if rule == "double_plus_one":
        out = [2 * n + 1 for n in nums]
        prompt = f"按规则 f(x)=2x+1 转换列表：{nums}"
        reasoning = "对每个数先乘 2 再加 1，得到新列表。"
        answer = str(out)
    elif rule == "square_minus_one":
        out = [n * n - 1 for n in nums]
        prompt = f"按规则 f(x)=x*x-1 转换列表：{nums}"
        reasoning = "对每个数先平方再减 1，得到新列表。"
        answer = str(out)
    else:
        out = ["偶" if n % 2 == 0 else "奇" for n in nums]
        prompt = f"判断列表中每个数的奇偶：{nums}"
        reasoning = "能被 2 整除的是偶数，否则是奇数。"
        answer = " ".join(out)
    return Example("rule_transform", prompt, reasoning, answer)


def make_table_lookup(rng: random.Random) -> Example:
    people = rng.sample(CHINESE_NAMES, 4)
    scores = {name: rng.randint(50, 100) for name in people}
    query = rng.choice(people)
    prompt = f"成绩表：{scores}。{query}的成绩是多少？"
    reasoning = f"在成绩表中查找 {query}，对应的值是 {scores[query]}。"
    return Example("table_lookup", prompt, reasoning, str(scores[query]))


def make_relation_logic(rng: random.Random) -> Example:
    a, b, c = rng.sample(CHINESE_NAMES, 3)
    x = rng.randint(2, 30)
    y = rng.randint(1, 20)
    z = rng.randint(1, 20)
    value_b = x + y
    value_c = value_b - z
    item = rng.choice(ITEMS)
    prompt = f"{a}有{x}个{item}。{b}比{a}多{y}个。{c}比{b}少{z}个。{c}有多少个{item}？"
    reasoning = f"{b}有 {x}+{y}={value_b} 个；{c}比{b}少 {z} 个，所以 {value_b}-{z}={value_c}。"
    return Example("relation_logic", prompt, reasoning, f"{value_c}个{item}")


def make_set_logic(rng: random.Random) -> Example:
    universe = COLORS + ANIMALS
    a_set = set(rng.sample(universe, rng.randint(3, 5)))
    b_set = set(rng.sample(universe, rng.randint(3, 5)))
    op = rng.choice(["交集", "并集", "只在A中"])
    if op == "交集":
        result = sorted(a_set & b_set)
        reasoning = f"交集表示同时在 A 和 B 中的元素，所以结果是 {result}。"
    elif op == "并集":
        result = sorted(a_set | b_set)
        reasoning = f"并集表示属于 A 或 B 的所有元素，所以结果是 {result}。"
    else:
        result = sorted(a_set - b_set)
        reasoning = f"只在 A 中表示属于 A 但不属于 B 的元素，所以结果是 {result}。"
    prompt = f"A={sorted(a_set)}，B={sorted(b_set)}。求{op}。"
    answer = " ".join(result) if result else "空集"
    return Example("set_logic", prompt, reasoning, answer)


def make_path_reasoning(rng: random.Random) -> Example:
    start, mid, end = rng.sample(PLACES, 3)
    d1 = rng.randint(2, 30)
    d2 = rng.randint(2, 30)
    direct = rng.randint(2, 60)
    via = d1 + d2
    if via < direct:
        answer = f"经过{mid}"
        reasoning = f"经过{mid}的距离是 {d1}+{d2}={via}，直接走是 {direct}，{via} 更短。"
    else:
        answer = "直接走"
        reasoning = f"经过{mid}的距离是 {d1}+{d2}={via}，直接走是 {direct}，{direct} 不更长。"
    prompt = f"从{start}到{end}，直接距离{direct}。从{start}到{mid}是{d1}，从{mid}到{end}是{d2}。选择更短路线。"
    return Example("path_reasoning", prompt, reasoning, answer)


def make_string_reasoning(rng: random.Random) -> Example:
    letters = rng.sample(list("abcdefgxyz"), rng.randint(4, 7))
    op = rng.choice(["reverse", "count", "dedupe"])
    if op == "reverse":
        prompt = f"把序列反转：{' '.join(letters)}"
        answer = " ".join(reversed(letters))
        reasoning = "反转就是从最后一个元素开始依次写回到第一个元素。"
    elif op == "count":
        seq = [rng.choice(letters) for _ in range(rng.randint(7, 12))]
        target = rng.choice(letters)
        count = seq.count(target)
        prompt = f"序列：{' '.join(seq)}。字母 {target} 出现了几次？"
        answer = str(count)
        reasoning = f"逐个统计 {target}，它在序列中出现 {count} 次。"
    else:
        seq = [rng.choice(letters) for _ in range(rng.randint(7, 12))]
        seen = []
        for ch in seq:
            if ch not in seen:
                seen.append(ch)
        prompt = f"按首次出现顺序去重：{' '.join(seq)}"
        answer = " ".join(seen)
        reasoning = f"从左到右保留第一次出现的元素，得到 {' '.join(seen)}。"
    return Example("string_reasoning", prompt, reasoning, answer)


GENERATORS: list[Callable[[random.Random], Example]] = [
    make_arithmetic,
    make_word_problem,
    make_compare,
    make_sort,
    make_rule_transform,
    make_table_lookup,
    make_relation_logic,
    make_set_logic,
    make_path_reasoning,
    make_string_reasoning,
]


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for index, example in enumerate(examples, start=1):
            f.write(json.dumps(example.to_json(index), ensure_ascii=False) + "\n")


def generate_examples(count: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for i in range(count):
        generator = GENERATORS[i % len(GENERATORS)]
        examples.append(generator(rng))
    rng.shuffle(examples)
    return examples


def main() -> None:
    train_path = Path("data/corpus/reasoning_train.tl.jsonl")
    valid_path = Path("data/corpus/reasoning_valid.tl.jsonl")
    train_examples = generate_examples(50000, seed=0)
    valid_examples = generate_examples(2000, seed=1)
    write_jsonl(train_path, train_examples)
    write_jsonl(valid_path, valid_examples)
    print(f"wrote {len(train_examples)} examples to {train_path}")
    print(f"wrote {len(valid_examples)} examples to {valid_path}")
    print(json.dumps(train_examples[0].to_json(1), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

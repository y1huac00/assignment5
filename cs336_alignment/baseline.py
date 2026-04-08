import json
from pathlib import Path
from typing import Callable, Any

from vllm import LLM, SamplingParams

from drgrpo_grader import r1_zero_reward_fn

def load_jsonl(path: str) -> list[dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def format_r1_zero_prompt(example: dict[str, Any]) -> str:
    problem = example['question']

    prompt = f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \nUser: {problem} \nAssistant: <think>"

    return problem

def extract_gold_answer(example: dict[str, Any]) -> str:
    return example['answer']

def compute_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {}
    
    format_scores = [r['rewards'].get('format_reward', 0.0) for r in results]
    answer_scores = [r['rewards'].get('answer_reward', 0.0) for r in results]

    both_correct = [
        1.0 if (f == 1.0 and a ==1.0) else 0.0
        for f, a in zip(format_scores, answer_scores)
    ]

    metrics = {
        'num_examples': n,
        'format_rate': sum(format_scores) / n,
        'answer_rate': sum(answer_scores) / n,
        'both_rate': sum(both_correct) / n,
        "count_format1_answer1": sum(1 for f, a in zip(format_scores, answer_scores) if f == 1.0 and a == 1.0),
        "count_format1_answer0": sum(1 for f, a in zip(format_scores, answer_scores) if f == 1.0 and a == 0.0),
        "count_format0_answer0": sum(1 for f, a in zip(format_scores, answer_scores) if f == 0.0 and a == 0.0),
    }

    return metrics

def evaluate_vllm(vllm_model: LLM, reward_fn: Callable[[str, str], dict[str, float]], prompts: list[str], gold_answers: list[str],
    examples: list[dict[str, Any]], eval_sampling_params: SamplingParams,) -> list[dict[str, Any]]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for example, prompt, gold, output in zip(examples, prompts, gold_answers, outputs):
        # vLLM 通常从 outputs[i].outputs[0].text 取文本
        generation = output.outputs[0].text

        rewards = reward_fn(generation, gold)

        results.append({
            "example": example,
            "prompt": prompt,
            "gold_answer": gold,
            "generation": generation,
            "rewards": rewards,
        })

    return results


def main() -> None:
    data_path = "./data/gsm8k/test.jsonl"
    out_dir = Path("outputs/gsm8k_baseline_simpleprompt")
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_jsonl(data_path)

    # examples = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")['train']

    prompts = [format_r1_zero_prompt(ex) for ex in examples]
    gold_answers = [extract_gold_answer(ex) for ex in examples]

    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B",   # 如果作业要求本地路径，就改成本地路径
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


    results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        gold_answers=gold_answers,
        examples=examples,
        eval_sampling_params=sampling_params,
    )

    save_jsonl(str(out_dir / "results.jsonl"), results)

    metrics = compute_metrics(results)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
import torch
from typing import Callable, Any
from einops import rearrange
# from vllm.model_executor import set_random_seed as vllm_set_random_seed

# def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
#     """
#     start the inference process, here we use vLLM to hold a model on a GPU separate from the policy.
#     """
#     vllm_set_random_seed(seed)
#     world_size_patch = patch('torch.distributed.get_world_size', return_value=1)
#     profiling_patch = patch('vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling', return_value=None)

#     with world_size_patch, profiling_patch:
#         return LLM(
#             model=model_id,
#             device=device,
#             dtype=torch.bfloat16,
#             enable_prefix_caching=True,
#             gpu_memory_utilization=gpu_memory_utilizationm,
#         )

# def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
#     state_dict = policy.state_dict()
#     llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
#     llm_model.load_weights(state_dict.items())

def tokenize_prompt_and_output(prompts: list[str], outputs: list[str], tokenizer: Callable):
    assert len(prompts) == len(outputs)

    if tokenizer.pad_token_id is None:
        raise ValueError('tokenizer.pad_token_id is None.')
    
    prompts_tokenized = tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False,)['input_ids']
    outputs_tokenized = tokenizer(outputs, add_special_tokens=False, padding=False, truncation=False,)['input_ids']
    
    input_ids_list = []
    labels_list = []
    response_mask_list = []

    for prompt_ids, output_ids in zip(prompts_tokenized, outputs_tokenized):
        full_ids = prompt_ids + output_ids

        if len(full_ids) < 2:
            raise ValueError('length of prompt + output < 2')
        
        input_ids = full_ids[:-1]
        label_ids = full_ids[1:]

        prompt_len = len(prompt_ids)
        output_len = len(output_ids)

        response_mask = [0] * (prompt_len - 1) + [1] * output_len

        assert len(input_ids) == len(label_ids) == len(response_mask)

        input_ids_list.append(input_ids)
        labels_list.append(label_ids)
        response_mask_list.append(response_mask)

    max_len = max(len(x) for x in input_ids_list)
    batch_size = len(input_ids_list)
    pad_id = tokenizer.pad_token_id

    batch_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    batch_labels = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    batch_response_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (inp, lab, mask) in enumerate(zip(input_ids_list, labels_list, response_mask_list)):
        n = len(inp)
        batch_input_ids[i, :n] = torch.tensor(inp, dtype=torch.long)
        batch_labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        batch_response_mask[i, :n] = torch.tensor(mask, dtype=torch.long)
    
    return {
        'input_ids': batch_input_ids,
        'labels': batch_labels,
        'response_mask': batch_response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    entropy = - torch.sum(torch.exp(logp) * logp, dim=-1)

    return entropy

def get_response_log_probs(model, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False,) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    log_probs = torch.gather(logp, -1, rearrange(labels, 'b t -> b t 1'))

    log_probs = rearrange(log_probs, 'b t 1 -> b t')

    token_entropy = compute_entropy(logits) if return_token_entropy is True else None

    return {
        'log_probs': log_probs,
        'token_entropy': token_entropy
    }

def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float, dim: int | None = None,) -> torch.Tensor:
    mask = mask.to(tensor.dtype)
    masked_sum = (tensor * mask).sum(dim=dim)
    return masked_sum / normalize_constant

def sft_microbatch_train_step(policy_log_probs: torch.Tensor, response_mask: torch.Tensor, gradient_accumulation_steps: int, normalize_constant: float = 1.0,) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss = - policy_log_probs

    loss = masked_normalize(tensor=per_token_loss, mask=response_mask, normalize_constant=normalize_constant, dim=None)

    loss = loss / gradient_accumulation_steps

    loss.backward()

    metadata = {'sft_loss': loss.detach(),}

    return loss, metadata

@torch.no_grad()
def log_generations(
    model,
    tokenizer,
    prompt_strs: list[str],
    gold_answers: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> dict[str, Any]:
    """
    在训练过程中抽一批 prompt，让当前模型生成答案，再把生成文本、gold、reward、response 平均熵和长度统计整理成日志输出。
    """
    device = next(model.parameters()).device

    # 1. tokenize prompts for generation
    prompt_batch = tokenizer(prompt_strs, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False,)
    prompt_input_ids = prompt_batch['input_ids'].to(device)
    prompt_attention_mask = prompt_batch['attention_mask'].to(device)

    # 2. generate responses
    generated_ids = model.generate(
        input_ids=prompt_input_ids, 
        attention_mask=prompt_attention_mask, 
        max_new_tokens=max_new_tokens, 
        do_sample=True, 
        temperature=temperature, 
        top_p=top_p
    )

    # 3. decode only generated response part
    prompt_lens = prompt_attention_mask.sum(dim=-1).tolist()
    response_strs = []
    response_token_ids_list = []

    for i in range(len(prompt_strs)):
        full_ids = generated_ids[i]
        prompt_len = prompt_len[i]
        response_ids = full_ids[prompt_len:]
        response_token_ids_list.append(response_ids)
        response_str = tokenizer.decode(response_ids, skip_special_tokens=True)
        response_strs.append(response_str)

    # 4. reward
    reward_infos = [reward_fn(resp, gold) for resp, gold in zip(response_strs, gold_answers)]

    # 5. compute response token entropy
    tokenized = tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer)
    input_ids = tokenized('input_ids').to(device)
    labels = tokenized['labels'].to(device)
    response_mask = tokenized['response_mask'].to(device)

    score_out = get_response_log_probs(model=model, input_ids=input_ids, labels=labels, return_token_entropy=True)
    token_entropy = score_out['token_entropy']

    response_mask_f = response_mask.to(token_entropy.dtype)
    per_example_entropy_sum = (token_entropy * response_mask_f).sum(dim=1)
    per_example_response_len = response_mask_f.sum(dim=1).clamp_min(1.0)
    per_example_avg_entropy = per_example_entropy_sum / per_example_response_len

    # 6. length
    response_lengths = per_example_response_len.tolist()

    correct_flags = [
        float(info.get('answer_reward', 0.0)) == 1.0
        for info  in reward_infos
    ]

    def safe_mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if len(vals) > 0 else 0.0

    avg_response_length = safe_mean(response_lengths)
    correct_lengths = [l for l, c in zip(response_lengths, correct_flags) if c]
    incorrect_lengths = [l for l, c in zip(response_lengths, correct_flags) if not c]

    avg_response_length_correct = safe_mean(correct_lengths)
    avg_response_length_incorrect = safe_mean(incorrect_lengths)

    avg_token_entropy = safe_mean([x.item() for x in per_example_avg_entropy])

    # 7. per example records
    records = []
    for prompt, response, gold, reward_info, avg_ent, resp_len in zip(
        prompt_strs, response_strs, gold_answers, reward_infos, per_example_avg_entropy, response_lengths
    ):
        rec = {
            'prompt': prompt,
            'response': response,
            'gold_answer': gold,
            'avg_token_entropy': float(avg_ent.item()),
            'response_length': float(resp_len),
        }
        rec.update(reward_info)
        records.append(rec)
    
    return {
        'avg_response_length': avg_response_length,
        'avg_response_length_correct': avg_response_length_correct,
        'avg_response_length_incorrect': avg_response_length_incorrect,
        'avg_token_entropy': avg_token_entropy,
        'records': records,
    }


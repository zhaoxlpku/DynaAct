from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from typing import NamedTuple, Optional, Callable

import numpy as np
import torch
import re
import torch.nn as nn
from tqdm import trange
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import pickle

from math_utils import last_boxed_only_string, remove_boxed
from utils import get_state, get_model_device, build_sets_batch, extract_parentheses


def print_mcts_tree(node: MCTSNode, indent: str = "", last: bool = True) -> None:
    # Print current node
    branch = "└── " if last else "├── "
    print(
        f"{indent}{branch}Node {node.id} (Action: {node.action}, Q: {node.calc_q(node.cum_rewards) if node.cum_rewards else 0:.2f})")

    # Print state if exists
    if node.state:
        state_indent = indent + ("    " if last else "│   ")
        for sub in node.state:
            state_branch = "└─→ "
            print(f"{state_indent}{state_branch}[{sub.idx}] Q: {sub.sub_question} | A: {sub.sub_answer}")

    # Prepare indentation for children
    child_indent = indent + ("    " if last else "│   ")

    # Recursively print children
    if node.children:
        for i, child in enumerate(node.children):
            is_last = i == len(node.children) - 1
            print_mcts_tree(child, child_indent, is_last)


def get_letter_list(num_letters=26):
    return [chr(i) for i in range(65, 65 + min(num_letters, 26))]


def extract_parentheses_new(text):
    pattern = r'The correct answer is \((.*?)\)'
    matches = re.finditer(pattern, text)
    last_match = None
    for match in matches:
        last_match = match
    return last_match.group(1) if last_match else ''


def extract_answer(completion: str, task="math"):
    if task in ["math", "gsm8k"]:
        ans = remove_boxed(last_boxed_only_string(completion))
    elif task in ["mmlu", "mmlupro", "arc", "gpqa"]:
        ans = extract_parentheses_new(completion)
        if ans is None or ans.strip() not in get_letter_list():
            ans = None
    else:
        raise NotImplementedError
    return ans


class SubResult(NamedTuple):
    idx: int
    sub_question: str
    sub_answer: str


class MCTSNode:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
            self,
            state: list[SubResult],
            action: str,
            parent: MCTSNode = None,
            task: str = None,
            reward: float = 0.,
            is_terminal: bool = False,
            calc_q: Callable[[list[float]], float] = np.mean
    ):
        assert parent is not None or task is not None, "Either 'parent' or 'task' must be provided, but both are None"

        self.id = next(MCTSNode.id_iter)

        self.cum_rewards: list[float] = []
        self.reward = reward
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent

        self.children: list[MCTSNode] = None
        self.calc_q = calc_q
        if parent is None:
            self.depth = 0
            self.task = task
        else:
            self.depth = parent.depth + 1
            self.task = parent.task

    @property
    def Q(self) -> float:
        if len(self.cum_rewards) == 0:
            return self.reward
        else:
            return self.calc_q(self.cum_rewards)


def get_terminal_action(task_type):
    if task_type == "math":
        return "Complete the solution and present the final answer within \\boxed{}."
    else:
        return "Complete the solution and format your response as follows: \"The correct answer is (insert answer here)\"."


def calc_reward(judges):
    corrects = 0
    incorrects = 0
    for judge in judges:
        if "the solution is correct." in judge.lower() or "the proposed solution is correct." in judge.lower():
            corrects += 1
        elif "the solution is incorrect." in judge.lower() or "the proposed solution is incorrect." in judge.lower():
            incorrects += 1
        else:
            continue
    return corrects / max(1, incorrects + corrects)


def save_root_nodes(root_list, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(root_list, f)


# To recover root nodes from a file
def load_root_nodes(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_submodular_model(base_model_path,
                          lora_path="qlearning/1223_qlearning_v1/checkpoints/step-9000",
                          eos_token_id=128009):
    from qvalue_encoder import QValueEncoder

    model = QValueEncoder(
        llama_model_path=base_model_path,
        lora_path=lora_path,
        tau=0.8,
        pooling_method="lasttoken",
        eos_token_id=eos_token_id,
        is_trainable=False
    )
    model = model.cuda()
    model = nn.DataParallel(model)
    return model


class MCTS:
    def __init__(self,
                 model_path="/ossfs/workspace/nas/xueliang/hf_models/Meta-Llama-3.1-8B-Instruct",
                 embedding_path="output/action_embeddings.pt",
                 submodular_base_path="/ossfs/workspace/nas/xueliang/hf_models/Llama-3.2-1B-Instruct",
                 submodular_lora_path="qlearning/1223_qlearning_v1/checkpoints/step-9000",   # path to the lora module created using sqil
                 n_gpus=8,
                 seed=42,
                 temperature=0.6,
                 top_p=1.0,
                 max_len=1024,
                 max_depth: int = 8,
                 min_depth: int = 3,
                 n_confidence: int = 4,
                 n_attempts: int = 8,
                 n_iters: int = 2,
                 w_exp: float = 1.0,
                 submodular_weight: float = 0.9,
                 submodular_size: int = 5,
                 task_type: str = "math",
                 cum_reward: str = "mean",
                 calc_q: str = "max",
                 simulate_strategy: str | Callable[[list[float]], int] = 'max',  # default value
                 disable_tqdm: bool = True):
        self.embedding_path = embedding_path
        self.w_exp = w_exp
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.n_iters = n_iters
        self.cum_reward = np.mean if cum_reward == 'mean' else max
        self.calc_q = np.mean if calc_q == 'mean' else max

        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy, simulate_strategy)
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = disable_tqdm

        self.temperature = temperature
        self.top_p = top_p
        self.max_len = max_len
        self.n_confidence = n_confidence
        self.n_attempts = n_attempts

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LLM(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode="slow",
            dtype="bfloat16",
            tensor_parallel_size=n_gpus,
            seed=seed,
            gpu_memory_utilization=0.8,
        )

        self.submodular_tokenizer = AutoTokenizer.from_pretrained(submodular_base_path)
        self.submodular_model = load_submodular_model(
            base_model_path=submodular_base_path,
            lora_path=submodular_lora_path,
            eos_token_id=self.submodular_tokenizer.eos_token_id
        )
        self.submodular_weight = submodular_weight
        self.submodular_size = submodular_size
        self.task_type = task_type

    def is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or node.depth >= self.max_depth

    def uct(self, node: MCTSNode) -> float:
        return node.Q + self.w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards)))

    def uct_select(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=self.uct)

    def get_submodular_actions(self, nodes: list[MCTSNode]):
        states = [get_state(node.task, [sub_result.sub_question for sub_result in node.state]) for node in nodes]

        action_embeddings = torch.load(self.embedding_path)
        actions = action_embeddings["actions"]
        embeddings = action_embeddings["embeddings"]
        embeddings = embeddings.to(get_model_device(self.submodular_model))

        selected_indices = build_sets_batch(
            states,
            encoder=self.submodular_model,
            tokenizer=self.submodular_tokenizer,
            action_embeddings=embeddings,
            ratio=self.submodular_weight,
            set_size=self.submodular_size,
        )
        selected_indices = selected_indices.tolist()
        assert len(selected_indices) == len(nodes)
        selected_actions = []
        for node, batch_indices in zip(nodes, selected_indices):
            if node.is_terminal:
                selected_actions.append([])
            elif node.depth < self.min_depth - 1:
                selected_actions.append([actions[idx] for idx in batch_indices])
            elif node.depth < self.max_depth - 1:
                selected_actions.append([actions[idx] for idx in batch_indices] + [get_terminal_action(self.task_type)])
            else:
                selected_actions.append([get_terminal_action(self.task_type)])
        return selected_actions

    def get_solution_prompt(self, task, state, action):
        if self.task_type == "math":
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Solve this problem step by step, following these rules:\n\n"
                        "1. Begin each step with '### Step'\n"
                        "2. When you see 'Complete the solution and present the final answer within \\boxed{}':\n"
                        "   - Continue solving ALL remaining steps without '### Step'\n" 
                        "   - You MUST enclose your final answer within \\boxed{}\n\n"
                        f"Problem:\n{task}"
                    ),
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Analyze this multiple-choice question step by step, following these rules:\n\n"
                        "1. Begin each step with '### Step'\n"
                        "2. When you see 'Complete the solution and format your response as follows: \"The correct answer is (insert answer here)\"':\n"
                        "   - Continue solving ALL remaining steps without '### Step'\n" 
                        "   - You MUST state your answer in the format: 'The correct answer is (X)'\n\n"
                        f"Problem:\n{task}"
                    ),
                }
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        for idx, sub_question, sub_answer in state:
            prompt += f"### Step {idx}: {sub_question.strip()}\n{sub_answer.strip()}\n\n"
        prompt += f"### Step {len(state) + 1}: {action}\n"

        # print("\n\n\nsolution prompt: ")
        # print(prompt)
        # input(">>>>>")
        return prompt

    def get_solutions(self, node_list: list[MCTSNode], actions_list: list[list[str]], use_tqdm=None):
        prompts = []
        terminal_list = []
        for node, actions in zip(node_list, actions_list):
            if node.is_terminal:
                assert len(actions) == 0
            for action in actions:
                prompts.append(self.get_solution_prompt(node.task, node.state, action))
                terminal_list.append(action == get_terminal_action())

        if len(prompts) > 0:
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_len,
                stop=["### Step", self.tokenizer.eos_token, "<|eot_id|>"]
            )
            if use_tqdm is None:
                use_tqdm = not self.disable_tqdm

            with torch.no_grad():
                sub_answer_list = self.model.generate(prompts, sampling_params, use_tqdm=use_tqdm)
                sub_answer_list = [sub_answer.outputs[0].text.strip() for sub_answer in sub_answer_list]

            for _ in range(1, self.n_attempts):
                remaining_indices = [idx for idx in range(len(sub_answer_list)) if terminal_list[idx] and
                                     (extract_answer(sub_answer_list[idx], task=self.task_type) is None or
                                      extract_answer(sub_answer_list[idx], task=self.task_type).strip() == "")]
                if len(remaining_indices) == 0:
                    break
                remaining_prompts = [prompts[idx] for idx in remaining_indices]
                with torch.no_grad():
                    remaining_answer_list = self.model.generate(remaining_prompts, sampling_params, use_tqdm=use_tqdm)
                    remaining_answer_list = [sub_answer.outputs[0].text.strip() for sub_answer in remaining_answer_list]
                for idx, sub_answer in zip(remaining_indices, remaining_answer_list):
                    sub_answer_list[idx] = sub_answer

        else:
            sub_answer_list = []

        solutions_list = []
        indent = 0
        for node, actions in zip(node_list, actions_list):
            solutions_list.append(sub_answer_list[indent: indent + len(actions)])
            indent += len(actions)
        return solutions_list

    def get_reward_prompt(self, task, state, action, solution):
        proposed_solution = ""
        for idx, sub_question, sub_answer in state:
            proposed_solution += f"### Step {idx}: {sub_question.strip()}\n{sub_answer.strip()}\n\n"
        proposed_solution += f"### Step {len(state) + 1}: {action.strip()}\n{solution.strip()}"

        messages = [
            {
                "role": "user",
                "content": (
                    "You are given a problem and a proposed solution.\n\n"
                    f"**Problem:**\n{task}\n\n" 
                    f"**Proposed Solution:**\n{proposed_solution}\n\n"
                    "Your task is to critique if the proposed solution is correct. A solution is correct if:\n"
                    "- It completely solves the problem correctly, OR\n"
                    "- It is a partial solution that could be extended into a complete correct solution\n\n"
                    "Consider a solution incorrect only if it:\n"
                    "- Contains errors OR\n"
                    "- Takes an approach that cannot lead to a correct solution\n\n"
                    "**Conclude your response with EXACTLY ONE of the following statements:**\n"
                    "- \"The solution is correct\" if the solution is complete OR could lead to a correct solution\n"
                    "- \"The solution is incorrect\" if the solution contains errors or cannot lead to a correct solution\n\n"
                    "**This is NOT optional. Your response MUST end with either \"The solution is correct\" or \"The solution is incorrect.\"**\n"
                    "**Limit your response to 200 words.**"
                )
            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += "\n\n### Critique\n"
        # print("\n\n\nreward prompt: ")
        # print(prompt)
        # input(">>>>>")
        return prompt

    def get_rewards(
            self,
            node_list: list[MCTSNode],
            actions_list: list[list[str]],
            solutions_list: list[list[str]],
            use_tqdm=None):
        prompts = []
        for node, actions, solutions in zip(node_list, actions_list, solutions_list):
            if node.is_terminal:
                assert len(actions) == 0 and len(solutions) == 0
            assert len(actions) == len(solutions)
            for action, solution in zip(actions, solutions):
                prompts.append(self.get_reward_prompt(node.task, node.state, action, solution))

        if len(prompts) > 0:
            sampling_params = SamplingParams(
                n=self.n_confidence,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_len,
                stop=[self.tokenizer.eos_token, "<|eot_id|>"]
            )

            if use_tqdm is None:
                use_tqdm = not self.disable_tqdm

            with torch.no_grad():
                judges_list = self.model.generate(prompts, sampling_params, use_tqdm=use_tqdm)
                judge_list = [judge.text.strip() for judges in judges_list for judge in judges.outputs]

        else:
            judge_list = []

        indent = 0
        rewards_list = []
        for node, actions, solutions in zip(node_list, actions_list, solutions_list):
            rewards = []
            assert len(actions) == len(solutions)
            for action, solution in zip(actions, solutions):
                reward = calc_reward(judge_list[indent: indent + self.n_confidence])
                # if len(node.state) > 0:
                #     print("last state: ", node.state[-1])
                # print("action: ", action)
                # print("---")
                # print("solution: ", solution)
                # print("---")
                # print("critique: ", judge_list[indent])
                # print("---")
                # print("reward: ", reward)
                rewards.append(reward)
                indent += self.n_confidence
            # input(">>>")
            rewards_list.append(rewards)
        return rewards_list

    def dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return [(cur.reward, len(cur.cum_rewards), cur)]
        if cur.children is None:
            return [(-math.inf, 0, None)]
        visited_children = [x for x in cur.children]
        results = []
        for child in visited_children:
            results += self.dfs_max_reward(path + [child])
        return results

    def select(self, node_list: list[MCTSNode]) -> list[list[MCTSNode]]:
        path_list = []
        for node in node_list:
            path = []
            while True:
                path.append(node)
                if node.children is None:  # if a node has children, it must not be terminal
                    path_list.append(path)
                    break
                node = self.uct_select(node)
        return path_list

    def expand(self, node_list: list[MCTSNode], use_tqdm=None):
        assert all(node.children is None for node in node_list)
        assert all(node.is_terminal or node.depth < self.max_depth for node in node_list)
        # if is_terminal -> skip;
        # if node.depth < max_depth -> expand

        sub_questions_list = self.get_submodular_actions(node_list)
        sub_answers_list = self.get_solutions(node_list, sub_questions_list, use_tqdm)

        rewards_list = self.get_rewards(node_list, sub_questions_list, sub_answers_list, use_tqdm)
        for node, sub_questions, sub_answers, rewards in zip(
                node_list, sub_questions_list, sub_answers_list, rewards_list):
            if node.is_terminal:
                continue

            children = []
            for sub_question, sub_answer, reward in zip(sub_questions, sub_answers, rewards):
                idx = len(node.state) + 1
                child = MCTSNode(
                    state=node.state + [SubResult(idx=idx, sub_question=sub_question, sub_answer=sub_answer)],
                    action=sub_question,
                    parent=node,
                    reward=reward,
                    calc_q=self.calc_q,
                    is_terminal=(sub_question == get_terminal_action(method="rstar"))
                )
                children.append(child)
            node.children = children

    def simulate(self, path_list: list[list[MCTSNode]], iter_num: int):
        node_list = [path[-1] for path in path_list]
        complete_list = [self.is_terminal_with_depth_limit(node) for node in node_list]

        max_steps = self.max_depth
        for _ in trange(max_steps, desc=f"Iteration {iter_num}", leave=True, position=iter_num + 1):
            for i in range(len(node_list)):
                if complete_list[i]:
                    continue
                node = node_list[i]
                rewards = [child.reward for child in node.children]
                node = node.children[self.simulate_choice(rewards)]
                path_list[i].append(node)
                node_list[i] = node
                complete_list[i] = self.is_terminal_with_depth_limit(node)
            if all(complete_list):
                break
            self.expand(node_list)

    def back_propagate(self, path_list: list[list[MCTSNode]]):
        cum_reward_list = []
        for path in path_list:
            rewards = []
            cum_reward = -math.inf
            for node in reversed(path):
                rewards.append(node.reward)
                cum_reward = self.cum_reward(rewards[::-1])
                node.cum_rewards.append(cum_reward)
            cum_reward_list.append(cum_reward)
        return cum_reward_list

    def search(self, tasks: list[str], pickle_path: str):
        MCTSNode.reset_id()
        root_list = [MCTSNode(state=[], action=None, parent=None, task=task, calc_q=self.calc_q) for task in tasks]
        # for root in root_list:
        #     print_mcts_tree(root)
        #     print("========== new tree ===========")
        # input(">>>")

        for iter_num in trange(self.n_iters, desc="MCTS iteration", leave=True, position=0):
            # self.iterate(root_list)
            # print("-----------init-------------")
            # for root in root_list:
            #     print_mcts_tree(root)
            #     print("========== new tree ===========")
            path_list = self.select(root_list)
            self.expand([path[-1] for path in path_list], use_tqdm=True)

            # print("------------expand----------------")
            # for root in root_list:
            #     print_mcts_tree(root)
            #     print("========== new tree ===========")
            self.simulate(path_list, iter_num)

            # print("-------------simulate-------------")
            # for root in root_list:
            #     print_mcts_tree(root)
            #     print("========== new tree ===========")
            self.back_propagate(path_list)
            # print("-------------back-------------")
            # for root in root_list:
            #     print_mcts_tree(root)
            #     print("========== new tree ===========")
            # input(">>>")

        save_root_nodes(root_list, pickle_path)

        reward_list, output_list, answers_list = [], [], []
        for root in root_list:
            results = self.dfs_max_reward([root])

            answer_dict = dict()
            for r, n, node in results:
                if node is None or n == 0:
                    continue
                completion = ""
                for sub_result in node.state:
                    completion += f"### Step {sub_result.idx}: {sub_result.sub_question}\n{sub_result.sub_answer}\n\n"
                completion = completion.strip()

                answer = extract_answer(completion, task=self.task_type)
                if answer in answer_dict:
                    answer_dict[answer].append((r, n, completion))
                else:
                    answer_dict[answer] = [(r, n, completion)]
            sorted_answers = sorted(answer_dict.items(), key=lambda x: sum(r for r, n, _ in x[1]), reverse=True)
            r, n, output = sorted_answers[0][1][0]
            reward_list.append(r)
            output_list.append(output)
            answers_list.append(answer_dict)
        return reward_list, output_list, answers_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate large language models on critical datasets.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to store cached outputs.")
    parser.add_argument("--pickle_path", type=str, required=True, help="Directory to store cached outputs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to the pre-calculated action embeddings.")
    parser.add_argument("--submodular_base_path", type=str, required=True, help="Path to the embedding (base) model.")
    parser.add_argument("--submodular_lora_path", type=str, required=True, help="Path to the embedding (lora) model.")
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs to use for tensor parallelism.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling for generation.")
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum number of tokens to generate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--min_depth", type=int, default=3)
    parser.add_argument("--n_confidence", type=int, default=8)
    parser.add_argument("--n_iters", type=int, default=8)
    parser.add_argument("--n_attempts", type=int, default=8)
    parser.add_argument("--aggregate_child", type=str, default="max")
    parser.add_argument("--aggregate_reward", type=str, default="mean")
    parser.add_argument("--submodular_weight", type=float, default=0.9)
    parser.add_argument("--submodular_size", type=int, default=5)

    args = parser.parse_args()

    task_type = "math"
    if "mmlu" in args.data_path or "gpqa" in args.data_path or "arc" in args.data_path:
        task_type = "mmlu"

    tasks = []
    items = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            prompt = item["prompt"]
            if task_type == "mmlu":
                assert "Format your response as follows: \"The correct answer is (insert answer here)\"." in prompt
                prompt = prompt.split("Format your response as follows: \"The correct answer is (insert answer here)\".")[0].strip()
            tasks.append(prompt)
            items.append(item)

    mcts = MCTS(
        model_path=args.model_path,
        embedding_path=args.embedding_path,
        submodular_base_path=args.submodular_base_path,
        submodular_lora_path=args.submodular_lora_path,
        n_gpus=args.n_gpus,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_len=args.max_len,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        n_confidence=args.n_confidence,
        n_iters=args.n_iters,
        n_attempts=args.n_attempts,
        w_exp=1.0,
        submodular_weight=args.submodular_weight,
        submodular_size=args.submodular_size,
        task_type=task_type,
        cum_reward=args.aggregate_child,
        calc_q=args.aggregate_reward,
        simulate_strategy='max',
        disable_tqdm=True,
    )

    reward_list, output_list, answers_list = mcts.search(tasks, args.pickle_path)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item, reward, completion, answers in zip(items, reward_list, output_list, answers_list):
            item["answers"] = answers
            item["completion"] = completion
            item["reward"] = reward
            f.write(json.dumps(item) + "\n")



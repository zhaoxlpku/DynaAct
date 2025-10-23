import math
import os
import json
import random
from datetime import datetime

import faiss
import fire
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from qvalue_encoder import QValueEncoder
from utils import (
    load_from_hdf5,
    load_qlearning_data_new,
    load_available_actions,
    get_state,
    tokenize_sentences,
    refresh_index,
    search_index,
)


def loss_ratio(global_step, max_ratio=1.0, min_ratio=0.1, decay_rate=0.001):
    """
    Computes a decreasing loss ratio based on the global step.

    Args:
        global_step (int): The current global training step.
        max_ratio (float): The initial maximum ratio.
        min_ratio (float): The minimum value of the ratio.
        decay_rate (float): The decay rate for exponential decay.

    Returns:
        float: The computed loss ratio.
    """
    # Exponential decay with a floor at min_ratio
    global_step_tensor = torch.tensor(global_step, dtype=torch.float32)

    # Exponential decay with a floor at min_ratio
    ratio = max_ratio * torch.exp(-decay_rate * global_step_tensor)
    return max(ratio.item(), min_ratio)  # .item() to return a scalar float


def main(
        exp_name,
        data_path="demo/platypus_sketches.jsonl",
        llama_model_path="/ossfs/workspace/nas/xueliang/hf_models/Llama-3.2-1B-Instruct",
        log_dir="/ossfs/workspace/nas/xueliang/code/o1_reasoning/qlearning",
        num_epochs=1,
        training_batch_size=8,
        action_batch_size=512,
        num_nearest_actions=16,
        sampling_size=1024,
        tau=1.0,
        pooling_method="mean",
        lr=1e-4,
        decay_rate=1e-3,
        log_interval=10,
        index_refresh_interval=1000,
        checkpoint_interval=1000,
        max_state_length=2048,
        max_action_length=128,
        adam_epsilon=1e-8,
        warmup_steps=1000,
):
    exp_dir = os.path.join(log_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # with open("demo/available_actions.json", "r") as json_file:
    #     action_data = json.load(json_file)
    # problems = load_from_hdf5(data_path, use_tqdm=True)

    problems = []
    with open(data_path, encoding="utf-8") as f:
        for line in f.readlines():
            problems.append(json.loads(line))
    action_data = load_available_actions(problems, use_tqdm=True)
    training_data = load_qlearning_data_new(problems, set(action_data), use_tqdm=True)

    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    model = QValueEncoder(
        llama_model_path,
        tau=tau,
        pooling_method=pooling_method,  # mean, lasttoken
        eos_token_id=tokenizer.eos_token_id,
        is_trainable=True
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, eps=adam_epsilon)

    total_batches = math.ceil(len(training_data) / training_batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_batches * num_epochs
    )

    model = model.cuda()
    model = nn.DataParallel(model)

    print(f"--- Building index ---")
    faiss_index = refresh_index(
        model,
        tokenizer,
        action_data,
        action_batch_size,
        max_action_length=max_action_length
    )

    global_step = 0

    for epoch in range(num_epochs):
        random.shuffle(training_data)

        for batch_idx in range(total_batches):
            start_idx = batch_idx * training_batch_size
            end_idx = min((batch_idx + 1) * training_batch_size, len(training_data))
            batch = training_data[start_idx:end_idx]

            states = [get_state(item["problem"], item["sketch"][:-1]) for item in batch]
            true_actions = [item["sketch"][-1] for item in batch]
            pred_actions = search_index(
                model,
                tokenizer,
                faiss_index,
                states,
                action_data,
                max_state_length=max_state_length,
                k=1,
                sampling=True,
                sampling_size=sampling_size,
            )
            # print("state: ")
            # print(states)
            # print("---")
            # print("actions:")
            # print(pred_actions)
            # input(">>>")

            states_batch, actions_batch, next_states_batch, rewards_batch = [], [], [], []
            for item, state, true_action, pred_action in zip(batch, states, true_actions, pred_actions):
                rewards_batch.append(1)
                states_batch.append(state)
                actions_batch.append(true_action)
                next_states_batch.append(get_state(item["problem"], item["sketch"][:-1] + [true_action]))

                rewards_batch.append(0)
                states_batch.append(state)
                actions_batch.append(pred_action)
                next_states_batch.append(get_state(item["problem"], item["sketch"][:-1] + [pred_action]))

            candidate_actions_batch = search_index(
                model,
                tokenizer,
                faiss_index,
                next_states_batch,
                action_data,
                max_state_length=max_state_length,
                k=num_nearest_actions,
                sampling=True,
                sampling_size=sampling_size,
            )

            tokenized_states_batch = tokenize_sentences(
                states_batch,
                tokenizer,
                max_length=max_state_length,
                padding="longest"
            )
            tokenized_actions_batch = tokenize_sentences(
                actions_batch,
                tokenizer,
                max_length=max_action_length,
                padding="longest"
            )
            tokenized_next_batch = tokenize_sentences(
                next_states_batch,
                tokenizer,
                max_length=max_state_length,
                padding="longest"
            )
            flattened_actions = [action for candidate in candidate_actions_batch for action in candidate]
            tokenized_flattened_actions = tokenize_sentences(
                flattened_actions,
                tokenizer,
                max_length=max_action_length,
                padding="longest"
            )
            candidate_idx = 0
            tokenized_candidate_batch = []
            for candidate in candidate_actions_batch:
                tokenized_candidate_batch.append(
                    tokenized_flattened_actions[candidate_idx:candidate_idx + len(candidate)])
                candidate_idx += len(candidate)

            states_ids = torch.tensor(tokenized_states_batch, dtype=torch.long).cuda()
            actions_ids = torch.tensor(tokenized_actions_batch, dtype=torch.long).cuda()
            next_states_ids = torch.tensor(tokenized_next_batch, dtype=torch.long).cuda()
            candidate_actions_ids = torch.tensor(tokenized_candidate_batch, dtype=torch.long).cuda()
            rewards = torch.tensor(rewards_batch, dtype=torch.float).cuda()

            model.train()
            ratio = loss_ratio(global_step, decay_rate=decay_rate)
            loss = model(
                states=states_ids,
                actions=actions_ids,
                next_states=next_states_ids,
                candidate_actions=candidate_actions_ids,
                rewards=rewards,
                ratio=ratio,
                mode="loss"
            )
            loss = loss.mean()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            if grad_norm >= 1e2:
                print("WARNING: Exploding Gradients {:.2f}".format(grad_norm))

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step % log_interval == 0 and global_step != 0:
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("Step: %d \t| loss: %.8f \t| ratio: %.4f \t| lr: %.8f \t| %s" % (
                    global_step, loss.item(), ratio, scheduler.get_lr()[0], time_str))

            if global_step % checkpoint_interval == 0 and global_step != 0:
                save_path = os.path.join(checkpoint_dir, f"step-{global_step}")
                print(f"--- Saving model checkpoint to {save_path} ---\n")
                model.module.llama.save_pretrained(save_path)

            if global_step % index_refresh_interval == 0 and global_step != 0:
                print(f"--- Refreshing index ---")
                faiss_index = refresh_index(
                    model,
                    tokenizer,
                    action_data,
                    action_batch_size,
                    max_action_length=max_action_length
                )

            global_step += 1

    save_path = os.path.join(checkpoint_dir, f"step-{global_step}")
    print(f"--- Saving model checkpoint to {save_path} ---\n")
    model.module.llama.save_pretrained(save_path)


if __name__ == "__main__":
    fire.Fire(main)

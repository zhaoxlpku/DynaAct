import ast
import math
import warnings
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
import json
import re
from tqdm import tqdm
import os


TERMINAL_OBSERVATION = "No additional observations are required to proceed."


def load_from_hdf5(input_path, start_idx=0, end_idx=None, use_tqdm=False):
    import h5py

    list_of_problem = []
    with h5py.File(input_path, 'r') as hdf5_file:
        if use_tqdm:
            keys = tqdm(hdf5_file.keys(), desc="Reading from HDF5")
        else:
            keys = hdf5_file.keys()
        for idx in keys:
            group = hdf5_file[idx]
            if end_idx is not None and end_idx > 0:
                if int(idx) >= end_idx or int(idx) < start_idx:
                    continue
            problem = {"idx": int(idx)}  # Start with the index
            for key, value in group.attrs.items():
                # Decode JSON strings for lists/dicts; leave other types as is
                if key in ["sketch", "sketches", "list_of_solutions", "list_of_correctness", "retrieved_problems"]:
                    try:
                        problem[key] = json.loads(value) if isinstance(value, str) else value
                    except json.JSONDecodeError:
                        problem[key] = value
                else:
                    problem[key] = value
            list_of_problem.append(problem)
    return list_of_problem


def save_to_hdf5(problems, output_path):
    import h5py

    with h5py.File(output_path, 'w') as hdf5_file:
        for problem in tqdm(problems, desc="Writing to HDF5"):
            idx = problem.pop("idx", None)
            if idx is not None:
                group = hdf5_file.create_group(str(idx))
                for key, value in problem.items():
                    if isinstance(value, (int, float, str)):
                        if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
                            try:
                                json.loads(value)
                                warning_msg = (
                                    f"Warning: Value for key '{key}' (index {idx}) appears to be a JSON string. "
                                    "This may cause errors when loading data from HDF5. "
                                    f"Value: {value[:50]}..."  # Truncate value for readability
                                )
                                warnings.warn(warning_msg, UserWarning)
                            except json.JSONDecodeError:
                                pass
                        group.attrs[key] = value
                    elif isinstance(value, (list, dict)):
                        # Serialize dict as a JSON string
                        group.attrs[key] = json.dumps(value)
                    else:
                        print(f"Unsupported data type for key {key} in problem {idx}.")


def is_correct(correctness_rationale: str):
    if "the solution is correct" in correctness_rationale.lower() or (
            "the proposed solution is correct" in correctness_rationale.lower()):
        return True
    else:
        return False


def get_state(problem: str, observations: List[str]):
    if len(observations) == 0:
        return (
            f"You are given the following problem: {problem}\n\n"
            "Retrieve observations that provide essential hints for solving it."
        )
    else:
        sketch_text = "\n".join(f"{i + 1}. {obs}" for i, obs in enumerate(observations))
        return (
            f"You are given the following problem: {problem}\n\n"
            f"Consider the following ordered observations:\n{sketch_text}\n\n"
            "Based on these, retrieve additional observations that "
            "contribute substantively to resolving the problem "
            "rather than duplicating existing observations."
        )


def add_terminal_observation(observations):
    return observations + [TERMINAL_OBSERVATION]


def load_qlearning_data(list_of_problem, available_actions=None, use_tqdm=False):
    qlearning_data = []
    if not use_tqdm:
        data_to_iter = list_of_problem
    else:
        data_to_iter = tqdm(list_of_problem, desc="Loading training data")

    for problem in data_to_iter:
        sketches = problem["sketches"]
        # list_of_solutions = problem["list_of_solutions"]
        list_of_correctness = problem["list_of_correctness"]
        best_score = 0
        best_sketch = None

        for sketch, correctness in zip(sketches, list_of_correctness):
            score = np.mean([is_correct(rationale) for rationale in correctness])
            if available_actions is not None:
                if score > best_score and all(obs in available_actions for obs in sketch):
                    best_score = score
                    best_sketch = sketch
            else:
                if score > best_score:
                    best_score = score
                    best_sketch = sketch
        if best_sketch is None:
            continue

        best_sketch = add_terminal_observation(best_sketch)
        for timestep in range(len(best_sketch)):
            # state = get_state(problem["problem"], best_sketch[:timestep])
            # action = best_sketch[timestep]

            # print("state:", state)
            # print("action:", action)
            # print("---")
            qlearning_data.append({
                "problem": problem["problem"],
                "sketch": best_sketch[:timestep+1],
            })

    return qlearning_data


def load_qlearning_data_new(list_of_problem, available_actions=None, use_tqdm=False):
    qlearning_data = []
    if not use_tqdm:
        data_to_iter = list_of_problem
    else:
        data_to_iter = tqdm(list_of_problem, desc="Loading training data")

    for problem in data_to_iter:
        sketches = problem["sketches"]
        scores = problem["scores"]
        # list_of_solutions = problem["list_of_solutions"]
        # list_of_correctness = problem["list_of_correctness"]
        best_score = 0
        best_sketch = None

        for sketch, score in zip(sketches, scores):
            if available_actions is not None:
                if score > best_score and all(obs in available_actions for obs in sketch):
                    best_score = score
                    best_sketch = sketch
            else:
                if score > best_score:
                    best_score = score
                    best_sketch = sketch
        if best_sketch is None:
            continue

        best_sketch = add_terminal_observation(best_sketch)
        for timestep in range(len(best_sketch)):
            # state = get_state(problem["problem"], best_sketch[:timestep])
            # action = best_sketch[timestep]

            # print("state:", state)
            # print("action:", action)
            # print("---")
            qlearning_data.append({
                "problem": problem["problem"],
                "sketch": best_sketch[:timestep+1],
            })

    return qlearning_data


def load_available_actions(list_of_problem, use_tqdm=False):
    unique_observations = set()
    ordered_observations = []

    unique_observations.add(TERMINAL_OBSERVATION)
    ordered_observations.append(TERMINAL_OBSERVATION)
    # Extract and process observations
    if not use_tqdm:
        data_to_iter = list_of_problem
    else:
        data_to_iter = tqdm(list_of_problem, desc="Loading observations")
    for problem in data_to_iter:
        sketches = problem["sketches"]
        for sketch in sketches:
            if not sketch:  # Skip empty sketches
                continue
            for observation in sketch:
                assert isinstance(observation, str)
                # if len(observation.split()) < 5 or len(observation.split()) > 60:
                #     continue
                if observation not in unique_observations:
                    unique_observations.add(observation)
                    ordered_observations.append(observation)
    return ordered_observations


def create_attention_mask(token_ids: torch.Tensor, padding_token_id: int, device: torch.device) -> torch.Tensor:
    """
    Create the attention mask for a batch of token IDs.

    Args:
        token_ids (torch.Tensor): Tensor of token IDs with shape (batch_size, seq_len).
        padding_token_id (int): The token ID used for padding (e.g., 0 for many models).
        device (torch.device): The device where the tensor will be located (e.g., cuda, cpu).

    Returns:
        torch.Tensor: The attention mask tensor with shape (batch_size, seq_len) where valid tokens are 1 and padding tokens are 0.
    """
    attention_mask = (token_ids != padding_token_id).long()
    attention_mask = attention_mask.to(device)

    return attention_mask


def tokenize_sentences(
        sentences: List[str],
        tokenizer: Union[str, AutoTokenizer] = 'gpt2',  # Tokenizer model name or an existing tokenizer object
        max_length: int = 2048,  # Max sequence length for truncation/padding
        padding: str = 'longest',  # Padding strategy: 'max_length', 'longest', True/False
        truncation: bool = True,  # Enable truncation if sequences are longer than max_length
) -> List[List[int]]:
    """
    Tokenize a list of sentences into token IDs, with truncation, padding, and EOS token handling.

    Args:
        sentences (List[str]): List of sentences to encode.
        tokenizer (Union[str, AutoTokenizer]): Tokenizer to use, either a model name (ept, 'gpt2') or a pre-initialized tokenizer.
        max_length (int): Maximum length of the encoded sequences. Defaults to 2048.
        padding (Union[str, bool]): Padding strategy. Options: 'max_length', 'longest', True/False.
        truncation (bool): Whether to truncate sequences longer than max_length. Defaults to True.
        add_special_tokens (bool): Whether to add special tokens (like EOS or [CLS]). Defaults to True.
        return_tensors (str): Return format for the output: 'pt' (PyTorch), 'tf' (TensorFlow), 'np' (NumPy), 'list' (Python lists).

    Returns:
        List[List[int]]: List of tokenized sentences, each represented as a list of token IDs.
    """
    # Load the tokenizer (either from model name or pre-initialized tokenizer)
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    eos_token_id = tokenizer.eos_token_id

    encoded_sentences = []

    for sentence in sentences:
        encoding = tokenizer.encode_plus(
            sentence,
            max_length=max_length,
            padding=False,  # Disable padding for now to control padding later
            truncation=truncation,
            add_special_tokens=False,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze().tolist()  # Convert tensor to list
        if isinstance(input_ids, int):
            input_ids = [input_ids]

        # if eos_token_id not in input_ids:
        #     if len(input_ids) < max_length:
        #         input_ids.append(eos_token_id)

        # Append tokenized sentence to list
        encoded_sentences.append(input_ids)

    if padding == 'longest':
        max_len = max(len(seq) for seq in encoded_sentences)
        for i in range(len(encoded_sentences)):
            if len(encoded_sentences[i]) < max_len:
                encoded_sentences[i] = encoded_sentences[i] + [eos_token_id] * (max_len - len(encoded_sentences[i]))

    elif padding == 'max_length':
        for i in range(len(encoded_sentences)):
            if len(encoded_sentences[i]) < max_length:
                encoded_sentences[i] = encoded_sentences[i] + [eos_token_id] * (max_length - len(encoded_sentences[i]))

    for i in range(len(encoded_sentences)):
        if len(encoded_sentences[i]) > max_length:
            encoded_sentences[i] = encoded_sentences[i][:max_length]

    return encoded_sentences


def refresh_index(model, tokenizer, action_data, batch_size=256, max_action_length=512, dim=2048):
    import faiss

    index = faiss.IndexFlatL2(dim)
    # gpu_res = faiss.StandardGpuResources()

    total_batches = math.ceil(len(action_data) / batch_size)
    progress_bar = tqdm(total=total_batches, desc="Processing batches", dynamic_ncols=True)
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(action_data))  # Ensure not to exceed list length

        # Get the current batch
        batch = action_data[start_idx:end_idx]

        tokenized_actions = tokenize_sentences(batch, tokenizer, max_length=max_action_length)  # List[List[int]]
        actions = torch.tensor(tokenized_actions, dtype=torch.long).cuda()

        with torch.no_grad():
            action_embeddings = model(actions=actions, mode="action_embedding")  # Assuming model outputs embeddings as torch.Tensor
            action_embeddings_np = action_embeddings.cpu().detach().numpy()  # Move to CPU and convert to numpy array

        index.add(action_embeddings_np)

        progress_bar.set_postfix({"index_size": index.ntotal})
        progress_bar.update(1)

    progress_bar.close()
    # gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    print(f"Total embeddings added to FAISS index: {index.ntotal}")
    return index


def get_action_embeddings(model, tokenizer, action_data, batch_size=256, max_action_length=512):
    action_embeddings_list = []
    total_batches = math.ceil(len(action_data) / batch_size)
    progress_bar = tqdm(total=total_batches, desc="Processing batches", dynamic_ncols=True)

    total_actions_processed = 0
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(action_data))  # Ensure not to exceed list length

        # Get the current batch
        batch = action_data[start_idx:end_idx]

        tokenized_actions = tokenize_sentences(batch, tokenizer, max_length=max_action_length)  # List[List[int]]
        actions = torch.tensor(tokenized_actions, dtype=torch.long).cuda()

        with torch.no_grad():
            action_embeddings = model(actions=actions, mode="action_embedding")  # Assuming model outputs embeddings as torch.Tensor

        action_embeddings_list.append(action_embeddings.cpu())  # Collect tensor in CPU memory

        # Update total actions processed
        total_actions_processed += len(batch)

        progress_bar.set_postfix({"processed_actions": total_actions_processed})
        progress_bar.update(1)

    progress_bar.close()
    action_embeddings_tensor = torch.cat(action_embeddings_list, dim=0)

    return action_embeddings_tensor


def calculate_distance_scores(action_embeddings, current_sets_indices):
    """
    Args:
        action_embeddings: (n_actions, 2048)
        current_sets_indices: (batch_size, current_set_size)
    Returns:
        scores: (batch_size, n_actions) - for each action, sum of min distances after adding it to set
    """
    batch_size, current_set_size = current_sets_indices.shape
    n_actions = action_embeddings.shape[0]
    device = action_embeddings.device

    # Calculate pairwise distances between all actions
    normalized_embeddings = F.normalize(action_embeddings, p=2, dim=1)
    distances = 1 - torch.mm(normalized_embeddings, normalized_embeddings.t())  # (n_actions, n_actions)

    # For each batch and each candidate action:
    # 1. Get distances between current set items and candidate
    # distances: (n_actions, n_actions)
    # current_sets_indices: (batch_size, current_set_size)
    expanded_distances = distances[current_sets_indices]  # (batch_size, current_set_size, n_actions)
    # print(f"the size of expanded_distances is {expanded_distances.size()}")

    # For each batch and candidate:
    # - For current set items: min distance to (other current items + candidate)
    # - For candidate: min distance to current set items
    min_distances = torch.zeros(batch_size, current_set_size + 1, n_actions, device=device)
    # print(f"the size of min_distances is {min_distances.size()}")

    # For current set items - get min distance to other current items
    # Get distances between all pairs in current set
    current_indices = current_sets_indices.unsqueeze(2).expand(-1, -1, current_set_size)
    current_indices_t = current_indices.transpose(1, 2)
    current_set_distances = distances[current_indices, current_indices_t]  # (batch_size, current_set_size, current_set_size)

    current_set_distances.diagonal(dim1=1, dim2=2).fill_(float('inf'))  # mask self-distances
    # print(f"the size of current_set_distances is {current_set_distances.size()}")

    min_current_distances = torch.min(current_set_distances, dim=2).values  # (batch_size, current_set_size)
    # print(f"the size of min_current_distances is {min_current_distances.size()}")

    # Compare with distances to candidate
    min_distances[:, :current_set_size] = torch.min(
        min_current_distances.unsqueeze(-1).expand(-1, -1, n_actions),
        expanded_distances
    )

    # For candidate - min distance to current set items
    min_distances[:, -1] = torch.min(expanded_distances, dim=1).values

    # Sum all minimum distances
    scores = min_distances.sum(dim=1)  # (batch_size, n_actions)

    return scores


def calculate_qvalue_scores(state_embeddings, action_embeddings):
    """
    Args:
        state_embeddings: (batch_size, 2048)
        action_embeddings: (n_actions, 2048)
    Returns:
        scores: (batch_size, n_actions) - dot product between state and action embeddings
    """
    # Normalize embeddings
    state_embeddings = F.normalize(state_embeddings, p=2, dim=1)  # (batch_size, 2048)
    action_embeddings = F.normalize(action_embeddings, p=2, dim=1)  # (n_actions, 2048)

    # Calculate dot product scores
    scores = torch.mm(state_embeddings, action_embeddings.t())  # (batch_size, n_actions)

    return scores


def get_model_device(model):
    """
    Get model device handling both nn.DataParallel and regular model cases
    Args:
        model: torch model (can be wrapped in DataParallel or not)
    Returns:
        device: torch.device
    """
    if isinstance(model, nn.DataParallel):
        return next(model.module.parameters()).device
    else:
        return next(model.parameters()).device


def build_sets_batch(states, encoder, tokenizer, action_embeddings, max_state_length=2048, ratio=0.8, set_size=5):
    device = get_model_device(encoder)
    tokenized_states = tokenize_sentences(
        states,
        tokenizer,
        max_length=max_state_length,
        padding="longest"
    )
    states = torch.tensor(tokenized_states, dtype=torch.long).to(device)
    with torch.no_grad():
        state_embeddings = encoder(states=states, mode="state_embedding")

    batch_size = state_embeddings.shape[0]
    selected_indices = torch.zeros((batch_size, set_size), dtype=torch.long).cpu()

    # print("batch_size: ", batch_size)
    # print("the size of state_embeddings: ", state_embeddings.size())
    # print("the size of action_embeddings: ", action_embeddings.size())

    state_embeddings = state_embeddings.cpu()
    action_embeddings = action_embeddings.cpu()

    for i in range(set_size):
        # Get distance scores for current sets
        if i == 0:
            current_indices = None
            distance_scores = torch.zeros((batch_size, action_embeddings.shape[0])).cpu()
        else:
            current_indices = selected_indices[:, :i]  # shape: (batch_size, current_size)
            distance_scores = calculate_distance_scores(action_embeddings, current_indices)  # (batch_size, n_actions)
        # print("distance:")
        # print(distance_scores)

        # Get qvalue scores
        qvalue_scores = calculate_qvalue_scores(state_embeddings, action_embeddings)  # (batch_size, n_actions)
        # print("qvalue:")
        # print(qvalue_scores)
        # input(">>>")

        # Combine scores (you can adjust the weights)
        scores = (1 - ratio) * distance_scores + ratio * qvalue_scores  # (batch_size, n_actions)
        scores[:, 0] = float('-inf')

        # Select actions with maximum scores for each batch
        new_indices = scores.argmax(dim=1)  # (batch_size,)
        selected_indices[:, i] = new_indices

    return selected_indices


def search_similar_embeddings(index, query_embeddings, k):
    """
    Given a FAISS index and a set of query embeddings, retrieve the top K nearest neighbor indices.

    Args:
        index (faiss.Index): A pre-built FAISS index (e.g., IndexFlatL2) on the database embeddings.
        query_embeddings (numpy.ndarray): A [num_queries, dim] matrix of query embeddings.
        k (int): The number of nearest neighbors to retrieve for each query embedding.

    Returns:
        (numpy.ndarray, numpy.ndarray): A tuple containing:
            - indices: The indices of the K nearest neighbors for each query embedding.
            - distances: The L2 distances of the K nearest neighbors for each query embedding.
    """
    # Perform the search to get the K nearest neighbors
    distances, indices = index.search(query_embeddings, k)

    return indices, distances


def distances_to_probabilities(distances):
    # Exponentiate the distances to get positive values, which are more suitable for probabilities.
    exp_distances = np.exp(distances - np.max(distances))  # Subtract max to prevent overflow issues.

    # Normalize the exponentiated distances to get the probabilities.
    probabilities = exp_distances / np.sum(exp_distances)

    return probabilities


def sample_sentences(sentences, probabilities, k):
    """
    Sample `k` non-repetitive sentences based on given probabilities.

    Args:
        sentences (list): A list of sentences to sample from.
        probabilities (list): A list of probabilities associated with each sentence.
        k (int): The number of sentences to sample without repetition.

    Returns:
        list: A list of `k` sampled sentences.
    """
    if k > len(sentences):
        raise ValueError("k cannot be greater than the number of sentences")

    # Use numpy's choice for sampling without replacement
    sampled_indices = np.random.choice(
        len(sentences), size=k, replace=False, p=probabilities
    )
    return [sentences[i] for i in sampled_indices]


def get_nearest_actions(index, query_embeddings, k, dataset, sampling=False, sampling_size=1024):
    """
    Retrieve the top K nearest sentences from the FAISS index for a batch of query embeddings.

    Args:
        index (faiss.Index): FAISS index object.
        query_embeddings (numpy.ndarray): A [num_queries, dim] matrix of query embeddings.
        k (int): The number of nearest neighbors to retrieve.
        dataset (Dataset): A dataset object containing the actual sentences (e.g., ActionDataset).

    Returns:
        list of lists: A list of K nearest sentences for each query in the input batch.
    """
    # Perform the search to get the top K nearest indices and distances
    indices, distances = search_similar_embeddings(
        index,
        query_embeddings,
        k if not sampling else sampling_size
    )

    # Convert indices to sentences using the dataset
    nearest_actions = []
    for i in range(indices.shape[0]):  # For each query
        sentences = [dataset[idx] for idx in indices[i]]  # Get corresponding sentences
        probabilities = distances_to_probabilities([distance for distance in distances[i]])
        # print(sentences)
        # print(probs)
        # input(">>>")
        if sampling:
            sentences = sample_sentences(sentences, probabilities, k)

        if k == 1:
            nearest_actions.append(sentences[0])
        else:
            nearest_actions.append(sentences)

    return nearest_actions


def search_index(model, tokenizer, index, state_data, action_data, batch_size=None, max_state_length=2048, k=16, sampling=False, sampling_size=1024):
    if batch_size is None:
        batch_size = len(state_data)

    total_batches = math.ceil(len(state_data) / batch_size)
    all_state_embeddings = None
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(state_data))  # Ensure not to exceed list length

        batch = state_data[start_idx:end_idx]

        tokenized_states = tokenize_sentences(batch, tokenizer, max_length=max_state_length)
        states = torch.tensor(tokenized_states, dtype=torch.long).cuda()

        with torch.no_grad():
            state_embeddings = model(states=states, mode="state_embedding")
            state_embeddings_np = state_embeddings.cpu().detach().numpy()

        if all_state_embeddings is None:
            all_state_embeddings = state_embeddings_np
        else:
            all_state_embeddings = np.concatenate((all_state_embeddings, state_embeddings_np), axis=0)

    return get_nearest_actions(index, all_state_embeddings, k, action_data, sampling=sampling, sampling_size=sampling_size)


def get_state_embeddings(states, model, tokenizer, max_state_length=1024):
    with torch.no_grad():
        tokenized_states = tokenize_sentences(states, tokenizer, max_length=max_state_length)
        states = torch.tensor(tokenized_states, dtype=torch.long).cuda()

        state_embeddings = model(states=states, mode="state_embedding")
    return state_embeddings


def get_top_indices(embeddings, faiss_index):
    # Ensure the embeddings are in float32 format, which is required by FAISS
    embeddings = embeddings.astype(np.float32)

    # Perform the search: retrieve the top-1 neighbor (k=1)
    distances, indices = faiss_index.search(embeddings, k=1)

    # 'indices' contains the indices of the nearest neighbor for each item in the batch
    return indices.squeeze()


def extract_code_snippet(text):
    # Extract the code between ```python and the closing ```
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return the Python code, removing surrounding whitespace
    match = re.search(r'```(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return the Python code, removing surrounding whitespace
    return None


def extract_list_from_code(code_str):
    # Parse the cleaned code string into a Python AST
    try:
        parsed_code = ast.parse(code_str)
        # Loop through all top-level statements
        for node in parsed_code.body:
            if isinstance(node, ast.Assign):  # Look for assignments
                for target in node.targets:
                    if isinstance(node.value, ast.List):  # Check if the assigned value is a List
                        # Extract the list from the assignment
                        return [ast.literal_eval(e) for e in node.value.elts]  # Convert list elements to their literal values
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.List):
                # Extract and return the elements from the list
                return [ast.literal_eval(e) for e in node.value.elts]
    except Exception as e:
        # print(f"Error parsing code: {e}")
        pass
    return None


def extract_parentheses(text):
    pattern = r'The correct answer is \((.*?)\)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ''


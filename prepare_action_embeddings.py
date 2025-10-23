from utils import get_action_embeddings, load_available_actions, load_qlearning_data_new
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from qvalue_encoder import QValueEncoder
import json
import fire

def main(
    data_path="demo/platypus_sketches.jsonl",
    output_path="action_embeddings.pt",
    submodular_base_path="/ossfs/workspace/nas/xueliang/hf_models/Llama-3.2-1B-Instruct",
    submodular_lora_path="qlearning/qlearning_v1/checkpoints/step-9000",
):
    problems = []
    with open(data_path, encoding="utf-8") as f:
        for line in f.readlines():
            problems.append(json.loads(line))
    print("number of problems: ", len(problems))
    action_data = load_available_actions(problems, use_tqdm=True)
    print("number of observations: ", len(action_data))

    training_data = load_qlearning_data_new(problems, set(action_data), use_tqdm=True)
    print("number of training data: ", len(training_data))
    print(">>>")

    tokenizer = AutoTokenizer.from_pretrained(submodular_base_path)
    model = QValueEncoder(
        llama_model_path=submodular_base_path,
        memory_ratio=0.2,
        lora_path=submodular_lora_path,
        tau=0.8,
        pooling_method="lasttoken",
        eos_token_id=tokenizer.eos_token_id,
        is_trainable=False
    )
    model = model.cuda()
    model = nn.DataParallel(model)

    action_embeddings = get_action_embeddings(
        model,
        tokenizer,
        action_data,
        batch_size=512,
        max_action_length=128,
    )
    assert len(action_embeddings) == len(action_data), "Number of embeddings must match number of actions"

    data = {
        'embeddings': action_embeddings,
        'actions': action_data
    }
    torch.save(data, output_path)



if __name__ == "__main__":
    fire.Fire(main)
    


    
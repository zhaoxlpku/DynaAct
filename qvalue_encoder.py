import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from peft import PeftModelForCausalLM, LoraConfig, TaskType
from tqdm import tqdm

class QValueEncoder(nn.Module):
    def __init__(self, llama_model_path, memory_ratio=0.1, lora_path=None, lora_rank=16, lora_alpha=32, tau=1.0, pooling_method="mean", eos_token_id=128009, is_trainable=False):
        super(QValueEncoder, self).__init__()

        self.memory_ratio = memory_ratio
        self.tau = tau
        self.pooling_method = pooling_method
        self.eos_token_id = eos_token_id

        # Load the pre-trained LLaMA model (base model)
        self.llama = LlamaForCausalLM.from_pretrained(llama_model_path)
        # self.llama_hidden_size = self.llama.config.hidden_size

        # Apply LoRA configuration (will add LoRA layers to the model)
        if lora_path is None:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],  # Target attention layers (e.g., query and value projections)
                task_type=TaskType.CAUSAL_LM  # Task type: causal language modeling
            )

            # Apply LoRA to the LLaMA model (will add the low-rank matrices)
            self.llama = PeftModelForCausalLM(self.llama, lora_config)
        else:
            self.llama = PeftModelForCausalLM.from_pretrained(self.llama, lora_path, is_trainable=is_trainable)
        self.llama.print_trainable_parameters()

        # # Projection layers for state and action embeddings
        # self.projection_state = nn.Linear(self.llama_hidden_size, hidden_size)
        # self.projection_action = nn.Linear(self.llama_hidden_size, hidden_size)

    def encode(self, token_ids):
        if self.pooling_method == "mean":
            attention_mask = torch.tensor(token_ids != self.eos_token_id, dtype=torch.long, device=token_ids.device)
            outputs = self.llama.model(input_ids=token_ids, attention_mask=attention_mask, output_hidden_states=True)
            # print(len(outputs.hidden_states))
            # print(type(outputs))

            last_hidden_state = outputs.hidden_states[-1]

            attention_mask_expanded = attention_mask.unsqueeze(-1)
            masked_embeddings = last_hidden_state * attention_mask_expanded

            valid_tokens_count = attention_mask_expanded.sum(dim=1, keepdim=True)
            valid_tokens_count = torch.maximum(valid_tokens_count, torch.ones_like(valid_tokens_count))

            avg_embeddings = masked_embeddings.sum(dim=1) / valid_tokens_count.squeeze(-1)  # Shape: [batch_size, dim]
            return F.normalize(avg_embeddings, p=2, dim=-1)
        elif self.pooling_method == "lasttoken":
            # attention_mask = torch.tensor(token_ids != self.eos_token_id, dtype=torch.long, device=token_ids.device)
            # outputs = self.llama.model(input_ids=token_ids, attention_mask=attention_mask, output_hidden_states=True)

            outputs = self.llama.model(input_ids=token_ids, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

            device = token_ids.device

            # Create a mask where eos_token_id is found
            eos_mask = (token_ids == self.eos_token_id)  # (batch_size, seq_len)

            # Convert eos_mask to int32 for argmax (necessary for CUDA)
            eos_mask_int = eos_mask.to(torch.long)  # (batch_size, seq_len)

            # Get the first occurrence of eos_token_id for each sequence in the batch
            first_eos_positions = eos_mask_int.argmax(dim=1)  # (batch_size,)

            # Handle the case when no EOS token is found in the sequence (argmax returns 0 when no EOS is found)
            eos_not_found = eos_mask_int.sum(dim=1) == 0  # (batch_size,) True if no EOS token found

            first_eos_positions[eos_not_found] = token_ids.size(1) - 1  # Default fallback: last token (seq_len - 1)

            # Create an index tensor for the batch (batch_size,)
            batch_indices = torch.arange(token_ids.size(0), device=device)  # (batch_size,)

            # Gather the embeddings at the first EOS token positions (or fallback position)
            eos_embeddings = last_hidden_state[batch_indices, first_eos_positions]  # (batch_size, dim)

            return F.normalize(eos_embeddings, p=2, dim=-1)
        else:
            raise NotImplementedError

    @torch.cuda.amp.autocast()
    def process_in_batches(self, func, inputs, desc=None):
        if inputs is None:
            return None

        batch_size = 1

        outputs = []
        total_batches = (inputs.size(0) + batch_size - 1) // batch_size

        for i in range(0, inputs.size(0), batch_size):
            with torch.cuda.amp.autocast():
                end_idx = min(i + batch_size, inputs.size(0))
                batch_input = inputs[i:end_idx]
                batch_output = func(batch_input)
                outputs.append(batch_output)

        return torch.cat(outputs, dim=0)

    def forward(self, states=None, actions=None, mode="state_embedding"):
        if mode == "action_embedding":
            return self.process_in_batches(self.get_action_embedding, actions, desc="Processing action embeddings")
        elif mode == "state_embedding":
            return self.process_in_batches(self.get_state_embedding, states, desc="Processing state embeddings")
        else:
            raise NotImplementedError

    def get_state_embedding(self, states):
        return self.encode(states)

    def get_action_embedding(self, actions):
        return self.encode(actions)


if __name__ == "__main__":
    pass

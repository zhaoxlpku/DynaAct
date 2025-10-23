import torch
import torch.nn as nn


class TopKSimilarActions(nn.Module):
    def __init__(self, actions_list, K):
        super(TopKSimilarActions, self).__init__()
        self.actions_list = actions_list
        self.K = K

    def forward(self, batch_tensor, actions_tensor):
        """
        Retrieve the top-K similar actions for a batch of inputs using inner product.

        Args:
            batch_tensor (torch.Tensor): A tensor of shape [batch_size, dim].
            actions_tensor (torch.Tensor): A tensor of shape [num_actions, dim].

        Returns:
            List[List[int]]: A list of lists, where each sub-list contains the top-K action identifiers.
        """
        # Calculate inner product similarity
        similarities = torch.matmul(batch_tensor, actions_tensor.t())  # Shape: [batch_size, num_actions]

        # Get the top K indices for each batch
        top_k_indices = torch.topk(similarities, self.K, dim=1).indices
        # print("top_k_indices: ")
        # print(top_k_indices)

        # Retrieve the corresponding actions
        # top_k_actions = [[self.actions_list[idx.item()] for idx in indices] for indices in top_k_indices]
        # print("top_k_actions:")
        # print(top_k_actions)
        # input(">>>")

        return top_k_indices


# Example usage
if __name__ == "__main__":
    # Sample data
    batch_tensor = torch.rand(5, 10)  # Example batch of size 5 with dimension 10
    actions_tensor = torch.rand(20, 10)  # Example actions of size 20 with dimension 10
    num_actions_list = list(range(20))  # Placeholder action identifiers
    K = 3  # Retrieve top 3 actions

    # Create the model and wrap it with DataParallel
    model = TopKSimilarActions(num_actions_list, K)
    model = nn.DataParallel(model)  # Wrap the model in DataParallel

    # Move model and tensors to the same device if necessary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    batch_tensor = batch_tensor.to(device)
    actions_tensor = actions_tensor.to(device)

    # Retrieve top-K actions
    top_actions = model(batch_tensor, actions_tensor)
    print(top_actions)
"""
the value of alpha is determined empirically. 
In the experiments described in the paper, 
the hyperparameter alpha is searched in the 
scopes {0, 0.01, 0.1, 1, 10, 100} on five fine-grained datasets. 
The best value of alpha is chosen based on 
the performance of the method on these datasets.

"""


def aggregate_features(global_feature, local_feature, alpha):
    """
    Aggregates global and local features.
    
    Args:
        global_feature (torch.Tensor): Global feature tensor.
        local_feature (torch.Tensor): Local feature tensor.
        alpha (float): Weight balancing the effect of different features.

    Returns:
        torch.Tensor: Aggregated feature tensor.
    """
    aggregated_feature = global_feature + alpha * local_feature
    return aggregated_feature
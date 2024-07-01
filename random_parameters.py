import torch.nn.init as init


def reset_parameters(parameters) -> None:
    for param in parameters:
        if param.requires_grad:
            if len(param.size()) > 1:
                init.xavier_uniform_(param.data)
            else:
                init.normal_(param.data, mean=0.0, std=0.02)

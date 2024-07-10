import numpy as np
from tqdm import tqdm


def get_grad_norms(model, loader, loss_fn):
    grad_norms = {}
    param_grad_norms = {}
    losses = []
    for x, y in tqdm(loader):
        logits, cache = model.run_with_cache(x)
        loss = loss_fn(logits, y)
        model.zero_grad()
        loss.backward()
        losses.append(loss.item())
        for k, v in cache.items():
            if k not in grad_norms:
                grad_norms[k] = v.grad.mean(dim=0) / len(loader)
            else:
                grad_norms[k] += v.grad.mean(dim=0) / len(loader)

        for k, v in model.named_parameters():
            if k not in param_grad_norms:
                param_grad_norms[k] = v.grad.mean(dim=0) / len(loader)
            else:
                param_grad_norms[k] += v.grad.mean(dim=0) / len(loader)

    for k, v in grad_norms.items():
        grad_norms[k] = v.norm().item()
    for k, v in param_grad_norms.items():
        param_grad_norms[k] = v.norm().item()

    return {
        "grad_norms": grad_norms,
        "param_grad_norms": param_grad_norms,
        "loss": np.mean(losses),
    }

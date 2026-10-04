# Type stub for the compiled extension (csrc/bindings.cpp), so type checkers can see it.
import torch

def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    variant: str = "opt",
) -> torch.Tensor: ...

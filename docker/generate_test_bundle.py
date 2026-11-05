"""Generate a lightweight ONNX bundle for slim serving testing."""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn
from mangosta.export import export_onnx


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "/build/test_bundle.onnx.bundle"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    net = Net().eval()
    export_onnx(
        net,
        output_path,
        sample_input=torch.randn(1, 8),
        as_bundle=True,
    )
    print(f"Generated test bundle at {output_path}")


if __name__ == "__main__":
    main()

"""Fixed inputs and fixed weights for the numeric tests (numpy and torch only), and the writer of their references.

The references in tests/fixtures/ were written once by this file from the code as released (the K2 rehearsal of
that code reproduced the submitted rehearsal CSV); the tests compare the code against them. To rewrite them, which
is only right after a deliberate change of the numerics:

    python -B -m tests.numeric_fixtures --write
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).absolute().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
FORWARD_REFERENCE = FIXTURES / "forward_fixed_weights_32.json"
MAP_REFERENCE = FIXTURES / "run2_map_synthetic_1024.npz"
END_TO_END_SIZE = 1024


def fixed_weights(model):
    """Overwrite every parameter of model with values from a formula (no random generator)."""
    with torch.no_grad():
        for index, (name, value) in enumerate(sorted(model.state_dict().items())):
            count = value.numel()
            wave = torch.sin(torch.arange(count, dtype=torch.float64) * 0.731 + index * 1.37)
            if name.endswith(".weight") and value.dim() == 4:            # convolutions: about 1/sqrt(fan in)
                fan_in = value.shape[1] * value.shape[2] * value.shape[3]
                values = wave * (1.7 / fan_in ** 0.5)
            elif name.endswith(".weight"):                               # GroupNorm scale
                values = 1.0 + 0.25 * wave
            else:                                                        # biases
                values = 0.1 * wave
            value.copy_(values.reshape(value.shape).to(value.dtype))
    return model


def fixed_input(size=32):
    """A (1, 1, size, size) float32 tensor in [0, 1] from a formula."""
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    image = 0.5 + 0.45 * np.sin(rows * 0.61 + cols * 0.23) * np.cos(cols * 0.47 - rows * 0.19)
    return torch.from_numpy(image.astype(np.float32)[None, None])


def synthetic_image(size=END_TO_END_SIZE):
    """A uint8 H-alpha-like test image from a formula: a bright disc with dark curved streaks."""
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    centre = size / 2.0
    radius = np.hypot(rows - centre, cols - centre)
    disc = np.where(radius < size * 0.47, 170.0, 15.0)
    streaks = 0.0
    for k, (a, b, c) in enumerate(((0.011, 0.004, 3.0), (0.007, -0.009, 1.0), (-0.005, 0.012, 5.0))):
        line = np.sin(rows * a + cols * b + c) * 120 + (k - 1) * 150
        streaks = streaks + 90.0 * np.exp(-((rows - centre - line) ** 2) / 60.0)
    texture = 8.0 * np.sin(rows * 0.37) * np.cos(cols * 0.29)
    return np.clip(disc - streaks * (radius < size * 0.47) + texture, 0, 255).astype(np.uint8)


def forward_output():
    from solar_filament.model import UNet
    model = fixed_weights(UNet()).eval()
    with torch.inference_mode():
        return model(fixed_input()).double().numpy()[0, 0]


def end_to_end_map():
    from solar_filament import inference
    candidate = json.loads((ROOT / "candidate.json").read_bytes())
    item = candidate["models"][0]
    model = inference.load_model(ROOT / item["file"], item["sha256"])
    return inference.quantise(inference.probability_map(model, synthetic_image()))


def write():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    output = forward_output()
    FORWARD_REFERENCE.write_bytes((json.dumps({
        "note": "UNet logits for tests.numeric_fixtures.fixed_input() with fixed_weights(); float64 repr of float32 values",
        "shape": list(output.shape), "values": [float(v) for v in output.ravel()]}, indent=0) + "\n").encode("utf-8"))
    np.savez_compressed(MAP_REFERENCE, p=end_to_end_map())


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python -B -m tests.numeric_fixtures --write")
    write()

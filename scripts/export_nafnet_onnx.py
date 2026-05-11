import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.models.archs.arch_util import LayerNorm2d


class ExportLayerNorm2d(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = nn.Parameter(source.bias.detach().clone())
        self.eps = source.eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


def replace_layer_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, LayerNorm2d):
            setattr(module, name, ExportLayerNorm2d(child))
        else:
            replace_layer_norm(child)


def load_state_dict(checkpoint_path, param_key):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and param_key in checkpoint:
        state_dict = checkpoint[param_key]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("net_g."):
            key = key[len("net_g.") :]
        cleaned[key] = value
    return cleaned


def build_model():
    return NAFNet(
        img_channel=3,
        width=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )


def set_static_output_shape(onnx_path, height, width):
    import onnx

    model = onnx.load(onnx_path.as_posix())
    output_shape = [1, 3, height, width]
    for output in model.graph.output:
        dims = output.type.tensor_type.shape.dim
        for dim, value in zip(dims, output_shape):
            dim.dim_value = value
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path.as_posix())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="experiments/pretrained_models/NAFNet-GoPro-width64.pth",
    )
    parser.add_argument("--output", default="rdk_x5/NAFNet-GoPro-width64-256.onnx")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--param-key", default="params")
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("NAFNet export size should be divisible by 16.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    state_dict = load_state_dict(args.weights, args.param_key)
    missing, unexpected = model.load_state_dict(state_dict, strict=not args.no_strict)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    replace_layer_norm(model)
    model.eval()

    dummy = torch.randn(1, 3, args.height, args.width)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path.as_posix(),
            input_names=["input"],
            output_names=["output"],
            opset_version=args.opset,
            do_constant_folding=True,
        )
    set_static_output_shape(output_path, args.height, args.width)
    print(f"Exported ONNX: {output_path}")


if __name__ == "__main__":
    main()

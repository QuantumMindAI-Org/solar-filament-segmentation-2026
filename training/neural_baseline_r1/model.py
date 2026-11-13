"""From-scratch grayscale U-Net and fixed, full-resolution reconstruction."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 20261108
TILE = 512
STRIDE = 384
MAX_INSTANCES = 512


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outgoing), nn.ReLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outgoing), nn.ReLU())


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.ModuleList([Block(a, b) for a, b in
                                  ((1, 16), (16, 32), (32, 64), (64, 128), (128, 256))])
        self.up = nn.ModuleList([Block(a, b) for a, b in
                                ((384, 128), (192, 64), (96, 32), (48, 16))])
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, value):
        skips = []
        for index, block in enumerate(self.down):
            value = block(value)
            if index < 4:
                skips.append(value)
                value = F.max_pool2d(value, 2)
        for block, skip in zip(self.up, reversed(skips)):
            value = F.interpolate(value, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            value = block(torch.cat((value, skip), dim=1))
        return self.head(value)


def loss(logits, target):
    """Equal BCE and soft-Dice terms; no validation-selected weights."""
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    dice = (2 * (probability * target).sum(axes) + 1) / (
        probability.sum(axes) + target.sum(axes) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - dice).mean()


def augment(image, mask, rng):
    """Paired right-angle rotations and independent horizontal reflection."""
    turns = rng.randrange(4)
    image, mask = np.rot90(image, turns), np.rot90(mask, turns)
    if rng.randrange(2):
        image, mask = image[:, ::-1], mask[:, ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def training_crop(image, mask, rng, tile=TILE):
    if image.shape != mask.shape or min(image.shape) < tile:
        raise ValueError('TRAINING_PAIR_SHAPE')
    if rng.randrange(2) and mask.any():
        points = np.flatnonzero(mask)
        y, x = divmod(int(points[rng.randrange(len(points))]), mask.shape[1])
        y = min(max(0, y-tile//2), image.shape[0]-tile)
        x = min(max(0, x-tile//2), image.shape[1]-tile)
    else:
        y = rng.randrange(image.shape[0]-tile+1)
        x = rng.randrange(image.shape[1]-tile+1)
    return augment(image[y:y+tile, x:x+tile], mask[y:y+tile, x:x+tile], rng)


def positions(size, tile=TILE, stride=STRIDE):
    if not (type(size) is int and type(tile) is int and type(stride) is int
            and 16 <= tile <= size <= 2048 and 0 < stride <= tile):
        raise ValueError('TILE_GEOMETRY')
    result = list(range(0, size - tile + 1, stride))
    if result[-1] != size - tile:
        result.append(size - tile)
    return result


def tiled_probability(model, image, *, device, check=lambda: None, tile=TILE, stride=STRIDE):
    """Average sigmoid probabilities in overlaps; never stitch thresholded masks."""
    if (not isinstance(image, np.ndarray) or image.ndim != 2 or image.dtype != np.uint8
            or min(image.shape) < tile or max(image.shape) > 2048):
        raise ValueError('IMAGE_CONTRACT')
    total = np.zeros(image.shape, np.float64)
    count = np.zeros(image.shape, np.uint16)
    model.eval()
    with torch.inference_mode():
        for y in positions(image.shape[0], tile, stride):
            for x in positions(image.shape[1], tile, stride):
                check()
                crop = image[y:y+tile, x:x+tile].astype(np.float32) / 255.0
                value = torch.from_numpy(crop[None, None]).to(device)
                logits = model(value)
                if tuple(logits.shape) != (1, 1, tile, tile):
                    raise ValueError('MODEL_OUTPUT_SHAPE')
                probability = logits.sigmoid()[0, 0].cpu().numpy()
                if not np.isfinite(probability).all():
                    raise ValueError('NONFINITE_PREDICTION')
                total[y:y+tile, x:x+tile] += probability
                count[y:y+tile, x:x+tile] += 1
                check()
    if not count.all():
        raise ValueError('UNCOVERED_PIXEL')
    return (total / count).astype(np.float32)


def instances(probability, *, maximum=MAX_INSTANCES):
    """Fixed >=0.5 threshold and eight-connectivity. No area removal or truncation."""
    from scipy import ndimage
    if (not isinstance(probability, np.ndarray) or probability.ndim != 2
            or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1
            or type(maximum) is not int or not 1 <= maximum <= MAX_INSTANCES):
        raise ValueError('PROBABILITY_CONTRACT')
    labels, number = ndimage.label(probability >= 0.5, structure=np.ones((3, 3), np.uint8))
    if number > maximum:
        raise ValueError('INSTANCE_BUDGET_EXCEEDED_NO_TRUNCATION')
    return labels, int(number)

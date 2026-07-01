# Path: src/data/transforms.py
# Status: MODIFIED
"""Image transforms for CIFT/RIFT.

CIFT expects BCHW float tensors in [-1, 1].
The old loader only used ToTensor(), which gives [0, 1].
"""

def build_transform(size=256, train=False):
    try:
        from torchvision import transforms as T

        ops = [T.Resize((size, size))]

        if train:
            ops.append(T.RandomHorizontalFlip())

        ops.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        return T.Compose(ops)

    except Exception:
        return lambda x: x
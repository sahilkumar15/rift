# Path: src/data/transforms.py
# Status: NEW
"""Image transforms; torchvision if present, else identity."""
def build_transform(size=256, train=False):
    try:
        from torchvision import transforms as T
        ops=[T.Resize((size,size)), T.ToTensor()]
        if train: ops.insert(1, T.RandomHorizontalFlip())
        return T.Compose(ops)
    except Exception:
        return lambda x: x

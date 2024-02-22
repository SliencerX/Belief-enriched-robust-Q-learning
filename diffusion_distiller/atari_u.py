from diffusion_distiller.train_utils import *
from diffusion_distiller.unet_ddpm import UNet
from diffusion_distiller.celeba_dataset import *
# from train_utils import *
# from unet_ddpm import UNet
# from celeba_dataset import *
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import transforms as T, utils
from PIL import Image

BASE_NUM_STEPS = 256
BASE_TIME_SCALE = 1

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

def make_model():
    net = UNet(in_channel = 1,
        channel = 64,
        channel_multiplier = [1, 2],
        n_res_blocks = 2,
        attn_strides = [8, 16],
        attn_heads = 4,
        use_affine_time = True,
        dropout = 0,
        fold = 1)
    net.image_size = [1, 1, 84, 84]
    return net

def make_dataset():
    return Dataset('./pong_pic', 84, augment_horizontal_flip = True, convert_image_to = None)

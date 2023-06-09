# -*-coding:utf-8-*-
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import os
import collections
import numpy as np
import random
from skimage.util import random_noise
from torchvision.transforms import InterpolationMode

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class TestDataLoader(nn.Module):
    def __init__(self, image_dir, batch_size=1, num_workers=4):
        super(TestDataLoader, self).__init__()
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self):
        """Build and return a data loader."""
        transform = list()
        transform.append(T.Resize([256, 320]))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)

        dataset = ImageFolder(self.image_dir, transform)

        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers,
                                      drop_last=False)
        return data_loader


class TrainDataLoader(nn.Module):
    def __init__(self, image_dir, batch_size=4, image_size=256, crop_size=256, num_workers=4):
        super(TrainDataLoader, self).__init__()
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.crop_size = crop_size

    def forward(self):
        """Build and return a data loader."""
        dataset = TrainData(self.image_dir, self.image_size, self.crop_size)
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      drop_last=True)
        return data_loader





class TrainData(data.Dataset):
    def __init__(self, data_root, image_size, crop_size):
        super(TrainData, self).__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.crop_size = crop_size

        self.dir_muddy = os.path.join(self.data_root, "trainA")
        # self.dir_muddy = os.path.join(self.data_root, "Imgs")
        self.muddy_paths = sorted(make_dataset(self.dir_muddy))
        self.muddy_path = None
        self.muddy_img = None

        self.dir_clean = os.path.join(self.data_root, "trainB")
        # self.dir_clean = os.path.join(self.data_root, "GT")
        self.clean_paths = sorted(make_dataset(self.dir_clean))
        self.clean_path = None
        self.clean_img = None

        if len(self.muddy_paths) != len(self.clean_paths):
            raise Exception(" data pairs are not the same size!!!")

        self.data_size = len(self.muddy_paths)
        self.transform = T.Compose([
            Resize(self.image_size),
            # RandomCrop(self.crop_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            # RandomNoise(),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        results = dict()
        self.muddy_path = self.muddy_paths[item % self.data_size]
        self.clean_path = self.clean_paths[item % self.data_size]

        results["muddy"] = Image.open(self.muddy_path).convert('RGB')
        results["clean"] = Image.open(self.clean_path).convert('RGB')

        basename = os.path.basename(self.muddy_path)
        out = self.transform(results)
        out["name"] = basename
        return out

    def __len__(self):

        return getattr(self, "data_size")

    def name(self):
        return 'UnalignedDataset'





class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_dic):
        """
        Args:
            img dict(PIL Image): Image to be scaled.
            clean: clean Image
            muddy: muddy Image

        Returns:
            PIL Image: Rescaled image.
        """
        muddy, clean = img_dic["muddy"], img_dic["clean"]
        muddy = F.resize(muddy, self.size, self.interpolation)
        clean = F.resize(clean, self.size, self.interpolation)
        img_dic["muddy"] = muddy
        img_dic["clean"] = clean
        return img_dic

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomCrop(T.RandomCrop):
    def __init__(self, size, padding=0, pad_if_needed=False):
        super(RandomCrop, self).__init__(size, padding, pad_if_needed)

    def __call__(self, img_dic):
        """
        Args:
            img dict(PIL Image): Image to be scaled.
            clean: clean Image
            muddy: muddy Image
        Returns:
            PIL Image: Cropped image.
        """
        muddy, clean = img_dic["muddy"], img_dic["clean"]
        w, h = muddy.size
        if w > self.size[0] and h > self.size[1]:
            i, j, h, w = self.get_params(muddy, self.size)
            muddy = F.crop(muddy, i, j, h, w)
            clean = F.crop(clean, i, j, h, w)
            img_dic["muddy"] = muddy
            img_dic["clean"] = clean
        return img_dic

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomNoise(object):
    def __init__(self, p=0.5):
        super(RandomNoise, self).__init__()
        self.p = p

    def __call__(self, img_dic):
        """
            Args:
            img dict(PIL Image): Image to be scaled.
            clean: clean Image
            muddy: muddy Image
            Returns:
            PIL Image: Cropped image.
        """
        muddy = img_dic["muddy"]
        muddy_array = np.array(muddy)
        if random.random() < self.p:
            noise_var = np.random.uniform(0.0, 0.04)
            muddy_array = random_noise(muddy_array/255, 'gaussian', clip=True, var=noise_var) * 255
        if random.random() < self.p:
            muddy_array = random_noise(muddy_array / 255, 's&p', clip=True) * 255
        if random.random() < self.p:
            noise_var = np.random.uniform(0.0, 0.04)
            muddy_array = random_noise(muddy_array/255, 'speckle', clip=True, var=noise_var) * 255
        muddy = Image.fromarray(muddy_array.astype('uint8')).convert('RGB')
        img_dic["muddy"] = muddy
        return img_dic


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_dic):
        """
        Args:
            img_dic (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            muddy, clean = img_dic["muddy"], img_dic["clean"]
            img_dic["muddy"] = F.hflip(muddy)
            img_dic["clean"] = F.hflip(clean)
        return img_dic

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_dic):
        """
        Args:
            img_dic (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            muddy, clean = img_dic["muddy"], img_dic["clean"]
            img_dic["muddy"] = F.vflip(muddy)
            img_dic["clean"] = F.vflip(clean)
        return img_dic

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic_dic):
        """
        Args:
            pic_dic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        muddy, clean = pic_dic["muddy"], pic_dic["clean"]
        pic_dic["muddy"] = F.to_tensor(muddy)
        pic_dic["clean"] = F.to_tensor(clean)
        return pic_dic

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_dic):
        """
        Args:
            tensor_dic (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        muddy, clean = tensor_dic["muddy"], tensor_dic["clean"]
        tensor_dic["muddy"] = F.normalize(muddy, self.mean, self.std)
        tensor_dic["clean"] = F.normalize(clean, self.mean, self.std)
        return tensor_dic

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
if __name__ == '__main__':
    sample_dir = "/media/hry/udata/candelete_test/5434"
    loader = TrainDataLoader("/media/hry/udata/candelete_test/1/", 4)()
    for i, img_dic in enumerate(loader):
        muddy, clean = img_dic["muddy"], img_dic["clean"]
        sample_path = os.path.join(sample_dir, '{}-muddy.jpg'.format(i))
        save_image((denorm(muddy)), sample_path, nrow=1, padding=0)
        sample_path = os.path.join(sample_dir, '{}-clean.jpg'.format(i))
        save_image((denorm(clean)), sample_path, nrow=1, padding=0)



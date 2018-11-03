import os
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

class ImageDataset(Dataset):
    def __init__(self, root, transform):
        super(ImageDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)
    
    def name(self):
        return 'ImageDataset'


def train_transform(cropSize=256):
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(cropSize),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import os
import random


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class MultiTaskBrainDataset(Dataset):
    """
    MultiTaskBrainDataset
    ----------------------
    A unified PyTorch Dataset for three different brain-MRI tasks:
        1. Multi-class classification (4 tumor types)
        2. Binary detection (tumor vs. no tumor)
        3. Segmentation (image + mask pairs)

    Expected directory structure:
        base_dir/
        ├── classification_task/
        │   ├── train/
        │   │   ├── glioma/
        │   │   ├── meningioma/
        │   │   ├── pituitary/
        │   │   └── no_tumor/
        │   └── test/ (same structure)
        └── segmentation_task/
            ├── train/
            │   ├── images/
            │   └── masks/
            └── test/
                ├── images/
                └── masks/

    For segmentation mode, no-tumor images are pulled from the classification
    folder and returned with an all-zero (blank) mask.
    """

    CLASS_MAP = {
        "glioma": 0,
        "meningioma": 1,
        "pituitary": 2,
        "no_tumor": 3
    }

    def __init__(
        self,
        base_dir,
        mode="classification",
        train_or_test="train",
        augment=False,
        new_height=224,
        new_width=224,
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotation_prob=0.3,
        rotation_deg=10,
        normalize=True,
        val_split=0.2,
        subset="train",
        seed=42,
        augment_bias_field=False,
        bias_field_prob=0.3,
        augment_noise=False,
        noise_prob=0.3,
        augment_deform=False,
        deform_prob=0.3
):
        self.base_dir = base_dir
        self.mode = mode
        self.train_or_test = train_or_test
        self.augment = augment
        self.new_height = new_height
        self.new_width = new_width
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_prob = rotation_prob
        self.rotation_deg = rotation_deg
        self.augment_bias_field = augment_bias_field
        self.bias_field_prob = bias_field_prob
        self.augment_noise = augment_noise
        self.noise_prob = noise_prob
        self.augment_deform = augment_deform
        self.deform_prob = deform_prob
        self.normalize = normalize
        self.seed = seed

        seed_everything(seed)

        if mode in ("classification", "detection"):
            self.directory = os.path.join(base_dir, "classification_task", train_or_test)
            self.samples = []
            for class_name in self.CLASS_MAP:
                folder = os.path.join(self.directory, class_name)
                for file in sorted(os.listdir(folder)):
                    image_path = os.path.join(folder, file)
                    self.samples.append((image_path, class_name))

        elif mode == "segmentation":
            self.directory = os.path.join(base_dir, "segmentation_task", train_or_test)
            img_dir = os.path.join(self.directory, "images")
            mask_dir = os.path.join(self.directory, "masks")
            self.samples = []

            # Tumour samples with real masks
            for img in sorted(os.listdir(img_dir)):
                mask = img.replace(".jpg", ".png")
                img_path = os.path.join(img_dir, img)
                mask_path = os.path.join(mask_dir, mask)
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

            # No-tumor samples from classification folder — mask will be
            # generated as all zeros in __getitem__
            no_tumor_dir = os.path.join(
                base_dir, "classification_task", train_or_test, "no_tumor"
            )
            if os.path.exists(no_tumor_dir):
                for file in sorted(os.listdir(no_tumor_dir)):
                    img_path = os.path.join(no_tumor_dir, file)
                    self.samples.append((img_path, None))

        if train_or_test == "train":
            rng = random.Random(seed)
            rng.shuffle(self.samples)

            split_idx = int((1 - val_split) * len(self.samples))

            if subset == "train":
                self.samples = self.samples[:split_idx]
            elif subset == "val":
                self.samples = self.samples[split_idx:]
            else:
                raise ValueError("subset must be 'train' or 'val'")

    def __len__(self):
        return len(self.samples)

    def load_image(self, path):
        # Backwards-compatible image loading for older PyTorch versions.
        with open(path, "rb") as f:
            data = f.read()
        byte_tensor = torch.tensor(list(data), dtype=torch.uint8)
        return torchvision.io.decode_image(byte_tensor)

    def __getitem__(self, index):
        if self.mode in ("classification", "detection"):
            image_path, class_name = self.samples[index]

            image = self.load_image(image_path)
            image = image[0].unsqueeze(0).float()

            if self.normalize:
                image = image / 255.0

            image = self.transform_image(image)

            if self.mode == "classification":
                label_idx = self.CLASS_MAP[class_name]
                label = F.one_hot(torch.tensor(label_idx), num_classes=4).float()
            else:
                label = [0] if class_name == "no_tumor" else [1]
                label = torch.tensor(label, dtype=torch.float32)

            return image, label

        elif self.mode == "segmentation":
            image_path, mask_path = self.samples[index]

            image = self.load_image(image_path)
            image = image[0].unsqueeze(0).float()

            if self.normalize:
                image = image / 255.0

            if mask_path is None:
                # No-tumor sample: resize and augment image, return blank mask
                image = TF.resize(image, [self.new_height, self.new_width])
                image, _ = self.transform_pair(image, image)
                mask = torch.zeros(1, self.new_height, self.new_width, dtype=torch.int)
            else:
                # Tumour sample: load and align real mask
                mask = self.load_image(mask_path)
                mask = (mask / 255.0).round()

                image = TF.resize(image, [self.new_height, self.new_width])
                mask = TF.resize(
                    mask,
                    [self.new_height, self.new_width],
                    interpolation=TF.InterpolationMode.NEAREST_EXACT
                )

                image, mask = self.transform_pair(image, mask)
                mask = mask.int()

            return image, mask

    def transform_image(self, image):
        image = TF.resize(image, [self.new_height, self.new_width])

        if self.augment:
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
            if random.random() < self.rotation_prob:
                angle = random.uniform(-self.rotation_deg, self.rotation_deg)
                image = TF.rotate(image, angle)
            # augment morphological differences
            if self.augment_deform and (random.random() < self.deform_prob):
                img_np = image.squeeze()
                shape = img_np.shape

                alpha = np.random.uniform(45,55)
                sigma = 10
                # random displacement fields
                rng = np.random.RandomState()
                dx = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
                dy = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
                # coordinate grid
                x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                indices = [np.reshape(y + dy, -1), np.reshape(x + dx, -1)]
                # apply displacement fields
                deformed = map_coordinates(img_np, indices, order=1).reshape(shape)
                image = torch.from_numpy(deformed[np.newaxis, ...]).float()
            # augment intrinsic noise
            if self.augment_bias_field and (random.random() < self.bias_field_prob):
                noise_scale = np.random.uniform(0.002, 0.2)
                noise = np.random.normal(loc=0.0, scale=noise_scale, size=image.shape)
                small_noise = np.clip(noise, 0.0, 1.0)
                image = (image + small_noise).float()
            # augment field inhomogeneities
            if self.augment_noise and (random.random() < self.noise_prob):
                # create low variance map
                rng = np.random.RandomState()
                noise = rng.randn(*image.shape)
                smooth = gaussian_filter(noise, sigma=20, mode='reflect')
                smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-12) # normalize
                # create and apply bias field
                alpha = np.random.uniform(0.1, 0.5)  # scales the bias field
                bias_field = np.exp(alpha * smooth)
                bias_field_rescaled = bias_field / np.mean(bias_field)  # rescale 0-1
                image = (image * bias_field_rescaled).float()

        if self.normalize:
            mask = image > 0.00
            if mask.sum() > 100:
                pixels = image[mask]
                mean = pixels.mean()
                std = pixels.std()
                image = (image - mean) / (std + 1e-6)

        return image

    def transform_pair(self, image, mask):
        if self.augment:
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() < self.rotation_prob:
                angle = random.uniform(-self.rotation_deg, self.rotation_deg)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            # augment morphological differences
            if self.augment_deform and (random.random() < self.deform_prob):
                img_np = image.numpy().squeeze()
                mask_np = mask.numpy().squeeze()
                shape = img_np.shape
                alpha = np.random.uniform(45,55)
                sigma = 10
                # random displacement fields
                rng = np.random.RandomState()
                dx = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
                dy = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
                # coordinate grid
                x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                indices = [np.reshape(y + dy, -1), np.reshape(x + dx, -1)]
                # apply displacement fields
                deformed_image = map_coordinates(img_np, indices, order=1).reshape(shape)
                deformed_mask = map_coordinates(mask_np, indices, order=0).reshape(shape)
                image = torch.from_numpy(deformed_image[np.newaxis, ...]).float()
                mask = torch.from_numpy(deformed_mask[np.newaxis, ...]).float()
            # augment intrinsic noise
            if self.augment_bias_field and (random.random() < self.bias_field_prob):
                noise_scale = np.random.uniform(0.002, 0.2)
                noise = np.random.normal(loc=0.0, scale=noise_scale, size=image.shape)
                small_noise = np.clip(noise, 0.0, 1.0)
                image = (image + small_noise).float()
            # augment field inhomogeneities
            if self.augment_noise and (random.random() < self.noise_prob):
                # create low variance map
                rng = np.random.RandomState()
                noise = rng.randn(*image.shape)
                smooth = gaussian_filter(noise, sigma=20, mode='reflect')
                smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-12) # normalize
                # create and apply bias field
                alpha = np.random.uniform(0.1, 0.5)  # scales the bias field
                bias_field = np.exp(alpha * smooth)
                bias_field_rescaled = bias_field / np.mean(bias_field)  # rescale 0-1
                image = (image * bias_field_rescaled).float()

        if self.normalize:
            fg = image > 0.00
            if fg.sum() > 100:
                pixels = image[fg]
                mean = pixels.mean()
                std = pixels.std()
                image = (image - mean) / (std + 1e-6)

        return image, mask
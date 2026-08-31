from torchvision import transforms

from preprocessing.clahe import CLAHETransform

import config


train_transform = transforms.Compose([

    CLAHETransform(),

    transforms.ToPILImage(),

    transforms.Resize(
        (config.IMAGE_SIZE,
         config.IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.RandomAffine(
        degrees=0,
        scale=(0.9,1.1)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])


test_transform = transforms.Compose([

    CLAHETransform(),

    transforms.ToPILImage(),

    transforms.Resize(
        (config.IMAGE_SIZE,
         config.IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])
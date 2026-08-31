
import os
import random

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import config

from preprocessing.transforms import train_transform
from preprocessing.transforms import test_transform


class CovidDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.transform = transform
        self.samples = []

        classes = {
            "COVID": 0,
            "Healthy": 1
        }

        for class_name, label in classes.items():

            class_folder = os.path.join(
                root_dir,
                class_name
            )

            if not os.path.exists(class_folder):
                continue

            patient_folders = sorted(
                os.listdir(class_folder)
            )

            for patient in patient_folders:

                patient_path = os.path.join(
                    class_folder,
                    patient
                )

                if not os.path.isdir(patient_path):
                    continue

                for image_name in os.listdir(patient_path):

                    if image_name.lower().endswith(
                        (
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".bmp"
                        )
                    ):

                        image_path = os.path.join(
                            patient_path,
                            image_name
                        )

                        self.samples.append(
                            (
                                image_path,
                                label,
                                patient
                            )
                        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label, patient = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders():

    random.seed(config.SEED)

    dataset_root = config.DATASET_PATH

    patient_dict = {}

    classes = {
        "COVID": 0,
        "Healthy": 1
    }

    # ---------------------------------------
    # Group images by patient
    # ---------------------------------------

    for class_name, label in classes.items():

        class_folder = os.path.join(
            dataset_root,
            class_name
        )

        if not os.path.exists(class_folder):
            continue

        for patient in sorted(
            os.listdir(class_folder)
        ):

            patient_path = os.path.join(
                class_folder,
                patient
            )

            if not os.path.isdir(patient_path):
                continue

            key = (
                class_name,
                patient
            )

            patient_dict[key] = []

            for image_name in os.listdir(patient_path):

                if image_name.lower().endswith(
                    (
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".bmp"
                    )
                ):

                    patient_dict[key].append(
                        os.path.join(
                            patient_path,
                            image_name
                        )
                    )
           
    # ---------------------------------------
    # Stratified Patient Split
    # ---------------------------------------

    covid_patients = []
    healthy_patients = []

    for patient in patient_dict.keys():

        if patient[0] == "COVID":
            covid_patients.append(patient)
        else:
            healthy_patients.append(patient)

    random.shuffle(covid_patients)
    random.shuffle(healthy_patients)

    def split_patients(patient_list):

        n = len(patient_list)

        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        train = patient_list[:train_end]
        val = patient_list[train_end:val_end]
        test = patient_list[val_end:]

        return train, val, test

    covid_train, covid_val, covid_test = split_patients(
        covid_patients
    )

    healthy_train, healthy_val, healthy_test = split_patients(
        healthy_patients
    )

    train_patients = covid_train + healthy_train
    val_patients = covid_val + healthy_val
    test_patients = covid_test + healthy_test

    random.shuffle(train_patients)
    random.shuffle(val_patients)
    random.shuffle(test_patients)

    train_samples = []
    val_samples = []
    test_samples = []

    for patient in train_patients:

        label = classes[patient[0]]

        for image in patient_dict[patient]:

            train_samples.append(
                (
                    image,
                    label
                )
            )

    for patient in val_patients:

        label = classes[patient[0]]

        for image in patient_dict[patient]:

            val_samples.append(
                (
                    image,
                    label
                )
            )

    for patient in test_patients:

        label = classes[patient[0]]

        for image in patient_dict[patient]:

            test_samples.append(
                (
                    image,
                    label
                )
            )

    class PatientDataset(Dataset):

        def __init__(
            self,
            samples,
            transform
        ):

            self.samples = samples
            self.transform = transform

        def __len__(self):

            return len(self.samples)

        def __getitem__(self, idx):

            image_path, label = self.samples[idx]

            image = Image.open(
                image_path
            ).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label
        
    train_dataset = PatientDataset(
        train_samples,
        train_transform
    )

    val_dataset = PatientDataset(
        val_samples,
        test_transform
    )

    test_dataset = PatientDataset(
        test_samples,
        test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    train_covid = sum(
        label == 0
        for _, label in train_samples
    )

    train_healthy = sum(
        label == 1
        for _, label in train_samples
    )

    val_covid = sum(
        label == 0
        for _, label in val_samples
    )

    val_healthy = sum(
        label == 1
        for _, label in val_samples
    )

    test_covid = sum(
        label == 0
        for _, label in test_samples
    )

    test_healthy = sum(
        label == 1
        for _, label in test_samples
    )

    print("=" * 60)
    print("Stratified Patient-Level Split")
    print("=" * 60)

    print(
        f"Train Images      : {len(train_dataset)} "
        f"(COVID={train_covid}, Healthy={train_healthy})"
    )

    print(
        f"Validation Images : {len(val_dataset)} "
        f"(COVID={val_covid}, Healthy={val_healthy})"
    )

    print(
        f"Test Images       : {len(test_dataset)} "
        f"(COVID={test_covid}, Healthy={test_healthy})"
    )

    print("=" * 60)

    return (
        train_loader,
        val_loader,
        test_loader
    )





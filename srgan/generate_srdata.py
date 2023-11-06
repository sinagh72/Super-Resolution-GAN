import os

import cv2
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from srgan import Generator
from train import set_seed, AnimalDatSet, get_test_transformation


# Function to save images from a dataloader
def save_generated_images(dataloader, folder_name):
    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Image counter for cats and dogs
    cat_counter = 0
    dog_counter = 0

    # Loop through the dataloader
    for i, data in enumerate(tqdm(dataloader)):
        # Move images and labels to the same device as the generator
        images = data["img"].to(device)
        labels = data["label"].to(device)

        # Generate images
        with torch.no_grad():
            generated_images = generator(images)

        # Rescale images to [0, 255]
        generated_images = generated_images * 255

        # Save each image
        for img, label in zip(generated_images, labels):
            # Move to CPU and convert to numpy
            img = img.cpu().numpy()
            # Rescale and convert to [height, width, channels] format
            img = np.transpose(img, (1, 2, 0))

            # Determine the filename based on the label
            filename = ''
            if label == 1:  # Assuming 1 is for cats
                filename = f'cat{cat_counter}.jpg'
                cat_counter += 1
            else:  # Assuming 0 or any other number is for dogs
                filename = f'dog{dog_counter}.jpg'
                dog_counter += 1

            # Save the image using OpenCV
            cv2.imwrite(os.path.join(folder_name, filename), img)


if __name__ == "__main__":
    batch_size = 32
    # Set seed for reproducibility
    seed = 10
    set_seed(seed)
    # Create dataset and split it into train, and validation
    cat_dog_dataset = AnimalDatSet(data_root="../dogs-vs-cats (1)/train/train", transformation=get_test_transformation(32))
    train_data, val_data, _ = random_split(cat_dog_dataset, [0.6, 0.1, 0.3],
                                           generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    device = "cuda:0"
    # Create the generator
    generator = Generator(num_residual_blocks=24).to(device)
    generator.load_state_dict(torch.load("best_generator.pth"))
    generator.eval()
    # Save images for training dataset
    save_generated_images(train_loader, 'generated_train_images')

    # Save images for validation dataset
    save_generated_images(val_loader, 'generated_val_images')


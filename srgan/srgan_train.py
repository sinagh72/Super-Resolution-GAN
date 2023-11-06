import copy
import os
import random
from collections import deque

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.utils import clip_grad_norm_, spectral_norm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm

from srgan import weights_init, calculate_psnr, \
    Generator, Discriminator, FeatureExtractor
from train import set_seed, AnimalDatSet, get_transformation, get_test_transformation


class MergedDataset(Dataset):
    def __init__(self, dataset_32, dataset_128):
        self.dataset_32 = dataset_32
        self.dataset_128 = dataset_128
        assert len(dataset_32) == len(dataset_128), "Datasets must be the same size"

    def __len__(self):
        return len(self.dataset_32)

    def __getitem__(self, idx):
        data_32 = self.dataset_32[idx]
        data_128 = self.dataset_128[idx]
        return (data_128, data_32)


def save_images(hr_image, lr_image, fake_image, label, epoch, show=False):
    lr_image = np.transpose(lr_image.cpu().numpy(), (1, 2, 0))
    hr_image = np.transpose(hr_image.cpu().numpy(), (1, 2, 0))
    fake_image = np.transpose(fake_image.cpu().numpy(), (1, 2, 0))

    name = "Cat" if label == 1 else "Dog"
    cv2.imwrite(f"results/{name}_{epoch}_generated.png", fake_image * 255)
    cv2.imwrite(f"results/{name}_{epoch}_hr.png", hr_image * 255)
    cv2.imwrite(f"results/{name}_{epoch}_lr.png", lr_image * 255)

    if show:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs = axs.ravel()

        axs[0].imshow(lr_image, cmap='gray')
        axs[0].set_title('Low Resolution')

        axs[1].imshow(hr_image, cmap='gray')
        axs[1].set_title('High Resolution')

        axs[2].imshow(fake_image, cmap='gray')
        axs[2].set_title('Generated Image')

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        # fig.suptitle(definitions[label])
        plt.show()


def apply_spectral_norm(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.utils.spectral_norm(module)


def train_SRGAN(gen, disc, feature_extractor, train_loader, val_loader,
                epochs, device="cpu"):
    gen.train()
    disc.train()
    best_psnr = 0
    epochs_since_improvement = 0

    # Define loss functions
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.MSELoss().to(device)
    criterion_feature_matching = torch.nn.MSELoss().to(device)

    # Setup AdamW optimizers with different learning rates
    optimizer_G = torch.optim.AdamW(gen.parameters(), lr=1e-5, betas=(0.5, 0.999))
    optimizer_D = torch.optim.AdamW(disc.parameters(), lr=1e-6, betas=(0.5, 0.999))

    # Scheduler for the optimizers (Reduce learning rate when a metric has stopped improving)
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=10, verbose=True)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1, patience=10, verbose=True)

    # Initialize historical discriminators
    historical_discriminators = []
    historical_disc = Discriminator().to(device)  # Create one instance to be used for all historical states

    # Training loop
    for epoch in range(epochs):
        genLossSum = 0
        disLossSum = 0

        for i, (high_data, low_data) in enumerate(tqdm(train_loader)):
            # Transfer data to device
            low_res = low_data["img"].to(device)
            high_res = high_data["img"].to(device)

            # Adversarial ground truths with label smoothing and noise
            valid = torch.FloatTensor(high_res.size(0), 1).uniform_(0.7, 1.2).to(device)
            fake = torch.FloatTensor(high_res.size(0), 1).uniform_(0.0, 0.3).to(device)
            valid = torch.clamp(valid, 0, 1)
            fake = torch.clamp(fake, 0, 1)

            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_high_res = gen(low_res)

            # Adversarial loss
            validity = disc(gen_high_res)
            loss_GAN = criterion_GAN(validity, valid)

            # Content loss
            loss_content = criterion_content(gen_high_res, high_res)

            # Feature matching loss
            real_features = feature_extractor(high_res).detach()
            fake_features = feature_extractor(gen_high_res)
            loss_feature_matching = criterion_feature_matching(fake_features, real_features)

            # Total loss for the generator
            loss_G = loss_content + 0.001 * loss_GAN + 0.006 * loss_feature_matching
            loss_G.backward()
            clip_grad_norm_(gen.parameters(), 1)  # Clip gradients for generator
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion_GAN(disc(high_res), valid)
            fake_loss = criterion_GAN(disc(gen_high_res.detach()), fake)
            loss_D = (real_loss + fake_loss) / 2

            # Historical averaging
            historical_losses = []
            for disc_state in historical_discriminators:
                historical_disc.load_state_dict(disc_state)
                historical_disc.eval()  # Set to evaluation mode for inference
                with torch.no_grad():  # No need to track gradients for historical discriminators
                    historical_losses.append(criterion_GAN(historical_disc(high_res), valid))

            if historical_losses:
                historical_loss = sum(historical_losses) / len(historical_losses)
                loss_D += 0.001 * historical_loss

            loss_D.backward()
            clip_grad_norm_(disc.parameters(), 1)  # Clip gradients for discriminator
            optimizer_D.step()

            # Save the current discriminator state
            if len(historical_discriminators) >= 10:
                historical_discriminators.pop(0)
            historical_discriminators.append(copy.deepcopy(disc.state_dict()))

            # Calculate the losses
            genLossSum += loss_G.item()
            disLossSum += loss_D.item()

        # Update learning rates
        scheduler_G.step(genLossSum)
        scheduler_D.step(disLossSum)

        # Print the losses
        print(
            f"Epoch {epoch + 1}/{epochs} - Generator Loss: {genLossSum / (i + 1)} - Discriminator Loss: {disLossSum / (i + 1)}")

        psnr_values = []
        gen.eval()  # Set the generator to evaluation mode
        rnd = random.randint(0, batch_size - 1)
        with torch.no_grad():
            for i, (high_res, low_res) in enumerate(val_loader):
                high_res_real = high_res["img"].to(device)
                label = low_res["label"]
                low_res = low_res["img"].to(device)
                high_res_fake = gen(low_res)
                # Save some sample images
                if i == 0:  # Change this condition to save images as needed
                    save_images(high_res_real[rnd], low_res[rnd], high_res_fake[rnd], label[rnd], epoch)

                # Calculate PSNR
                psnr_value = calculate_psnr(high_res_real, high_res_fake, device)
                psnr_values.append(psnr_value)

        avg_psnr = sum(psnr_values) / len(psnr_values)
        print(f"Average PSNR on validation set for epoch {epoch + 1}: {avg_psnr} dB")

        # Check for improvement
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            epochs_since_improvement = 0
            torch.save(gen.state_dict(), 'best_generator.pth')
            torch.save(disc.state_dict(), 'best_discriminator.pth')
        else:
            epochs_since_improvement += 1

        # Early stopping
        if epochs_since_improvement == 10:
            print("No improvement in PSNR for 10 consecutive epochs, stopping training.")
            break

        # Set the generator back to train mode
        gen.train()


if __name__ == "__main__":

    batch_size = 64
    seed = 10
    set_seed(seed)
    # Create dataset and split it into train, and validation
    cat_dog_dataset = AnimalDatSet(data_root="../dogs-vs-cats (1)/train/train",
                                   transformation=get_test_transformation(size=128))
    train_data_128, val_data_128, _ = random_split(cat_dog_dataset, [0.65, 0.05, 0.3],
                                                   generator=torch.Generator().manual_seed(seed))

    # Create dataset and split it into train, and validation
    cat_dog_dataset = AnimalDatSet(data_root="../dogs-vs-cats (1)/train/train",
                                   transformation=get_test_transformation(size=32))
    train_data_32, val_data_32, _ = random_split(cat_dog_dataset, [0.65, 0.05, 0.3],
                                                 generator=torch.Generator().manual_seed(seed))

    merged_train_data = MergedDataset(train_data_32, train_data_128)
    merged_val_data = MergedDataset(val_data_32, val_data_128)

    train_loader = DataLoader(merged_train_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
                              num_workers=4)
    val_loader = DataLoader(merged_val_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                            drop_last=False)

    # Sets string definitions based on integer values
    definitions = {0: "Dog", 1: "Cat"}
    samples = 10
    epochs = 150
    lr_g = 1e-3  # Learning rate for generator
    lr_d = 1e-4  # Learning rate for discriminator
    device = "cuda:0"
    # Create the generator
    generator = Generator(num_residual_blocks=24).to(device)
    # Run a forward pass with a dummy 32 to initialize parameters
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Adjust the size as necessary for your model
    generator(dummy_input)
    # Apply the weight initialization
    generator.apply(weights_init)
    # Create the discriminator
    discriminator = Discriminator().to(device)
    dummy_input = torch.randn(1, 3, 128, 128).to(device)  # The size should match the discriminator's expected input
    discriminator(dummy_input)
    discriminator.apply(weights_init)
    # Instantiate the feature extractor
    feature_extractor = FeatureExtractor().eval()  # Set to evaluation mode

    # Move to the device and make sure to not track gradients
    feature_extractor.to('cuda' if torch.cuda.is_available() else 'cpu')
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    train_SRGAN(gen=generator, disc=discriminator, feature_extractor=feature_extractor,
                train_loader=train_loader, val_loader=val_loader, epochs=epochs,
                device=device)

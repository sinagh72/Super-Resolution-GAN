import glob
import itertools
import os

import numpy as np
import torch
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision.models import ResNet18_Weights
import lightning.pytorch as pl
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from train import Model, AnimalDatSet, set_seed, get_test_transformation
from lightning.pytorch.loggers import TensorBoardLogger

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(0)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda(0)


def denormalize(tensor, mean, std):
    return tensor * std + mean


def to_255_range(tensor):
    return (tensor * 255).clamp(0, 255).byte()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    model_path = "./checkpoints/model_A"
    batch_size = 32
    tb_logger = TensorBoardLogger(save_dir=os.path.join(model_path, "tb_log_test/"), name="resnet18")
    # Set seed for reproducibility
    seed = 10
    set_seed(seed)

    cat_dog_dataset = AnimalDatSet(data_root="./dogs-vs-cats (1)/train/train", transformation=get_test_transformation(),
                                   mode="test")
    _, _, test_data = random_split(cat_dog_dataset, [0.6, 0.1, 0.3],
                                   generator=torch.Generator().manual_seed(seed))

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    trainer = pl.Trainer(default_root_dir=model_path,
                         accelerator="gpu",
                         max_epochs=100,
                         logger=[tb_logger],
                         )

    saved_path = glob.glob(model_path + '/resnet*.ckpt')
    checkpoint = torch.load(saved_path[0])

    model_architecture = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = Model(model=model_architecture, lr=1e-4, wd=1e-6)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.cuda(0)
    out = trainer.test(model, test_loader)
    # print(out)
    all_preds = []
    all_labels = []
    cat = 0
    dog = 0

    from torchvision import transforms

    t = transforms.ToPILImage()
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch["img"], batch["label"]
            inputs, labels = inputs.cuda(0), labels.cuda(0)

            outputs = model(inputs)
            # print(outputs)
            # denormalized_image = denormalize(inputs, mean, std)
            # image_255_range = to_255_range(denormalized_image)
            # PIL.Image.Image.show(t(image_255_range.squeeze(0)))
            # Convert sigmoid output to binary labels
            preds = (torch.nn.functional.sigmoid(outputs) >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion mat/rix

    class_names = ["Dog", "Cat"]
    plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized confusion matrix')

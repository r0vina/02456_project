import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
import numpy as np
from structure_tensor import eig_special_2d, structure_tensor_2d, eig_special_3d, structure_tensor_3d
import torch.nn.functional as F
import skimage.io
import matplotlib.pyplot as plt
import scipy as scp
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch.optim
# import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# import torch.nn.functional as F
import wandb
import numpy as np
import torch
import torch.nn as nn
import os
from torch.nn.functional import normalize

torch.cuda.empty_cache()
wandb.init(project = "DLProject", name = f"lr-{os.environ['LEARNING_RATE']}-ep-{os.environ['EPOCHS']}-bs-{os.environ['BATCH_SIZE']}", entity = "02456_project")
wandb.config = {
	"learning_rate":float(os.environ['LEARNING_RATE']),
	"epochs":int(os.environ['EPOCHS']),
	"batch_size":int(os.environ['BATCH_SIZE'])
}
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir)) - 4

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"img_{idx}.jpeg")
        image = skimage.io.imread(img_path, as_gray=True)

        if self.transform:
            x, y = self.transform(image)
        return x, y


class Transform_structure(object):

    def __call__(self, img):
        scale = 0.25
        rho = 8
        sigma = rho / 2
        downsampled_img = scp.ndimage.zoom(img, scale)
        downsampled_S = structure_tensor_2d(downsampled_img, sigma, rho)
        downsampled_val, downsampled_vec = eig_special_2d(downsampled_S)
        downsampled_vec = torch.from_numpy(downsampled_vec)

        transformed_structure_tensor = F.interpolate(downsampled_vec, 400)
        transformed_structure_tensor = transformed_structure_tensor.permute(0, 2, 1)
        transformed_structure_tensor = F.interpolate(transformed_structure_tensor, size=400)
        # transformed_structure_tensor = transformed_structure_tensor.permute(1, 2, 0)

        S = structure_tensor_2d(img, sigma, rho)
        val, vec = eig_special_2d(S)
        # vec = torch.from_numpy(vec).permute(1,2,0)

        img = np.reshape(img, (1, 400, 400))
        img = torch.from_numpy(img)
        img = normalize(img)
        input_img = torch.cat((img, transformed_structure_tensor), 0)
        return input_img, vec

# input_conv og output_conv er integers som angiver dimensionen af hhv. input og output
def double_convolution(input_conv, output_conv):
    convolution = nn.Sequential(
        nn.Conv2d(input_conv, output_conv, kernel_size=3, padding="same"),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(output_conv),
        nn.Dropout2d(Drop_P),
        nn.Conv2d(output_conv, output_conv, kernel_size=3, padding="same"),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(output_conv),
        )
    return convolution

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_1 = double_convolution(3, 16)
        self.down_conv_2 = double_convolution(16, 32)
        self.down_conv_3 = double_convolution(32, 64)
        self.down_conv_4 = double_convolution(64, 128)
        self.down_conv_5 = double_convolution(128, 256)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up_trans_4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.up_trans_5 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)

        self.up_conv_1 = double_convolution(256, 128)
        self.up_conv_2 = double_convolution(128, 64)
        self.up_conv_3 = double_convolution(64, 32)
        self.up_conv_4 = double_convolution(32, 16)

        self.out = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)

    def forward(self, input_data):
        # Encoder
        x1 = self.down_conv_1(input_data)
        x2 = self.max_pool(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool(x7)
        x9 = self.down_conv_5(x8)

        # Decoder which concatenates from the encoder layers
        y = self.up_trans_1(x9)
        y = self.up_conv_1(torch.cat([y, x7], 1))
        y = self.up_trans_2(y)
        y = self.up_conv_2(torch.cat([y, x5], 1))
        y = self.up_trans_3(y)
        y = self.up_conv_3(torch.cat([y, x3], 1))
        y = self.up_trans_4(y)
        y = self.up_conv_4(torch.cat([y, x1], 1))

        # Apply final conv3d layer and sigmoid
        y = self.out(y)
        y_out = torch.sigmoid(y)

        return y_out


# Hyperparameters

LEARNING_RATE =wandb.config['learning_rate']
BATCH_SIZE = wandb.config['batch_size']
NUM_EPOCHS = wandb.config['epochs']
Drop_P = 0.2

LOAD_MODEL = False
LOADPATH = f"model_hpc_ba{BATCH_SIZE}-lr{LEARNING_RATE}-ep{NUM_EPOCHS}.pth.tar"
SAVEPATH = f"model_hpc_ba{BATCH_SIZE}-lr{LEARNING_RATE}-ep{NUM_EPOCHS}.pth.tar"
LOSS_PATH = f"loss_array_ba{BATCH_SIZE}-lr{LEARNING_RATE}-ep{NUM_EPOCHS}.npy"

def training():
    # Parameters
    model = UNet()
    out = 0
    y = 0

    # Set device, send model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    model = nn.DataParallel(model)
    model.to(device)

    # Loss function
    loss_func = torch.nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Dataload
    transformed_dataset = CustomImageDataset(img_dir="trainingimages/",
                                             transform=Transform_structure())

    dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE,
                            shuffle=True)

    # Load model after model and optimizer initialization
    if LOAD_MODEL:
        checkpoint = torch.load(LOADPATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_nr = checkpoint['epoch']
        loss = checkpoint['loss']
        total_loss = checkpoint['total_loss']
        loss_pr_epoch = checkpoint['loss_pr_epoch']

        # print("Model's state_dict:")
        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(f"Loaded model: {LOADPATH}")

    else:
        epoch_nr = 1
        total_loss = []
        loss_pr_epoch = []

    # Training mode initialized
    model.train()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_loss = []
        for i, (feature, y) in enumerate(dataloader):
            # Dataload. Send to device.
            feature, y = feature.float(), y.float()
            feature, y = feature.to(device), y.to(device)

            # Data to model
            out = model(feature)

            # Calculate loss, do backpropagation and step optimizer
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save loss
            loss_np = loss.detach().cpu()
            loss_np = loss_np.numpy()
            total_loss.append(loss_np)
            epoch_loss.append(loss_np)
            wandb.log({"loss":loss})
    #        wandb.watch(model)
            # Print loss
            print(f"Epoch {epoch_nr}, batch {i}: loss = {float(loss)}")

        # Save average loss for each epoch
        epoch_loss = np.mean(epoch_loss)
        loss_pr_epoch.append(epoch_loss)
        print(f"Loss for epoch {epoch_nr}: {epoch_loss}")

        # Epoch counter
        epoch_nr += 1

        # Save model at 100 and 200 epochs
        if epoch == 30:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_nr,
                'loss': loss,
                'total_loss': total_loss,
                'loss_pr_epoch': loss_pr_epoch
            }, f"model_hpc_ba{BATCH_SIZE}-lr{LEARNING_RATE}-ep30of{NUM_EPOCHS}.pth.tar")
        if epoch == 60:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_nr,
                'loss': loss,
                'total_loss': total_loss,
                'loss_pr_epoch': loss_pr_epoch
            }, f"model_hpc_ba{BATCH_SIZE}-lr{LEARNING_RATE}-ep60of{NUM_EPOCHS}.pth.tar")

    # Save the model in the end
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch_nr,
        'loss': loss,
        'total_loss': total_loss,
        'loss_pr_epoch': loss_pr_epoch
    }, SAVEPATH)

    # Save array of loss
    loss_arr = np.array(total_loss)
    np.save(LOSS_PATH, loss_arr)

    return

if __name__ == "__main__":
    training()

    print("-" * 40)
    print("All Done")

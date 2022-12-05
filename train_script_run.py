import os
from torchvision.io import read_image, ImageReadMode
import numpy as np
from structure_tensor import eig_special_2d, structure_tensor_2d, eig_special_3d, structure_tensor_3d
import torch.nn.functional as F
import skimage.io
import matplotlib.pyplot as plt
import scipy as scp
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch.optim
import wandb
import numpy as np
import torch
import torch.nn as nn
import os
from torch.nn.functional import normalize
import time

torch.cuda.empty_cache()
wandb.init(project = "DLProject", name = f"lr-{os.environ['LEARNING_RATE']}-ep-{os.environ['EPOCHS']}-bs-{os.environ['BATCH_SIZE']}-ss-{os.environ['SCALE_SIZE']}", entity="02456_project")
wandb.config = {
	"learning_rate":float(os.environ['LEARNING_RATE']),
	"epochs":int(os.environ['EPOCHS']),
	"batch_size":int(os.environ['BATCH_SIZE']),
"scale_size":float(os.environ['SCALE_SIZE'])

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
        start_overall = time.time()
        scale = float(os.environ['SCALE_SIZE'])
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

        start_orig_st = time.time()
        S = structure_tensor_2d(img, sigma, rho)
        val, y = eig_special_2d(S)
        # vec = torch.from_numpy(vec).permute(1,2,0)
        end_orig_st = time.time()
        structure_tensor_time = end_orig_st - start_orig_st


        img = np.reshape(img, (1, 400, 400))
        img = torch.from_numpy(img)
        img = normalize(img)
        input_img = torch.cat((img, transformed_structure_tensor), 0)
        end_overall = time.time()
        overall_time = end_overall - start_overall

        wandb.log({"overall_transform_time": overall_time})
        wandb.log({"transform_time_exOrig": overall_time - structure_tensor_time})
        return input_img, y

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
SCALE_SIZE = wandb.config['scale_size']
Drop_P = 0.2

LOAD_MODEL = True
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

    test_split = 0.2
    validation_split = 0.1
    shuffle_dataset = False
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(transformed_dataset)
    indices = list(range(800))
    train_test_split = int(np.floor(test_split * dataset_size))
    train_val_split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[train_test_split:], indices[:train_test_split]
    np.random.shuffle(train_indices)
    val_indices = indices[:train_val_split]
    

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_indices = list(range(800))
    test_sampler = SequentialSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=BATCH_SIZE, 
                                                    sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=1,
                                                    sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=1,
                                                    sampler=test_sampler)

    # create folder to store outputs
    store_folder = f"{os.getcwd()}/ba{BATCH_SIZE}-lr{LEARNING_RATE}-ep{NUM_EPOCHS}-ss{SCALE_SIZE}"
    if not os.path.exists(store_folder):
        os.makedirs(store_folder)

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

    # Training loop
    for epoch in range(NUM_EPOCHS):
        break
        # Training mode initialized
        model.train()

        epoch_loss = []

        # training
        for i, (feature, y) in enumerate(train_loader):
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
            # Print loss
            print(f"Epoch {epoch_nr}, batch {i}: loss = {float(loss)}")

        # Save average loss for each epoch
        epoch_loss = np.mean(epoch_loss)
        loss_pr_epoch.append(epoch_loss)
        print(f"Loss for epoch {epoch_nr}: {epoch_loss}")

        # Epoch counter
        epoch_nr += 1

        # Save model at 30 and 60 epochs
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
            }, f"model_hpc_ba{BATCH_SIZE}-lr{LEARNING_RATE}-ep60of{NUM_EPOCHS}.pth.tar")

        # test on validation set
        val_losses = []
        if epoch % 1 == 0:
            model.eval()
            
            for i, (val_feature, val_y) in enumerate(validation_loader):
                # Dataload. Send to device.
                val_feature, val_y = val_feature.float(), val_y.float()
                val_feature, val_y = val_feature.to(device), val_y.to(device)

                # Data to model
                val_out = model(val_feature)

                # Calculate loss, do backpropagation and step optimizer
                val_loss = loss_func(val_out, val_y)

                # Save loss
                val_loss_np = val_loss.detach().cpu()
                val_loss_np = val_loss_np.numpy()
                val_losses.append(val_loss_np)
                wandb.log({"val_loss":val_loss_np})

            # Save average loss for each epoch
            mean_val_loss = np.mean(val_losses)
            print(f"Mean loss for validation {mean_val_loss}")
    
    # Testing 
    model.eval()
    test_losses= []
    cnt = 0
    y_out = []
    for i, (test_feature, test_y) in enumerate(test_loader):
        # Dataload. Send to device.
        test_feature, test_y = test_feature.float(), test_y.float()
        test_feature, test_y = test_feature.to(device), test_y.to(device)
        
        start = time.time()

        # Data to model
        test_out = model(test_feature)

        end = time.time()
        wandb.log({"testDataThroughModel_time": end-start})

        # Calculate loss, do backpropagation and step optimizer
        test_loss = loss_func(test_out, test_y)

        # Save loss
        test_loss_np = test_loss.detach().cpu()
        test_loss_np = test_loss_np.numpy()
        test_losses.append(test_loss_np)
        wandb.log({"test_loss": test_loss_np})
        y_out.append(test_out.detach().cpu().numpy())
        #if cnt % 10 == 0:
        #torch.save(test_out, f"{store_folder}/{i}_out.pt")
        #torch.save(test_y, f"{store_folder}/{i}_y.pt")

        cnt += 1
    np.save("yout", y_out)
    # Save average loss for each epoch
    mean_test_loss = np.mean(test_losses)
    print(f"Mean loss for validation {mean_test_loss}")
    

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

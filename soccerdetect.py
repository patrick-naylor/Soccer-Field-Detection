import torch
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import interpolate


class imageDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index]
        X = self.transform(image)
        return [X]

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5, padding='same')
        self.fc1 = nn.Linear(int(12*72*128*.25), int(4*72*128*.25))
        self.fc2 = nn.Linear(int(4*72*128*.25), int(2*72*128*.25))       
        self.fc3 = nn.Linear(int(2*72*128*.25), int(1*72*128*.25))
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Model()

def field_mask(image, model_path, pad=0):
    '''
        Applies CNN model to mask images of soccer fields to only include
        the playing field.

        moodel_path is path to pretrained CNN model.

        pad is a buffer zone applied to mask to allow for more confidence. 
        pad works by zooming mask image then expanding to original image size.
    '''
    with torch.no_grad():
        #Shring image to model input size
        image_shrunk = cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA)
        image_shrunk = np.expand_dims(image_shrunk, 0)
        #Convert image to pytoch DataLoader
        image_ds = imageDataset(X=image_shrunk)
        image_dl = DataLoader(image_ds, 1)
        #Load Model
        model.load_state_dict(torch.load(model_path))
        #Generate mask
        for input in image_dl:
            mask = model(input[0])
        mask = mask.reshape(36, 64)
        mask_detach = mask.numpy()
        #Add desired padding to mask and expand to original size
        if pad != 0:
            mask_detach = mask_detach[pad:-pad, pad:-pad]
        x = np.linspace(0, 1, mask_detach.shape[0])
        y = np.linspace(0, 1, mask_detach.shape[1])
        f = interpolate.interp2d(y, x, mask_detach, kind='linear')
        x2 = np.linspace(0, 1, 720)
        y2 = np.linspace(0, 1, 1280)
        mask_expand = f(y2, x2)
        #Make binary mask
        mask_expand[mask_expand>.5] = 1
        mask_expand[mask_expand<=.5] = 0
        #Apply mask to image
        masked_image = np.zeros((720, 1280, 3))
        for i in range(3):
            masked_image[:, :, i] = mask_expand * image[:, :, i]

        return masked_image

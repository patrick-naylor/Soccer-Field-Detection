import torch
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import glob
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


model_path = '/Users/patricknaylor/Desktop/Field_Detection/data/Models/model_latest.pth'
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

def field_mask(image):
    with torch.no_grad():
        image_shrunk = cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA)
        print(image_shrunk.shape)
        image_shrunk = np.expand_dims(image_shrunk, 0)
        image_ds = imageDataset(X=image_shrunk)
        image_dl = DataLoader(image_ds, 1)
        model.load_state_dict(torch.load(model_path))
        for input in image_dl:
            mask = model(input[0])
        mask = mask.reshape(36, 64)
        mask_detach = mask.numpy()
        x = np.linspace(0, 1, mask_detach.shape[0])
        y = np.linspace(0, 1, mask_detach.shape[1])
        f = interpolate.interp2d(y, x, mask_detach, kind='linear')

        x2 = np.linspace(0, 1, 720)
        y2 = np.linspace(0, 1, 1280)
        mask_expand = f(y2, x2)
        mask_expand[mask_expand>.5] = 1
        mask_expand[mask_expand<=.5] = 0
        print(mask_expand)
        masked_image = np.zeros((720, 1280, 3))
        for i in range(3):
            print(mask_expand.shape, image[:, :, i].shape)
            masked_image[:, :, i] = mask_expand * image[:, :, i]
            print(masked_image[:,:,i])

        return masked_image

#TESTING
image_path = '/Users/patricknaylor/Desktop/Field_Detection/Images/Masked/'
images = list(glob.glob(image_path + '*.jpg'))
test_image_path = images[15]
test_image = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)
masked_image = field_mask(test_image)
print(masked_image)
plt.imshow(masked_image[:,:,:].astype('int'))
plt.colorbar()
plt.show()
#TESTINGJ
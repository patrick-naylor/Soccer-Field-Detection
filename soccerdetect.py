import torch
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import glob


model_path = 'model_latest.pth'
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
    image_shrunk = cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA)
    im_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image_shrunk = im_transforms(image_shrunk)
    model.load_state_dict(torch.load(model_path))
    mask = model(image_shrunk)
    mask = mask.reshapel(36, 64)
    print(mask.size(), image_shrunk.size())

#TESTING
image_path = '/Users/patricknaylor/Desktop/Field_Detection/Images/Masked/'
images = list(glob.glob(image_path + '*.jpg'))
test_image_path = images[0]
test_image = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)
field_mask(test_image)
#TESTINGJ
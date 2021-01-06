import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(3, 16, 2, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 2, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 2, padding=1)

    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(64*7*7, 1024)
    self.fc2 = nn.Linear(1024, 2)

    self.dropout = nn.Dropout(0.25)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))

    # flatten image input
    x = x.view(-1, 64*7*7)
    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x


model = Net()


PATH = 'cnn_model_100_adam.pt'
model.load_state_dict(torch.load(PATH))
model.eval()


def transform_image(image_bytes):
  transform = transforms.Compose([transforms.Resize(50),
                                transforms.CenterCrop(50),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

  image = Image.open(io.BytesIO(image_bytes))
  return transform(image).unsqueeze_(0)


def get_prediction(image_tensor):
  output = model(image_tensor)
  _, pred = torch.max(output.data, 1)
  return pred

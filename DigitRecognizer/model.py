import torch
import torch.nn as nn
import torchvision.transforms as T
import os

torch.set_printoptions(threshold=10_000)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features_extract = nn.Sequential(
            nn.Conv2d(kernel_size=5, in_channels=1, out_channels=6),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(kernel_size=5, in_channels=6, out_channels=16),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.features_extract(x)
        x = torch.flatten(x, 1)
        prob = nn.functional.softmax(self.classifier(x), dim=1)

        return prob


# init
model = CNN()


def build():
    path = os.path.join(os.path.dirname(__file__), "best.pt")
    model.load_state_dict(torch.load(path))
    model.eval()


def predict(img):
    # model accept 1x32x32
    # canvas return shape: 32x32x4 (r, g, b, a)

    img = (T.ToTensor())(img)
    img = torch.narrow(img, 0, 0, 3)
    img = (T.Compose([
        T.Resize((32, 32)),
        T.Grayscale()
    ]))(img)

    if img.sum().item() == 0:
        return {x: 0 for x in range(10)}

    # img = nn.functional.avg_pool2d(img, (10, 10))
    img = img.unsqueeze(0)

    with torch.no_grad():
        probs = model(img)
        output = {
            i: probs[0][i].item() for i in range(probs.size(1))
        }
        return output

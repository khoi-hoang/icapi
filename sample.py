# From ML guy with <3

import PIL.Image
import torch
import torchvision
from torch import Tensor
from torchvision import transforms

# Load the pretrain model.
model = torchvision.models.resnet18(pretrained=True).eval()

# Create the image's preprocessor
preprocessor = preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Let's guess the name of this cute bird
image = PIL.Image.open('./static/sample_images/bird.jpg')

# Doing some Deep Learning magic, no need to care about the details
input_tensor = preprocessor(image)  # type: Tensor

input_batch = input_tensor.unsqueeze(0)  # type: Tensor

with torch.no_grad():   # Disable grad to speed up the inference performance
    output = model(input_batch)  # type: Tensor

index = torch.argmax(output)  # type: Tensor

with open('./static/labels/resnet.txt', "r") as f:
    labels = [s.strip() for s in f.readlines()]

print(labels[index])
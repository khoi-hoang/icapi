import logging
from abc import abstractmethod

import torch
import torchvision.models
from PIL.JpegImagePlugin import JpegImageFile
from torch import Tensor
from torchvision import transforms

from domain.services.labeler import Labeler


class ImageClassification:
    @abstractmethod
    def classify_jpeg(self, input_image: JpegImageFile) -> str:
        """
        Takes an image and returns the best guess of what it is.
        :param input_image:
        :return: Something like 'chair', 'table', 'golden retriever', 'AK-47'...
        """
        raise NotImplementedError


class ResNetImageClassification(ImageClassification):
    def __init__(self, labeler: Labeler) -> None:
        self.__labeler = labeler  # To annotate the result (from index to English for us mere mortals).
        self.__model = torchvision.models.resnet18(pretrained=True)  # Obtain the pretrained model.
        self.__model.eval()  # Put it in evaluation mode, don't know why, just how it works internally.

        # Every model expects a certain input, we must preprocess it into normalized tensors.
        # https://pytorch.org/hub/pytorch_vision_resnet/
        self.__preprocessor = preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify_jpeg(self, input_image: JpegImageFile) -> str:
        input_tensor = self.__preprocessor(input_image)  # type: Tensor
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        with torch.no_grad():
            output = self.__model(input_batch)  # Feed the batch to the model and get the output!

        index = torch.argmax(output)  # type: Tensor
        return self.__labeler.get_label(index.item())

from abc import abstractmethod

import torch
from PIL.JpegImagePlugin import JpegImageFile
from torch import Tensor
from torch.nn import Module
from torchvision import transforms

from domain.services.labeler import Labeler


class ImageClassifier:
    @abstractmethod
    def classify_jpeg(self, input_image: JpegImageFile) -> str:
        """
        Takes an image and returns the best guess of what it is.
        :param input_image:
        :return: Something like 'chair', 'table', 'golden retriever', 'AK-47'...
        """
        raise NotImplementedError


class ResNetImageClassifier(ImageClassifier):
    def __init__(
            self,
            model: Module,
            preprocessor: transforms.Compose,
            labeler: Labeler
    ) -> None:
        self.__model = model
        self.__preprocessor = preprocessor
        self.__labeler = labeler

    def classify_jpeg(self, input_image: JpegImageFile) -> str:
        input_tensor = self.__preprocessor(input_image)  # type: Tensor
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        with torch.no_grad():
            output = self.__model(input_batch)  # Feed the batch to the model and get the output!

        index = torch.argmax(output)  # type: Tensor
        return self.__labeler.get_label(index.item())

from abc import abstractmethod
import os


class Labeler:
    @abstractmethod
    def get_label(self, index: int) -> str:
        """
        :return: The corresponding label given the index of a model.
        """
        raise NotImplementedError


class ResNetLabeler(Labeler):
    SOURCE_PATH = '../../static/labels/resnet.txt'

    def __init__(self) -> None:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, self.SOURCE_PATH)

        with open(filename, "r") as f:
            self.labels = [s.strip() for s in f.readlines()]

    def get_label(self, index: int) -> str:
        return self.labels[index]

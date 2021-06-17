import torchvision
from dependency_injector import containers
from dependency_injector.providers import Singleton, Object
from torchvision import transforms

from domain.services.image_classifier import ResNetImageClassifier
from domain.services.labeler import ResNetLabeler


class ResnetContainer(containers.DeclarativeContainer):
    # Preprocessor
    preprocessor = Object(transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    # Model
    model = Object(torchvision.models.resnet18(pretrained=True).eval())

    # Labeler
    labeler = Singleton(ResNetLabeler)

    # Classifier
    image_classifier = Singleton(
        ResNetImageClassifier,
        model=model,
        preprocessor=preprocessor,
        labeler=labeler
    )

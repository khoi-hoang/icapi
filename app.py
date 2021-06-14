from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.datastructures import FileStorage

from application.flask.model.flask_response import ClassificationResponse
from domain.services.image_classification import ResNetImageClassification
from domain.services.labeler import ResNetLabeler

app = Flask(__name__)

resnet156_labeler = ResNetLabeler()
ic = ResNetImageClassification(resnet156_labeler)


@app.route('/classify/resnet', methods=['POST'])
def classify_resnet18():
    input_raw_jpeg = request.files['file']  # type: FileStorage
    input_pillow_img = Image.open(input_raw_jpeg)
    result = ic.classify_jpeg(input_pillow_img)

    return jsonify(ClassificationResponse(result).to_dict())


if __name__ == '__main__':
    app.run()

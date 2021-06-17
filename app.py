from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.datastructures import FileStorage

from application.di.resnet_container import ResnetContainer
from application.flask.model.flask_response import ClassificationResponse

app = Flask(__name__)
resnetContainer = ResnetContainer()
resnet_image_classifier = resnetContainer.image_classifier()


@app.route('/model/resnet', methods=['POST'])
def classify_resnet():
    input_raw_jpeg = request.files['file']  # type: FileStorage
    input_pillow_img = Image.open(input_raw_jpeg)
    result = resnet_image_classifier.classify_jpeg(input_pillow_img)

    return jsonify(ClassificationResponse(result).to_dict())


if __name__ == '__main__':
    app.run()

from abc import abstractmethod


class FlaskResponse:
    @abstractmethod
    def to_dict(self) -> dict:
        """
        Necessary for jsonify to work its magic.
        :return:
        """
        raise NotImplementedError


class ClassificationResponse(FlaskResponse):
    def to_dict(self) -> dict:
        return {
            'result': self.__result
        }

    def __init__(self, result: str) -> None:
        self.__result = result

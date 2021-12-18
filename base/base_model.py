from abc import ABC, abstractmethod


class BaseSentimentModel(ABC):
    """
    A base class for designing the Sentiment Model
    """

    @abstractmethod
    def model_load(self, **kwargs):
        """Function to load the model

        :param kwargs:
        :type kwargs:
        ...
        :raises Exception:
        ...
        :return data: Loaded model
        :rtype data: object
        """
        pass

    @abstractmethod
    def model_predict_data(self, **kwargs):
        """Function to predict on the text data

        :param kwargs:
        :type kwargs:
        ...
        :raises Exception:
        ...
        :return prediction: Value of the prediction
        :rtype prediction: float
        """
        pass


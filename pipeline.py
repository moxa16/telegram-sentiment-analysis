from data.data_process import DataProcess
from model.sa_model import SentimentModel
import pandas as pd


class Pipeline:
    """
    Main Pipeline to run the code
    """

    def __init__(self, **kwargs):
        """Constructor Function

        :param kwargs['file_path']: File path of the json file to load
        :type kwargs['file_path']: str
        :param kwargs['use_checkpoint']: Flag to skip preprocessing and start from the saved data
        :type kwargs['use_checkpoint']: bool

        """
        # class attributes
        self.variables = {
            'file_path': kwargs['file_path'],
            'use_checkpoint': kwargs['use_checkpoint']
        }

        # Components
        self.data_process = DataProcess()
        self.model_pipe = SentimentModel()

    def run(self):
        """Function to run the pipeline
        """

        if self.variables['use_checkpoint'] is False:
            # extracting the data
            json_data = self.data_process.data_extract(
                file_path=self.variables['file_path'], src='json')

            # transforming the data
            transformed_data = self.data_process.data_transform(data=json_data)

            # checkpoint transformed data
            transformed_data = self.data_process.data_load(save_path='assets/transformed_data.csv', save_type='csv',
                                                           data=transformed_data)
        else:
            transformed_data = pd.read_csv('assets/transformed_data.csv')

        # load the model
        self.model_pipe.model_load()

        # do model prediction on the text data
        prediction_data = self.model_pipe.model_predict_data(data=transformed_data)

        # plot image on the prediction data
        self.model_pipe.plot_data(data=prediction_data)

        return True


some_obj = Pipeline(file_path='assets/raw_data.json', use_checkpoint=False)
print(some_obj.run())

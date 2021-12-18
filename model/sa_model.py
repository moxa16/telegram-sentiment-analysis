import pandas as pd
import spacy
from logzero import logger
from base.base_model import BaseSentimentModel
from spacytextblob.spacytextblob import SpacyTextBlob
import plotly.graph_objects as go


class SentimentModel(BaseSentimentModel):
    """
    An implementation and inheriting from base.base_model.BaseSentimentModel
    """
    def __init__(self):
        self.variables = {}

    def model_load(self):
        logger.info('Loading Model...')
        self.variables['nlp'] = spacy.load('en_core_web_sm')
        self.variables['nlp'].add_pipe("spacytextblob")

    def model_predict_data(self, **kwargs):
        logger.info('Predicting on Data...')
        self.variables['df'] = kwargs['data'].copy()
        logger.info('Calculating Polarity...')
        self.variables['df']['polarity'] = self.variables['df']['text'].apply(self.get_polarity)
        logger.info('Categorizing Sentiment...')
        self.variables['df']['sentiment'] = self.variables['df']['polarity'].apply(self.get_sentiment)
        return self.variables['df']

    def get_polarity(self, txt):
        """Function to get the polarity

         :param txt: Text value
         :type txt: str
         ...
         :raises Exception:
         ...
         :return polarity: Polarity value of the string
         :rtype polarity: float
         """
        doc = self.variables['nlp'](txt)
        return doc._.polarity

    @staticmethod
    def get_sentiment(polarity):
        """Function to get the sentiment

         :param polarity: Float value
         :type polarity: float
         ...
         :raises Exception:
         ...
         :return sentiment: Sentiment value as per the bin
         :rtype sentiment: str
         """
        if polarity < 0:
            return 'negative'
        elif polarity == 0:
            return 'neutral'
        elif polarity > 0:
            return 'positive'

    def plot_data(self, **kwargs):
        """Function to plot the data

         :param data: DataFrame containing the predicted values
         :type data: pandas.DataFrame
         ...
         :raises Exception:
         ...
         :return status: Flag stating if plotting was successful
         :rtype status: bool
         """
        logger.info('Plotting on Prediction Data...')
        plot_data = {
            'date': [],
            'num_msgs_per_day': [],
            'avg_sentiment': []
        }
        for grp, sub_df in kwargs['data'].groupby('date'):
            plot_data['date'].append(grp)
            plot_data['num_msgs_per_day'].append(sub_df['text'].shape[0])
            plot_data['avg_sentiment'].append(round(sub_df['polarity'].mean(), 3))

        plot_data = pd.DataFrame(plot_data)
        plot_data.to_csv('assets/plot_data.csv', index=False)

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            y=plot_data['num_msgs_per_day'].to_list(),
            x=plot_data['date'].to_list(),
            orientation='v',
            marker=dict(
                color='rgba(246, 78, 139, 0.6)',
                line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
            )
        ))
        fig1.update_layout(
            title="Telegram Number Messages Per Day",
            xaxis_title="Date",
            yaxis_title="Number of Messages",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        fig1.write_image('assets/num_msgs_per_day.png')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=plot_data['avg_sentiment'].to_list(),
            x=plot_data['date'].to_list()
            ))
        fig2.update_layout(
            title="Average Sentiment Per Day",
            xaxis_title="Date",
            yaxis_title="Average Sentiment",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )

        fig2.write_image('assets/avg_sentiment.png')
        return True



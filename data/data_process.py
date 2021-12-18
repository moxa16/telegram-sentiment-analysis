"""
Python script to implement the Data Process Pipeline
"""
import pandas as pd
from datetime import datetime
from langdetect import detect
import spacy
import string
import emoji
from tqdm import tqdm
from logzero import logger


from base.base_data_process import BaseDataProcess


class DataProcess(BaseDataProcess):
    """
    An Implementation and inheriting from base.base_data_process.BaseDataProcess
    """
    EN = spacy.load('en_core_web_sm')
    STOPWORDS = EN.Defaults.stop_words

    def data_extract(self, **kwargs):
        file_path = kwargs['file_path']
        src = kwargs['src']

        if src == 'json':
            import json
            with open(file_path, 'r',  encoding='utf8') as f_reader:
                json_data = json.load(f_reader)
        return json_data

    def data_transform(self, **kwargs):
        # convert to data frame
        logger.info('convert to data frame...'.title())
        _data = self.data_converter(kwargs['data'])

        # remove unique characters
        logger.info('remove unique characters..'.title())
        _data = self.remove_unique_char(_data)

        # filter data as per string length
        logger.info('filter data as per string length...'.title())
        _data = self.filter_as_per_len(_data)

        # remove non-english words
        logger.info('remove non-english words...'.title())
        _data = self.remove_non_eng(_data)

        # # make texts lower case
        logger.info('make texts lower case...'.title())
        _data['text'] = _data['text'].str.lower()

        _data = pd.read_csv('assets/isshibdogechkpt.csv')

        # filter 'shib' and 'doge'
        logger.info("filter in 'shib' and 'doge' only...".title())
        _data = self.filter_data(_data)

        # remove stopwords
        logger.info("remove stopwords...".title())
        _data['text'] = _data['text'].apply(self.remove_stop_words)

        # remove punctuations
        logger.info("remove punctuations...".title())
        _data['text'] = _data['text'].apply(self.remove_punctuations)

        # filter data as per string length
        logger.info('filter data as per string length...'.title())
        _data = self.filter_as_per_len(_data)

        return _data

    def data_load(self, **kwargs):
        """Function to save the dataframe

         :param kwargs['save_path']: File path to save the data
         :type kwargs['save_path']: str
         :param kwargs['data']: DataFrame to save
         :type kwargs['data']: pandas.DataFrame
         :param kwargs['save_type']: Type of file to save
         :type kwargs['save_type']: str
         ...
         :raises Exception:
         ...
         :return data: Transformed data, as is
         :rtype data: pandas.DataFrame
         """
        if kwargs['save_type'] == 'csv':
            kwargs['data'].to_csv(kwargs['save_path'], index=False)
        return kwargs['data']

    def data_converter(self, json_data):
        """Function to convert json data to pandas data frame

        :param json_data: Loaded json data
        :type json_data: dict
        ...
        :raises Exception:
        ...
        :return data: Converted data
        :rtype data: pandas.DataFrame
        """
        _data = {
            'text': [],
            'date': []
        }
        for msg in tqdm(json_data['messages'], desc="Converting Data"):
            _data['date'].append(self.parse_date(msg['date'].replace('T', ' ').split(" ")[0]))
            _data['text'].append(msg['text'])

        _data = pd.DataFrame(_data)
        return _data

    @staticmethod
    def filter_as_per_len(data):
        """Function to filter as per length

         :param data: DataFrame containing the text
         :type data: pandas.DataFrame
         ...
         :raises Exception:
         ...
         :return data: Filtered DataFrame
         :rtype data: pandas.DataFrame
        """
        data['txt_len'] = data['text'].apply(lambda x: len(str(x).split()))
        data = data[data['txt_len'] >= 3]
        return data

    @staticmethod
    def filter_data(data):
        """Function to remove the punctuations

         :param data: Dataframe containing the string value
         :type data: pandas.DataFrame
         ...
         :raises Exception:
         ...
         :return data: Filtered data
         :rtype data: pandas.DataFrame
         """

        def is_shib_doge(txt):
            if 'shib' in str(txt) or 'doge' in str(txt):
                return True
            return False

        data['isShibDoge'] = data['text'].apply(is_shib_doge)
        data = data[data['isShibDoge'] == True]
        return data

    @staticmethod
    def parse_date(text, date_format='%Y-%M-%d'):
        """Function to parse date from the given text

        :param text: Text value
        :type text: str
        :param date_format: Format of the date, defaults to '%Y-%M-%d'
        :type date_format: str (Optional)
        ...
        :raises Exception:
        ...
        :return formatted_date: Format Stripped datatime value
        :rtype formatted_date: datetime.datetime
        """
        return datetime.strptime(text, date_format)

    @staticmethod
    def remove_non_eng(data):
        """Function to remove the non english text

         :param data: Data Frame to work on
         :type data: pandas.DataFrame
         ...
         :raises Exception:
         ...
         :return data: Filtered data with non english texts removed
         :rtype data: pandas.DataFrame
         """
        tqdm_obj = tqdm(total=len(data['text']), desc='Remove Non-English Words')

        def is_english(txt):
            tqdm_obj.update(1)
            try:
                val = detect(txt)
                return True if val == 'en' else False
            except Exception:
                return False

        data['text'] = data['text'].astype(str)
        data['isEnglish'] = data['text'].apply(is_english)
        data = data[data['isEnglish'] == True]
        tqdm_obj.close()
        return data

    def remove_stop_words(self, txt):
        """Function to remove the stopwords

         :param txt: Text value to remove the stopwords from
         :type txt: str
         ...
         :raises Exception:
         ...
         :return _str: Filtered string value
         :rtype _str: str
         """
        _str = []
        for token in txt.split():
            if token in self.STOPWORDS:
                _str.append(token)
        return ' '.join(_str)

    @staticmethod
    def remove_punctuations(txt):
        """Function to remove the punctuations

         :param txt: Text value to remove the stopwords from
         :type txt: str
         ...
         :raises Exception:
         ...
         :return _str: Filtered string value
         :rtype _str: str
         """
        no_punctuation_str = txt.translate(str.maketrans('', '', string.punctuation))
        return no_punctuation_str

    @staticmethod
    def remove_unique_char(data):
        """Function to remove the unique characters

         :param data: DataFrame containing the text
         :type data: pandas.DataFrame
         ...
         :raises Exception:
         ...
         :return data: Filtered DataFrame
         :rtype data: pandas.DataFrame
        """
        # remove emoji code
        def remove_emoji(txt):
            tmp = []
            if isinstance(txt, str):
                for token in txt.split():
                    if token not in emoji.UNICODE_EMOJI:
                        tmp.append(token)
            elif isinstance(txt, list):
                for elem in txt:
                    if isinstance(elem, str):
                        for token in elem.split():
                            if token not in emoji.UNICODE_EMOJI:
                                tmp.append(token)
                    elif isinstance(elem, dict):
                        for token in elem['text'].split():
                            if token not in emoji.UNICODE_EMOJI:
                                tmp.append(token)
                    else:
                        raise ValueError
            elif isinstance(txt, dict):
                if 'text' in txt:
                    for token in txt['text'].split():
                        if token not in emoji.UNICODE_EMOJI:
                            tmp.append(token)
            else:
                print('Skipped value {}'.format(txt))

            return ' '.join(tmp)

        data['text'] = data['text'].apply(remove_emoji)
        return data

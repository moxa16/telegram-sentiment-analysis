from abc import ABC, abstractmethod


class BaseDataProcess(ABC):
    """
    A base class to process data as per the logic
    """

    @abstractmethod
    def data_extract(self, **kwargs):
        """Function to extract the data

        :param kwargs['file_path']: File path to load data from
        :type kwargs['file_path']: str
        :param kwargs['src']: Source from the data is loaded
        :type kwargs['src']: str
        ...
        :raises Exception:
        ...
        :return data: Loaded data
        :rtype data: dict
        """
        pass

    @abstractmethod
    def data_transform(self, **kwargs):
        """Function to transform the data

        :param kwargs:
        :type kwargs:
        ...
        :raises Exception:
        ...
        :return data: Dataframe containing the loaded data
        :rtype data: pandas.DataFrame
        """
        pass

    @abstractmethod
    def data_load(self, **kwargs):
        """Function to load the data to SQL / Server

        :param kwargs:
        :type kwargs:
        ...
        :raises Exception:
        ...
        :return data: Dataframe containing the loaded data
        :rtype data: pandas.DataFrame
        """
        pass

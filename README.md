# MINDS - Sentiment Analysis on Telegram Messages
This project performs sentiment analysis on the messages of the telegram channel: from https://t.me/CryptoComOfficial dated from May 1 to May 15, 2021. 

## Sub-Tasks
1. Pre-processing of the Data
2. Computation of sentiment of each message
3. Plotting the number of messages per day and average sentiment per day

## Requirements
1. python==3.7
2. nltk==3.6.5
3. numpy==1.21.4
4. pandas==1.3.5
5. plotly==5.4.0
6. spacy==3.2.1
7. tqdm==4.62.3

## Workflow Diagram of the project
![Workflow Diagram](https://github.com/moxa16/telegram-sentiment-analysis/blob/main/assets/workflow_diagram.PNG)

## Steps to Execute the project
1. Clone this repository
2. Install requirements:

    ```python
    pip install -r requirements.txt
    ```
3. To run the project:

   ```python
   python3 pipeline.py
   ```

On execution of the project, the progress of the execution will be visible in the following manner:

![Execution Progress Report](https://github.com/moxa16/telegram-sentiment-analysis/blob/main/assets/execution_progress_report.PNG)

## Output

We create the following plots after performing sentiment analysis on the data:

1. Number of messages per day

![Number of Messages Per Day](https://github.com/moxa16/telegram-sentiment-analysis/blob/main/assets/number_of_messages_per_day.PNG)

2. Average sentiment per day

![Average Sentiment Per Day](https://github.com/moxa16/telegram-sentiment-analysis/blob/main/assets/average_sentiment_per_day.PNG)


# Description of folders and files in the repository
```
├──  assets
│    └── raw_data.json  - This contains the raw data directly scrapped from Telegram channel.
│
│
├──  base  
│    └── base_data_process.py  - This contains the template for class in data pipeline.
│    └── base_model.py  - This contains the template for class in model pipeline.
│ 
│
├──  data  
│    └── data_process.py  - This contains functions to perform Extract, Transform and Load operations on the data. 
│
│
├──  model
│    └── sa_model.py  - This contains functions to perform loading the model and prediction of sentiment 
│
│
├── pipeline.py - This contains the main function that sequentially executes all the components of the project.
│
│
├── requirements.txt 
```

## Reasons for choosing Spacy for performing sentiment analysis 
Spacy is a mature and batteries-included framework that comes with prebuilt models for common NLP tasks like classification, named entity recognition, and part-of-speech tagging. It’s very easy to train a model with your data: all the gritty details like tokenization and word embeddings are handled for you. SpaCy is written in Cython which makes it faster than a pure Python implementation, so it’s ideal for production. SpaCy can update itself to use the improved model, and the user doesn’t need to change anything. This is good for getting a model up and running quickly

The major drawback of Spacy is that it is not customizable and the project at hand did not require customization. Hence, Spacy is the best choice for performing the sentiment analysis.

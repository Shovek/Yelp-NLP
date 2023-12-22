# Dependencies: transformers, pytorch, beautifulsoup, pandas, numpy, requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModel
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np

# Model repo
repo = "bert-base-multilingual-uncased-sentiment" 

# Download pytorch model
model = AutoModelForSequenceClassification.from_pretrained(repo)
tokenizer = AutoTokenizer.from_pretrained(repo)

# Begin prompt for user input
print('See how other people feel about your favorite locations on Yelp!\nWe will take their reviews and rate the location from 1 to 5!')

# Grab YELP reviews via regexing 
user_input = input('Enter yelp link to your favorite establishment:\t')
r = requests.get(user_input)
soup = BeautifulSoup(r.text, "html.parser")
regex = re.compile(".*comment.*")
results = soup.find_all('p', {"class" : regex})
reviews = [result.text for result in results]


# Load reviews into a dataframe
dataframe = pd.DataFrame(np.array(reviews), columns = ['review'])


def sentiment_score(review):
  # Transform input tokens 
  tokens = tokenizer.encode(review, return_tensors="pt")
  result = model(tokens)
  
  # Sentiments are scored 1-5, and output is a list of probabilities for each sentiment value
  # so we grab the highest probability value and increase by 1 since lists are zero-indexed
  return int(torch.argmax(result.logits)) + 1

# Grab first 512 reviews and create a column for sentiment then calculate sentiment scores for a final score
dataframe['sentiment'] = dataframe['review'].apply(lambda x: sentiment_score(x[:512]))
final_score = dataframe['sentiment'].median()

print(final_score)

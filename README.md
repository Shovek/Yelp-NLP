# Yelp-NLP
A simple command line interface program for sentiment analysis on places listed on yelp

## How to run locally
1. It is necessary to have the following python dependencies installed:
- transformers
- pytorch
- beautifulsoup
- pandas
- numpy
- requests

2. Ensure that main.py and the folder containing the pre-trained model are in the same directory
3. Run the following command:  *python main.py*
> You will recieve a prompt to enter a yelp link for any place listed on yelp,
> paste the yelp url into the command line and hit return

4. A value will be output which is the resultant sentiment score based on the reviews left by customers
> 1 is the lowest score while 5 is the highest


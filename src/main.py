from preprocessing.bag_of_words import create_csv
from sentiment.gi_dictionary import get_sentiment_for

def main():
    filename = create_csv()
    sentiment_file = get_sentiment_for(filename)



if __name__ == '__main__':
    main()

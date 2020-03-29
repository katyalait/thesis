# Create a bag of words for the documents
# Extract the documents from txt file
# Dictionary of documents with dates, headlines, content
import logging
import os
from nltk.stem import WordNetLemmatizer
import copy
import argparse
from lexis_nexis_parser import ReadInputFile
import pandas as pd
import numpy as np
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from dateutil import parser
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
DEBUG=True
LOGGER = logging.getLogger('ReadInputFile')
INPUT_FILE = os.path.join(BASE, "results/parser/small_doc.txt")
OUTPUT_FILE = os.path.join(BASE, "results/bag_of_words/article_contents.csv")
STOP_WORDS = "let"

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

class ArgParser:
    def __init__(self):
        LOGGER.debug('Init ArgsParser')

    def parse_arguments(self):
        LOGGER.debug('Parsing the arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input", "-i", help="Input file to parse", required=True)
        parser.add_argument(
            "--output", "-o", help="Output csv file to which result is written", required=False)
        parser.add_argument(
            '--debug', '-d', help="Enable Debug (** Will flood the screen **)", action='store_true', required=False)
        args = parser.parse_args()
        return args

def printDebug(string):
	if DEBUG:
		print(string)

def create_token_vectors(documents):
    count = len(documents)
    index = 0
    stop_words = set(stopwords.words('english'))
    stop_words.add(STOP_WORDS)
    lemmatizer = WordNetLemmatizer()
    for doc in documents:
        index += 1
        progress(index, count, status="Tokenizing doc {}/{}".format(index, count))
        frequency_map = {}
        content = doc['content']
        sentences = sent_tokenize(content)
        for sentence in sentences:
            if len(sentence) < 1:
                continue
            word_tokens = word_tokenize(sentence)
            for word in word_tokens:
                if not word in stop_words and word.isalpha():
                    word = word.lower()
                    lemmatized_word = lemmatizer.lemmatize(word)
                    if not lemmatized_word in frequency_map:
                        frequency_map[lemmatized_word] = 1
                    else:
                        frequency_map[lemmatized_word] += 1
        doc['frequency_map'] = frequency_map
    return documents

def extract_value(data):
    if ':' in data:
        return data.split(":", 1)[1].strip()
    else:
        return data

def parse_date(data):
    value = extract_value(data)
    split_date = value.split(",")
    if len(split_date) >= 2:
        if "GENERAL_EDIT_CORRECTION ERROR" in split_date[1]:
            split_date[1] = split_date[1].split("GENERAL_EDIT_CORRECTION ERROR")[0]
        return parser.parse(split_date[0] + ", " + split_date[1])
    else:
        return split_date[0]

def parse_text_file(file):
    documents = []
    content = False
    counter = 0
    body = ""
    text_template = {
        'headline': '',
        'date': '',
        'source': '',
        'length': '',
        'content': '',
        'country': '',
        'publication_type': '',
        }
    document = copy.deepcopy(text_template)
    for line in file.split('\n'):
        temp_line = line.strip()
        if not temp_line  == '':
            counter += 1
            if temp_line.startswith("PUBLICATION-NAME:"):
                document['source'] = extract_value(temp_line)
                continue
            if "PUBLICATION-WRITTEN-DATE" in temp_line:
                date = parse_date(temp_line)
                printDebug(date)
                document['date'] = date
                continue
            if temp_line.startswith('PUBLICATION-HEADLINE:'):
                document['headline'] = extract_value(temp_line)
                continue
            if temp_line.startswith('LENGTH:'):
                document['length'] = extract_value(temp_line)
                continue
            if counter == 10:
                printDebug('content-start')
                body = ''
                content = True
            if temp_line.startswith("LOAD-DATE:"):
                document['content'] = body
                content = False
                continue
            if temp_line.startswith("PUBLICATION-SOURCE-COUNTRY:"):
                document['country'] = extract_value(temp_line)
                continue
            if temp_line.startswith("PUBLICATION-TYPE:"):
                document['publication_type'] = extract_value(temp_line)
                documents.append(document)
                document = copy.deepcopy(text_template)
                counter = 0
                continue
            if content:
                body += temp_line + "\n"
                continue
    return documents

def parse_cli():
    global INPUT_FILE
    global OUTPUT_FILE
    global DEBUG
    arg_parser = ArgParser()
    args = arg_parser.parse_arguments()
    DEBUG = args.debug
    if args.input:
        INPUT_FILE=args.input
        print(f'Input file is set to {INPUT_FILE}')
    if args.output:
        OUTPUT_FILE = args.output
        print(f'Output File set to {OUTPUT_FILE}')

def create_csv(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    file = open(input_file, "r").read()
    documents = parse_text_file(file)
    printDebug("Document dictionary length is " + str(len(documents)))

    documents = create_token_vectors(documents)
    # generate a csv to store this
    df = pd.DataFrame(documents)
    df.to_csv(output_file, index=True)
    # label each column with date and headline
    return output_file

def main():
    args = parse_cli()
    print(create_csv(INPUT_FILE, OUTPUT_FILE))

if __name__ == '__main__':
    main()

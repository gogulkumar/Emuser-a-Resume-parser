import numpy as np
import math
import pandas as pd
import re
import nltk
import string
import inspect
import json
from nltk.corpus import stopwords, wordnet, words
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import warnings
from sklearn.model_selection import train_test_split
from spacy.tokens.span import Span

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# spacy import
import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc
import jsonlines
# !pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.0.0/en_core_web_lg-2.0.0.tar.gz
import en_core_web_lg

import urllib.request, json
from spacy import displacy


def get_variables(d):
    '''
    :param d: dictionary of the object created
    :return: list of variables along with their datatype
    '''

    def get_type(x):
        # Prints the type of a variable in a slightly more readable fashion
        x = str(type(x))
        # convert type(string variable) from "class 'str's" to "str"
        g = re.search("\' ([^\']+)\'", x)
        if g:
            x = g.group(1)
        return x

    #
    variables = [(str(k) + ' [' + get_type(v) + ']') for k, v in d.items()]
    if len(variables) == 0:
        variables = ['-']
    return variables


def get_methods(o):
    '''
    :param o: the emuser object
    :return: the list of method present in the object
    '''
    return sorted(
        [name for name, description in inspect.getmembers(o, predicate=inspect.ismethod) if name != '__init__'],
        reverse=True)


def print_variables_and_methods(o):
    print("--------------")
    print("Variables : ")
    for x in get_variables(o.__dict__):
        print("\t", x)
    print("--------------")
    print()
    print("--------------")
    print("Methods : ")
    for x in get_methods(o):
        print("\t", x)
    print("--------------")


def print_help(o):
    print("Fetching the metadata of the class")
    return print_variables_and_methods(o)


class Emuser:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        skill_pattern_path = "skill_pattern.jsonl"
        ruler = self.nlp.add_pipe("entity_ruler")
        ruler.from_disk(skill_pattern_path)
        self.nlp.pipe_names
        self.pos_tag_dictionary = {}
        self.index = 0

    def preprocess(self, messages):
        '''
        Input:
            message: a string containing a message.
        Output:
            preprocessed_message_list: a list of words containing the processed message.

        '''
        processed_message = []
        for message in messages:
            message = re.sub('[^a-zA-Z,.]', ' ', message.replace("\xa0", " ").replace("\n", " ")).replace(","," , ")\
                .replace("/", " / ").replace(".", " . ")
            #message = re.sub('[,/]', ' ', message)
            message = message.lower()
            processed_message.append(message)
        return processed_message

    def pos_tagging(self, text_list):
        '''
        function_purpose : identifies the parts of speech of text
        :param text_list: list of the texts in the dataset
        :return: dictionary that contains words as keys and their respective Parts of speech as value
        '''

        for message in text_list:
            text = self.nlp(message)
            for word in text:
                if word not in self.pos_tag_dictionary:
                    self.pos_tag_dictionary[word.text.strip().lower()] = [word.pos_]
                else:
                    self.pos_tag_dictionary[word.text.strip().lower()].append(word.pos_)

            text = self.nlp(message.lower())
            for word in text:
                if word.text.strip() not in self.pos_tag_dictionary:
                    self.pos_tag_dictionary[word.text.strip()] = [word.pos_]
                else:
                    if word.pos_ not in self.pos_tag_dictionary[word.text.strip()]:
                        self.pos_tag_dictionary[word.text.strip()].append(word.pos_)

            return self.pos_tag_dictionary

    def json_parsing(self):
        with open('skill_pattern.jsonl', 'r') as json_file:
            json_list = list(json_file)
        skill_list = []
        for json_str in json_list:
            result = json.loads(json_str)
            word_list = []
            for skill in result['pattern']:
                try:
                    word_list.append(skill['LOWER'])
                except:
                    word_list.append(skill['TEXT'].lower())
            exact_word = ' '.join(word_list)
            if len(exact_word) > 1 and exact_word != " ":
                skill_list.append(exact_word)
        skill_list = list(set(skill_list))
        return skill_list

    def check_val(self, start, end, char_list):
        for values in char_list:
            if start >= values[0] and end <= values[1]:
                return True
        return False

    def component(self, doc, name, regular_expression):
        if name not in doc.spans:
            doc.spans[name] = []
        ents = list(doc.ents)
        #char_list = [[ents[index].start_char, ents[index].end_char] for index in range(len(ents))]
        for i, match in enumerate(re.finditer(regular_expression, doc.text)):
            start, end = match.span()
            #bool_val = self.check_val(start, end, char_list)
            #if not bool_val:
            try:
                span = doc.char_span(start, end, alignment_mode="expand")
                span_to_add = Span(doc, span.start, span.end, label="SKILL")
                ents.append(span_to_add)
                doc.spans[name].append(span_to_add)
                #char_list.append([start, end])
                doc.ents = ents
            except:
                continue

        # doc.ents = ents
        return doc

    def get_entities(self, text):
        '''
        function_purpose : identifies the entities in the text using spacy package
        :param text: message of the dataset( resume content)
        :return: doc which contains the text along with their entities
        '''
        doc = self.nlp(text)
        ents = []

        for ent in doc.ents:
            if ent.label_ in ["SKILL", "ORG", "PRODUCT"]:
                ents.append(ent)

        doc.ents = ents
        # displacy.render(doc, jupyter = True, style="ent")
        return doc

    def get_skilled_entities(self, message):
        '''
        function_purpose : filters the Skilled entities from the resume text
        :param message: message is the resume context
        :return: list that contains doc and entities associated with that
        '''
        if self.index % 100 == 0:
            print("finding the entities of record between", self.index, " ", min(self.index + 100, self.len_dataset),
                  " ...... please wait")
        doc_entities = self.get_entities(message)
        ents = []

        for ent in doc_entities.ents:
            if ent.text.lower() in self.pos_tag_dictionary:
                if (ent.label_ in ['ORG'] and 'NOUN' in self.pos_tag_dictionary[ent.text.lower()]) and len(
                        self.pos_tag_dictionary[ent.text.lower()]) > 1:
                    # print(ent, ent.label_,pos_tag_dictionary[ent.text.lower()] )
                    continue

            else:
                ents.append(ent)
        for skill in self.skill_list:
            doc_entities = self.component(doc_entities, skill, r'{}'.format(skill))
        #doc_entities.ents = ents
        return doc_entities, doc_entities.ents

    def predict(self, dataset, col_name='Resume_str'):
        '''
        function_purpose : Predicts the entities in the resume content and stores in the dataset
        :param dataset: the resume content dataset
        :param col_name: variable that contains resume content
        :return:
        '''

        self.doc_list = []
        self.ent_list = []
        self.len_dataset = dataset.shape[0]
        self.skill_list = self.json_parsing()
        dataset[col_name] = self.preprocess(dataset[col_name])
        dataset_messages = dataset[col_name]

        self.pos_tag_dictionary = self.pos_tagging(dataset_messages)

        for message in dataset_messages:
            doc, ents = self.get_skilled_entities(message)

            ents = doc.ents
            self.doc_list.append(doc)
            self.ent_list.append(ents)
            self.index = self.index + 1

        dataset['DOC'] = self.doc_list
        dataset['ENTS'] = self.ent_list
        return dataset

    def visualize_entities(self, doc_list=None):
        """
        Function_purpose: visualize the entities of the resume content
        :return: none
        """
        if doc_list is None:
            doc_list = self.doc_list
        for doc in doc_list:
            displacy.render(doc, jupyter=True, style="ent")

    def get_ents(self):
        """

        :return: list of entities in the resume contents
        """
        return self.ent_list

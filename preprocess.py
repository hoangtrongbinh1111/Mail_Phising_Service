import os
# from bs4 import BeautifulSoup
import mailparser
from collections import OrderedDict
import email
from urllib.request import urlretrieve
import tarfile
import shutil
import numpy as np
import glob
import re
import tldextract
import urllib.request
import warnings
import logging
import csv
import sys
import logging
import logging.config
# import yaml
import pandas as pd
from logging import Formatter
from logging.handlers import RotatingFileHandler
# import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
# sns.set_theme()
URLREGEX = r"^(https?|ftp)://[^\s/$.?#].[^\s]*$"
URLREGEX_NOT_ALONE = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
FLASH_LINKED_CONTENT = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F])+).*\.swf"
HREFREGEX = '<a\s*href=[\'|"](.*?)[\'"].*?\s*>'
IPREGEX = r"\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))\b"
MALICIOUS_IP_URL = r"\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\/(www|http|https|ftp))\b"
EMAILREGEX = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
GENERAL_SALUTATION = r'\b(dear|hello|Good|Greetings)(?:\W+\w+){0,6}?\W+(user|customer|seller|buyer|account holder)\b'
MAILS_DIR = 'mails'
phishing_dir = MAILS_DIR + '/phishingMails'
ham_dir_1 = MAILS_DIR + '/hard_ham'
ham_dir_2 = MAILS_DIR + '/easy_ham_2'



def load_mails(dirpath):
    """load emails from the specified directory"""
    files = []
    filepaths = glob.glob(dirpath + '/*')
    for path in filepaths:
        with open(path, 'rb') as f:
            byte_content = f.read()
            str_content = byte_content.decode('utf-8', errors='ignore')
            files.append(str_content)
    # logger.info("Loaded mails from '%s'",dirpath)
    return files


def getMailBody(mail):
    try:
        parsed_mail = mailparser.parse_from_string(mail)
        mail_body = parsed_mail.body.lower()
        subject = parsed_mail.subject
        headers = parsed_mail.headers
        
    except UnicodeDecodeError as Argument:
        parsed_mail = email.message_from_string(mail)
        body = ""
        if parsed_mail.is_multipart():
            for part in parsed_mail.walk():
                # returns a bytes object
                payload = part.get_payload(decode=True)
                strtext = payload.decode()
                body += strtext
        else:
            payload = parsed_mail.get_payload(decode=True)
            strtext = payload.decode()
            body += strtext
        headers = email.parser.HeaderParser().parsestr(mail)
        mail_body = body.lower()
        subject = headers['Subject']
    return [mail_body,subject,headers]

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from  nltk.tokenize import word_tokenize
import nltk 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
# import textdistance
import pdb

  
stop_words = set(stopwords.words('english')) #set of stopwords
 
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

# from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def getMailBody(mail):
    try:
        parsed_mail = mailparser.parse_from_string(mail)
        mail_body = parsed_mail.body.lower()
        subject = parsed_mail.subject
        headers = parsed_mail.headers

        # print (mail_body)
        # pdb.set_trace()
        
    except UnicodeDecodeError as Argument:
        parsed_mail = email.message_from_string(mail)
        body = ""
        if parsed_mail.is_multipart():
            for part in parsed_mail.walk():
                # returns a bytes object
                payload = part.get_payload(decode=True)
                strtext = payload.decode()
                body += strtext
        else:
            payload = parsed_mail.get_payload(decode=True)
            strtext = payload.decode()
            body += strtext
        headers = email.parser.HeaderParser().parsestr(mail)
        mail_body = body.lower()
        subject = headers['Subject']
    return [mail_body,subject,headers]

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext

def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r'',cleaned)
    return  cleaned

def filter_words_mail_body(mails, label):
	body, subject, heaser, labels = [], [], [], []
	for mail in mails:
		filtered_text_body = cleanpunc(cleanhtml(getMailBody(mail)[0]))
		filtered_text_subject = cleanpunc(cleanhtml(getMailBody(mail)[1]))
		# filtered_text_header = cleanpunc(cleanhtml(getMailBody(mail)[2]))
		body.append (filtered_text_body)
		subject.append (filtered_text_subject)
		# header.append (filtered_text_header)
		labels.append (label)
	return body, subject, labels

def load_sample(path):
    message = ""
    with open(path, 'rb') as f:
        byte_content = f.read()
        mail = byte_content.decode('utf-8', errors='ignore')
    #filter mail
    filtered_text_body = cleanpunc(cleanhtml(getMailBody(mail)[0]))
    filtered_text_subject = cleanpunc(cleanhtml(getMailBody(mail)[1]))
    return filtered_text_body, filtered_text_subject


def load_data (data_dir):
    phishing_dir = os.path.join (data_dir, "phishing")
    normal_dir = os.path.join (data_dir, "normal")

    phishing = load_mails(phishing_dir)
    phishing = list(set(phishing))
    normal = load_mails(normal_dir)
    normal = list(set(normal))

    body_1, subject_1, labels_1 = filter_words_mail_body (phishing, "phishing")
    body_2, subject_2, labels_2 = filter_words_mail_body (normal, "normal")
   

    body = body_1 + body_2
    subject = subject_1 + subject_2
    # header = header_1 + header_2
    label = labels_1 + labels_2
    return body, subject, label

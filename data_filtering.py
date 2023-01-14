import pandas as pd 
import numpy as np
import pickle
import re, string, unicodedata
import nltk
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer



def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def preprocess(sample):
    #sample = remove_URL(sample)
    sample = replace_contractions(sample)
    # Tokenize
    words = nltk.word_tokenize(sample)

    # Normalize
    return normalize(words)


def word2vecs():
    embeddings_index = {}
    with open("data/glove.6b.50d.txt") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)

            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print(len(embeddings_index))
    dbfile = open('data/glove_50d.pickle', 'ab')
    pickle.dump(embeddings_index, dbfile)                     
    dbfile.close()


def lobby_subjects():
    lobby_df = pd.read_csv("data/Lobbyist_and_LobbyistEntity_relation_to_subject.csv")

    lobby_df = lobby_df[['Subject', 'Lobbyist']].dropna()

    all_subject_words = []
    lobbyists = []
    lobbyist_subjects = []
    for s, l in zip(lobby_df.Subject, lobby_df.Lobbyist):

        lobbyist = " ".join(preprocess(l))
        lobbyists.append(lobbyist)
        lobby_words = preprocess(s)
        lobbyist_subjects.append(lobby_words)
        
        all_subject_words = all_subject_words + preprocess(s)
    
    all_subject_words = set(all_subject_words)

    vocab_dict = {}
    for i, w in enumerate(all_subject_words):
        vocab_dict[w] = i

    print(len(all_subject_words))
    print(len(lobbyists))
    print(len(lobbyist_subjects))

    lobby_df = pd.DataFrame({"lobbyist":lobbyists, "subject_words":lobbyist_subjects})
    dbfile = open('data/lobby_subjects.pickle', 'ab')
    pickle.dump({"all_subject_words":all_subject_words, "vocab_dict": vocab_dict, 'lobby_df':lobby_df}, dbfile)                     
    dbfile.close()

    return all_subject_words, lobby_df
    
 
def load_contrib_data():
    contrib_df1 = pd.read_csv("data/MNCFB_Contribution_data_2018To2022.csv")
    contrib_df2 = pd.read_csv("data/MNCFB_Contribution_data_2009To2017.csv")

    recipients = []
    contributors = []

    def add_data(df):
        contrib_df = df[['Recipient_reg_num', 'Recipient', 'Contributor']].dropna()
        for r, c in zip(contrib_df.Recipient_reg_num, contrib_df.Contributor):
            recipients.append(r)
            contributors.append(" ".join(preprocess(c)))
        
    add_data(contrib_df2)
    add_data(contrib_df1)
    
    contrib_df = pd.DataFrame({"recipient_id":recipients, "contributor":contributors})
    print(contrib_df.head)


    dbfile = open('data/contributions.pickle', 'ab')
    pickle.dump(contrib_df, dbfile)                     
    dbfile.close()
    print(len(contrib_df))

if __name__ == "__main__":
    #lobby_subjects()
    #dbfile = open('data/lobby_subjects.pickle', 'rb')     
    #db = pickle.load(dbfile)
    
    #word2vecs()
    load_contrib_data()
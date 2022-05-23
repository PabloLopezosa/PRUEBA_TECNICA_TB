import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def signs_tweets(tweet):
    signos = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)(#)")
    return signos.sub(' ', tweet.lower())


def remove_stopwords(df):
    spanish_stopwords = stopwords.words('spanish')
    return " ".join([word for word in df.split() if word not in spanish_stopwords])


def spanish_stemmer(x):
    stemmer = SnowballStemmer('spanish')
    return " ".join([stemmer.stem(word) for word in x.split()])

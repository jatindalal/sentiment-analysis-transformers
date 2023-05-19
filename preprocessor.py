import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

class Preprocessor:
    def __init__(self) -> None:
        self.stopwords = stopwords.words('english')
        self.stopwords.remove('no')
        self.stopwords.remove('not')
        self.stopwords.remove('nor')
        self.stopwords.remove('but')
        self.stopwords.remove('against')

        self.lemmatizer = WordNetLemmatizer()

        self.contraction_mappings = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

    def clean(self, text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove user mentions
        text = re.sub('#','', text) # remove hashtags
        text = ' '.join([self.contraction_mappings[word] if word in self.contraction_mappings else word for word in text.split(" ")]) # remove contractions
        text = re.sub(r'\W', ' ', text) # remove special characters
        text = re.sub(r'http\S+','',text) # remove url/links
        text = text.lower() # convert to lowercase

        words = text.split() # split to remove multiple white spaces
        return " ".join(words).strip()

    def remove_stop_words_and_lemmatize(self, text):
        tokens = [w for w in text.split() if not w in self.stopwords] # remove stopwords
        new_text = ""
        for token in tokens:
            new_text = new_text + self.lemmatizer.lemmatize(token, "v") + " "
        return new_text.strip()

    def preprocess(self, textlist):
        preprocessed_text = []

        for sentence in textlist:
            cleaned = self.clean(sentence)
            cleaned = self.remove_stop_words_and_lemmatize(cleaned)
            preprocessed_text.append(cleaned)

        return preprocessed_text
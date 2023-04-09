# import nltk
# nltk.download('punkt')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import argparse
import re
import pickle


MAX_LENGTHS = {
    "ecommerce": 1000,
    "tweet": 280,
    "drf": 320
}
ENCODINGS = {
    "ecommerce": ["Books", "Clothing & Accessories", "Electronics", "Household"],
    "tweet": ["anger", "fear", "happy", "love", "sadness", "surprise"],
    "drf": ["Ambiguous", "Emergency", "Flight Operations", "Ground Operations", "Weather"]
}


def get_normalized_words(text):
    text = re.sub(r"[^\w\s]", "", text)
    words = [w.lower() for w in word_tokenize(text)]
    stop_words = set(stopwords.words("english"))
    return [w for w in words if not w in stop_words]


def tokenize_and_pad(tokenizer, words, max_length):
    sequences = tokenizer.texts_to_sequences([words])
    return pad_sequences(sequences, maxlen=max_length)


def predict(text, context, model_type): 
    context = context.lower()
    model_type = model_type.lower()
    if context != "ecommerce" and context != "tweet" and context != "drf":
        return "ERROR: Context is invalid. Please choose 'ecommerce', 'tweet', or 'drf'."
    
    model = load_model(f"{model_type}/{context}_model")
    words = get_normalized_words(text)

    with open(f"{model_type}/{context}_tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    padded_sequences = tokenize_and_pad(tokenizer, words, MAX_LENGTHS[context])
    probabilities = model.predict(padded_sequences, verbose=0)
    prediction = np.argmax(probabilities)
    encoding = ENCODINGS[context]

    return f"{encoding[prediction]} with probability {probabilities[0][prediction] * 100}%"


def main():
    parser = argparse.ArgumentParser(
        description="This program takes in an example of text, a context (Ecommerce, Tweet, or DRF), and a model. It will return the predicted label and confidence for the provided text."
        )
    parser.add_argument("context", action="store")
    parser.add_argument("-m", "--model", action="store")
    args = parser.parse_args()
    text = input("Text to classify: ")
    print(predict(text, args.context, args.model))


if __name__ == "__main__":
    main()
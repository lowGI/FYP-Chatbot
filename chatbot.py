import re
import random
import numpy as np
import pickle
import json
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import webbrowser
lemmatizer = WordNetLemmatizer()

chatbot_model = load_model("Chatbot_Training/chatbot_model.h5")
intents_json = json.loads(open("Chatbot_Training/intents.json").read())
words = pickle.load(open("Chatbot_Training/words.pkl", "rb"))
classes = pickle.load(open("Chatbot_Training/classes.pkl", "rb"))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = chatbot_model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints):
    try:
        #print(ints)
        tag = ints[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    except:
        # When reach here, it means that there are no ints/class tags able to interpret user's message
        result = "Sorry, I don't understand you."

    return result
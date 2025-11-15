import random
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocessing Setup
lemmatizer = WordNetLemmatizer()
punct = dict((ord(i), None) for i in string.punctuation)

def preprocess(text):
    text = text.lower().translate(punct)
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(t) for t in tokens]

# Load Corpus
def load_corpus():
    with open('corpus.txt', 'r', errors='ignore') as f:
        data = f.read()
    return nltk.sent_tokenize(data)

# Greetings
greeting_inputs = ['hi', 'hello', 'hey', 'greetings']
greeting_responses = ['Hello!', 'Hi there!', 'Hey!', 'Yes, how can I help you?']

def check_greeting(text):
    for token in text.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)
    return None

# Generate response
def respond(user_query):
    sentences = load_corpus()
    sentences.append(user_query)

    tfidf = TfidfVectorizer(tokenizer=preprocess, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(sentences)

    sim_values = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
    index = sim_values.argsort()[0][-2]

    if sim_values[0][index] == 0:
        return "I cannot understand."
    return sentences[index]

# Main Chat Loop
print("Chatbot Ready! Type 'exit' to quit.")
while True:
    user = input("You: ").lower()

    if user == "exit":
        print("Bot: Goodbye!")
        break

    greet = check_greeting(user)
    if greet:
        print("Bot:", greet)
    else:
        print("Bot:", respond(user))

# ------------------------- Libraries import
# Word2Vec model
import gensim

# To calculate sosine similarity
import numpy as np
from numpy import dot
from numpy.linalg import norm

# To generate random values between 0 and 4
from random import randint

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ------------------------ Necessary Data reading
# Word2Vec model import
file_path = r"C:\Users\quimm\Downloads\GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)


# Reading the corpus
corpus_path = "saveDialogue.txt"
with open(corpus_path, "r") as f:
    text = f.readlines()
text = [line.strip() for line in text if line != "\n"]

# stopwords
stopwords_path = "StopWords.txt"


# ---------------- Implementation of classes and methods


class Dialogue:
    """
    Implements the logic behind a dialogue.
    It consists of the question, answer and word_embedding of the question
    """

    def __init__(self, question, answer):
        self.question = self.clean_sentence(question)
        self.answer = answer
        self.word_embedding = []

    def create_word_embedding(self, model, stopwords):

        # splitting the question to retrieve its tokens
        words = self.question.split()

        # Creates a vector based on the mean of the tokens in the sentence that are in the model and that are not stopwords
        word_embedding = [
            model[token] for token in words if token in model and token not in stopwords
        ]

        # handling sentences that do not fill the requirements
        if len(word_embedding) < 2:
            word_embedding.append(np.zeros(300))
            word_embedding.append(np.zeros(300))

        self.word_embedding = sum(word_embedding) / 300

        return

    def clean_sentence(self, sentence):
        """
        Cleans a sentence before computing word embeddings
        """
        # to lower
        text = sentence.lower()

        # remove punct
        text = text.replace(".", " ")
        text = text.replace(",", "")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace(":", "")
        text = text.replace("!", "")
        text = text.replace("?", "")
        text = text.replace("-", " ")
        text = text.replace("'", "")
        text = text.replace('"', "")

        return text


class Corpus:
    """
    Implements the storage of the training data.
    """

    def __init__(self, model, text, stopwords=[]):

        self.model = model
        self.stopwords = stopwords

        # encoding the input text as dialogues and computing their word embedding.
        self.corpus = []
        for line in text:
            line = line.split("\t")
            # print(line)
            dialogue = Dialogue(question=line[0], answer=line[1])
            dialogue.create_word_embedding(self.model, self.stopwords)
            self.corpus.append(dialogue)


class Chatbot:
    """
    Implements the chatbot.
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def find_relevant_answer(self, input_dialogue):
        """
        Finds a suitable answer to an input sentence
        """

        # Computing similarities throug cosine similarity
        similarities = []
        for i, train_dialogue in enumerate(self.corpus.corpus):

            cosine_similarity = dot(
                input_dialogue.word_embedding, train_dialogue.word_embedding
            ) / (
                norm(input_dialogue.word_embedding)
                * norm(train_dialogue.word_embedding)
            )

            # handling sentences whose words do not appear in the word2vec model and have a vector full of 0s
            if np.isnan(cosine_similarity):
                cosine_similarity = 0

            similarities.append((i, cosine_similarity))

        # Find relevant answers
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Retrieve the index of the chosen answer and return it
        chosen_answer_index = similarities[randint(0, 5)][0]
        chosen_answer = self.corpus.corpus[chosen_answer_index].answer

        return chosen_answer

    def start_conversation(self):
        print("The chatbot has started.")
        print("In order to shut it down, enter STOP")

        input_sentence = input("User: ")
        while input_sentence != "STOP":

            # encoding input as a dialogue to use its properties
            input_dialogue = Dialogue(question=input_sentence, answer="")
            input_dialogue.create_word_embedding(
                model=self.corpus.model, stopwords=self.corpus.stopwords
            )

            # Finding a relevant answer
            relevant_answer = self.find_relevant_answer(input_dialogue)
            print("Chatbot: ", relevant_answer)

            # Continue asking for more questions
            input_sentence = input("User: ")


def main():
    print("--- Program starting ---")

    # Making it optionally to add stopwords
    stopw_question = input("Do you want the model to remove stopwords or not? [Y/N]:")
    if stopw_question == "N":
        stopwords = []
    else:
        with open(stopwords_path, "r") as f:
            stopwords = f.readlines()
            stopwords = [st_word.strip() for st_word in stopwords]

    # Initializating the corpus and starting the chatbot
    corpus = Corpus(model=model, stopwords=stopwords, text=text)
    chatbot = Chatbot(corpus=corpus)
    chatbot.start_conversation()


if __name__ == "__main__":
    main()

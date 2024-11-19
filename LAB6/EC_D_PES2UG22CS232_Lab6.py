import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter

import matplotlib.pyplot as plt

class NaiveBayesClassifier:

    """
    A simple implementation of the Naive Bayes Classifier for text classification.
    """


    @staticmethod
    def plot_word_frequency(sentences: list):
        """
        Plots the frequency of words in the provided sentences. 
        """
        all_words = ' '.join(sentences).split()
        word_counts = Counter(all_words)
        most_common_words = word_counts.most_common(10) 
        words, counts = zip(*most_common_words)

        plt.figure(figsize=(10, 6))
        plt.bar(words, counts, color='skyblue')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Frequent Words in Training Dataset')
        plt.xticks(rotation=45)
        plt.yticks(np.arange(0, max(counts) + 1, 1))
        plt.tight_layout()
        plt.show()
        
        

    @staticmethod
    def preprocess(sentences: list, categories: list) -> tuple:

        """
        Preprocess the dataset to remove missing or incorrect labels and balance the dataset.

        Args:
            sentences (list): List of sentences to be processed.
            categories (list): List of corresponding labels.

        Returns:
            tuple: A tuple of two lists - (cleaned_sentences, cleaned_categories).
        """

        # TO DO
        cleaned_sentences_list = []
        cleaned_categories = []

        stop_words = ENGLISH_STOP_WORDS

        valid_categories = set(["technology", "travel", "entertainment","food","health","fashion"])

        for sentence, category in zip(sentences, categories):
            
            if category not in valid_categories:
                continue

            words = sentence.lower().split()
            filtered_words = [word for word in words if word not in stop_words]
            
            cleaned_sentence = " ".join(filtered_words)
            cleaned_sentences_list.append(cleaned_sentence)
            cleaned_categories.append(category)
            '''
            # cleaned_words = [''.join(e for e in word if e.isalnum()) for word in filtered_words]
            # cleaned_words = [word for word in cleaned_words if word]

            # cleaned_category = category.lower()

            # if cleaned_words:  
                # cleaned_sentences.append(" ".join(cleaned_words))
                # cleaned_categories.append(cleaned_category)
            '''  
            # NaiveBayesClassifier.plot_word_frequency(cleaned_sentences_list)
        NaiveBayesClassifier.plot_word_frequency(cleaned_sentences_list)
        return cleaned_sentences_list, cleaned_categories
        pass

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray) -> tuple:

        """
        Trains the Naive Bayes Classifier using the provided training data.
        
        Args:
            X (numpy.ndarray): The training data matrix where each row represents a document
                            and each column represents the presence (1) or absence (0) of a word.
            y (numpy.ndarray): The corresponding labels for the training documents.

        Returns:
            tuple: A tuple containing two dictionaries:
                - class_probs (dict): Prior probabilities of each class in the training set.
                - word_probs (dict): Conditional probabilities of words given each class.
        """

        # TO DO
        classes, class_counts = np.unique(y, return_counts=True)
        class_probs = {c: count / len(y) for c, count in zip(classes, class_counts)}

        word_probs = {c: {} for c in classes}

        for i, c in enumerate(y):
            for j in range(X.shape[1]):
                if j not in word_probs[c]:
                    word_probs[c][j] = 1
                word_probs[c][j] += X[i, j]

        for c in classes:
            total_words_in_class = sum(word_probs[c].values())
            for word in word_probs[c]:
                word_probs[c][word] /= total_words_in_class


        return class_probs, word_probs
    
        pass

    @staticmethod
    def predict(X: np.ndarray, class_probs: dict, word_probs: dict, classes: np.ndarray) -> list:

        """
        Predicts the classes for the given test data using the trained classifier.

        Args:
            X (numpy.ndarray): The test data matrix where each row represents a document
                            and each column represents the presence (1) or absence (0) of a word.
            class_probs (dict): Prior probabilities of each class obtained from the training phase.
            word_probs (dict): Conditional probabilities of words given each class obtained from training.
            classes (numpy.ndarray): The unique classes in the dataset.

        Returns:
            list: A list of predicted class labels for the test documents.
        """
        #TO DO
        predictions = []
        for doc in X:
            posteriors = {}
            for cls in classes:
                log_prob = np.log(class_probs[cls])
                
                for j in range(X.shape[1]):
                    if doc[j] == 1:
                        log_prob += np.log(word_probs[cls].get(j, 1e-6))
                posteriors[cls] = log_prob
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)

        return predictions
        pass
    
    '''
    # @staticmethod
    # def plot_word_frequency(sentences: list):
    #     """
    #     Plots the frequency of words in the provided sentences. 
    #     """
    #     all_words = ' '.join(sentences).split()
    #     word_counts = Counter(all_words)
    #     most_common_words = word_counts.most_common(10)
    #     words, counts = zip(*most_common_words)

    #     plt.figure(figsize=(10, 6))
    #     plt.bar(words, counts, color='skyblue')
    #     plt.xlabel('Words')
    #     plt.ylabel('Frequency')
    #     plt.title('Top 10 Most Frequent Words in Training Dataset')
    #     plt.xticks(rotation=45)
    #     plt.yticks(np.arange(0, max(counts) + 1, 1))
    #     plt.tight_layout()
    #     plt.show()
    '''
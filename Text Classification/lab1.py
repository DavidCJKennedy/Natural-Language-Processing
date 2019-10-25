#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, inspect, re, random
import numpy as np
import pylab as plt
from collections import Counter


#==============================================================================
## MARK - Class to handle the initialisation of the script in commandline

class CommandLine:
    
## MARK - Initialisation method, set class variables, get the file path of the datasets
    
    def __init__(self):
        # Retrieve command line arguments
        dataFolder = sys.argv[1]
        # Obtain the path to the current directory 
        # It is assumed that the data folder is also in this directory
        currentDirectory = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        directory = currentDirectory + '/' + dataFolder
        
        # Obtain the next direct sub-folders within the dataset folder
        dataFolder = next(os.walk(directory))[1][0]
        sentimentsFolder = next(os.walk(directory + '/' + dataFolder))[1]
        # Set the path to the negative reviews folder
        self.negDirPath = directory + '/' + dataFolder + '/' + sentimentsFolder[0]
        # Set the path to the positive reviews folder
        self.posDirPath = directory + '/' + dataFolder + '/' + sentimentsFolder[1]
    
    
#==============================================================================
## MARK - Generate Test and Training Datasets from the Full Dataset
           
class Datasets:
    
    # Initialise a blank numpy array for the training dataset filenames
    negTrainingFileNames = np.array([])
    posTrainingFileNames = np.array([])
    
    # Initialise a blank numpy array for the testing dataset filenames
    negTestingFileNames = np.array([])
    posTestingFileNames = np.array([])
    
    # Initialise a blank dictionary for the uni-gram datasets
    uniGramTrainingSet = {}
    uniGramTestingSet = {}
    
    # Initialise a blank dictionary for the bi-gram datasets
    biGramTrainingSet = {}
    biGramTestingSet = {}
    
    # Initialise a blank dictionary for the tri-gram datasets
    triGramTrainingSet = {}
    triGramTestingSet = {}
    
## MARK - Initialisation method, initialise the datasets for uni-gram, bi-gram and tri-gram models
    
    def __init__(self, negPath, posPath, stopwords):
        # Retrieve the filenames for each dataset, training to test dataset ratio = 0.8  
        negTrain, negTest = self.splitFullDataSet(negPath, 0.8)
        posTrain, posTest = self.splitFullDataSet(posPath, 0.8)
        
        # Append Retrieved training dataset file names to arrays
        self.negTrainingFileNames = np.append(self.negTrainingFileNames, negTrain)
        self.posTrainingFileNames = np.append(self.posTrainingFileNames, posTrain)
        
        # Generate the Uni-Gram training set from the postive and negative reviews
        self.uniGramTrainingSet.update(self.generateUniGramDataSet(-1, negPath, self.negTrainingFileNames, stopwords))
        self.uniGramTrainingSet.update(self.generateUniGramDataSet(1, posPath, self.posTrainingFileNames, stopwords))
        
        # Generate the Bi-Gram training set from the postive and negative reviews
        self.biGramTrainingSet.update(self.generateBiGramDataSet(-1, negPath, self.negTrainingFileNames, stopwords))
        self.biGramTrainingSet.update(self.generateBiGramDataSet(1, posPath, self.posTrainingFileNames, stopwords))
        
        # Generate the Tri-Gram training set from the postive and negative reviews
        self.triGramTrainingSet.update(self.generateTriGramDataSet(-1, negPath, self.negTrainingFileNames, stopwords))
        self.triGramTrainingSet.update(self.generateTriGramDataSet(1, posPath, self.posTrainingFileNames, stopwords))
        
        # Append Retrieved test dataset file names to arrays
        self.negTestingFileNames = np.append(self.negTestingFileNames, negTest)
        self.posTestingFileNames = np.append(self.posTestingFileNames, posTest)
        
        # Generate the Uni-Gram test set from the postive and negative reviews
        self.uniGramTestingSet.update(self.generateUniGramDataSet(-1, negPath, self.negTestingFileNames, stopwords))
        self.uniGramTestingSet.update(self.generateUniGramDataSet(1, posPath, self.posTestingFileNames, stopwords))
        
        # Generate the Uni-Gram test set from the postive and negative reviews
        self.biGramTestingSet.update(self.generateBiGramDataSet(-1, negPath, self.negTestingFileNames, stopwords))
        self.biGramTestingSet.update(self.generateBiGramDataSet(1, posPath, self.posTestingFileNames, stopwords))
        
        # Generate the Uni-Gram test set from the postive and negative reviews
        self.triGramTestingSet.update(self.generateTriGramDataSet(-1, negPath, self.negTestingFileNames, stopwords))
        self.triGramTestingSet.update(self.generateTriGramDataSet(1, posPath, self.posTestingFileNames, stopwords))
        
## MARK - Method to retrieve the file names of the test and training set
    
    def splitFullDataSet(self, path, ratio):
        # Retrieve all file names in the folder at path 
        fileNames = os.listdir(os.path.abspath(path))
        # Split filenames into the training and test set according to ratio
        trainingFileNames = fileNames[:int((len(fileNames)*ratio))]
        testingFileNames = fileNames[int((len(fileNames)*ratio)):]
        return trainingFileNames, testingFileNames
    
## MARK - Method to generate the Uni-Gram model ("Term1": Count)    

    def generateUniGramDataSet(self, polarity, dirPath, fileNames, stopwords):
        # Initialise a blank dataset dictionary to store Unigrams for reviews
        dataSet = {}
        
        # Iterate through each filename in the datset
        for fileName in fileNames:
            # Initialise a blank dictionary to store the bag of words for each review
            bagOfWords = {}
            # Open the review at the file path
            review = open(dirPath + '/' + fileName, 'r')
            # Read in the review
            text = review.read()
            # Use regex to split text by word and blank space
            symbols = re.sub("[\W]", " ", text).split()
            # Remove any symbols in the text that are a stopword
            symbols = [symbol for symbol in symbols if symbol not in stopwords]
            
            # Iterate through each remaining text symbol
            for symbol in symbols:
                # If symbol is not already in the bag of words model
                if symbol not in bagOfWords: 
                    # A new symbol found so add as new entry to the bag of words
                    bagOfWords[symbol] = 1
                else:
                    # Symbol already exists so add one to the count
                    bagOfWords[symbol] = bagOfWords[symbol] + 1
            
            # Having iterated through each symbol in the pre-processed review text 
            # Create a new entry to the dataset dictionary
            # Dictionary key contains the review label, either postive or negative,
            # and file name
            # Dictionary value is the bag of words
            dataSet[(polarity, fileName)] = bagOfWords
            # Close the review file to save memory 
            review.close()
        
        # Return complete dataset
        return dataSet
    
## MARK - Method to generate the Bi-Gram model, ("Term1", "Term2": Count)
    
    def generateBiGramDataSet(self, polarity, dirPath, fileNames, stopwords):
        # Initialise a blank dataset dictionary to store Bigrams for reviews
        dataSet = {}
        
        # Iterate through each filename in the datset
        for fileName in fileNames:
            # Initialise a blank dictionary to store the bigrams for each review
            biGrams = {}
            # Open the review at the file path
            review = open(dirPath + '/' + fileName, 'r')
            # Read in the review
            text = review.read()
            # Use regex to split text by word and blank space
            symbols = re.sub("[\W]", " ", text).split()
            # Remove any symbols in the text that are a stopword
            symbols = [symbol for symbol in symbols if symbol not in stopwords]
        
            # Iterate through the number of symbols in the review, to retrieve an index
            # len -1 as the final bigram takes up the final two symbols in the review
            for iteration in range((len(symbols) - 1)):
                # The bigram is the symbol at the index and the symbol immediately following 
                biGram = (symbols[iteration], symbols[iteration + 1])
                # If biGram is not already in the biGram model
                if biGram not in biGrams:
                    # A new biGram found so add as new entry
                    biGrams[biGram] = 1
                else:
                    # Bigram already exists so add one to the count
                    biGrams[biGram] = biGrams[biGram] + 1
            

            dataSet[(polarity, fileName)] = biGrams
            # Close the review file to save memory 
            review.close()
        
        # Return complete dataset
        return dataSet
    
## MARK - Method to generate the Tri-Gram model, ("Term1", "Term2", "Term3": Count)
        
    def generateTriGramDataSet(self, polarity, dirPath, fileNames, stopwords):
        # Initialise a blank dataset dictionary to store Trigrams for reviews
        dataSet = {}
        
        # Iterate through each filename in the datset
        for fileName in fileNames:
            # Initialise a blank dictionary to store the trigrams for each review
            triGrams = {}
            review = open(dirPath + '/' + fileName, 'r')
            text = review.read()
            # Use regex to split text by word and blank space
            symbols = re.sub("[\W]", " ", text).split()
            # Remove any symbols in the text that are a stopword
            symbols = [symbol for symbol in symbols if symbol not in stopwords]
        
            # Iterate through the number of symbols in the review, to retrieve an index
            # len -2 as the final trigram takes up the final three symbols in the review
            for iteration in range((len(symbols) - 2)):
                # The trigram is the symbol at the index and the two symbols immediately following 
                triGram = (symbols[iteration], symbols[iteration + 1], symbols[iteration + 2])
                if triGram not in triGrams:
                    triGrams[triGram] = 1
                else:
                    triGrams[triGram] = triGrams[triGram] + 1
            
            dataSet[(polarity, fileName)] = triGrams
            review.close()
            
        # Return complete dataset
        return dataSet
        

#==============================================================================
## MARK - Generate Test and Training Datasets from the Full Dataset

class Weights:
    # Initialise a blank class weight dictionary to store initial weights for each symbol
    weights = {}
    
## MARK - Initialisation method, setting class variables
    
    def __init__(self, dataset):
        # Retrieve initial weights for each symbol in training dataset
        self.weights = self.generateInitialWeights(dataset)
        
## MARK - Method to generate a blank weights dictionary, where all symbols have initial weight 0
        
    def generateInitialWeights(self, dataset):
        # Initialise a blank dictionary to store weights for each symbol
        weights = {}
        
        # Iterate through each item in the dataset 
        for filename, text in dataset.items():
            # Iterate through each symbol in the review 
            for symbol in text:
                # If symbol is already in the weights then it can be ignored 
                if symbol in weights:
                    continue
                # Else a new symbol has been found and is added to the weights
                # Setting the initial weighting of this symbol to 0
                else:
                    weights[symbol] = 0
                    
        return weights
    

#==============================================================================
## MARK - Class to Train the Term Weights using the Perceptron Algorithm
           
class Training:
    # Initialise a blank class weight dictionary to store the update trained weights
    weights = {}
    # Initialise a previous error value, assumed to be max error of 1
    previousError = 1.0
    # Initialise a current error value, to be overwritten
    currentError = 0.0
    # Initialise an errors array to store training error at each iteration
    errors = np.array([])
    # Initialise a seed counter so that the random seed remains repeatable
    # The seed value increases for each iteration as to produce different 
    # dataset order each iteration. 
    
    seedCounter = 0
    
## MARK - Initialisation method, setting class variables, begin training
    
    def __init__(self, dataset, weights):
        self.weights = weights
        
        # Whilst the difference in error between iterations is greater than 0.005
        # The model has not converged, so continue training. Once the difference
        # In error between the current and previous iteration is less than 0.005
        # The model is said to have converged and so training can be stopped
        while self.previousError - self.currentError > 0.005:
            
            # On training iterations other than the first set the previous error
            # To what was the current error, this is a new iteration and so the error is not current 
            if self.seedCounter > 0:
                self.previousError = self.currentError
                # Add iteration error to the array to allow for graph creation
                self.errors = np.append(self.errors, self.currentError)
            
            # Randomise the dataset order
            randomisedDataset = self.randomiseDataset(dataset)
            # Start a training iteration using the randomised dataset 
            # Return the error for that training iteration and updated weights
            weights, error = self.train(randomisedDataset, self.weights)
            # Average the weights for each training iteration
            averageWeights = {key: round((value/self.seedCounter), 3) for key, value in weights.items()}
            self.weights = averageWeights
            self.currentError = error
            
            print("Training Iteration: " + str(self.seedCounter) + ", with error: " + str(round((self.previousError - self.currentError), 4)))
        
        # Sort the weights array to find most negative and most postitive terms
        # Counter far faster than using .count to sort weights list, faster by n(O(n^2) vs O(n))
        sortedWeights = [weight for weight, _ in Counter(averageWeights).most_common()]
        # Print the 10 most postive and negative terms
        print(sortedWeights[:10])
        print(sortedWeights[-10:])
        
## MARK - Method to randomise the order of the dataset      
        
    def randomiseDataset(self, dataset):
        # Initialise a blank dictionary to store the randomised dataset from the input dataset
        randomisedDataset = {}
        
        # Initialise the random seed so the results are repeatable
        # The seed is the number that the random package uses to form the random numbers
        # If seed is not specified then the package would use the current time 
        # This would not be repeatable
        random.seed(self.seedCounter)
        # Get the keys in the dataset
        keys = list(dataset.keys())
        # Shuffle the keys so they have a random order
        random.shuffle(keys)
        
        # Generate a new random order dataset from the orignal dataset inputted 
        # Using the randomly ordered keys created
        for key in keys:
            randomisedDataset.update({key: dataset[key]})
        # Update the seed counter so the dataset for each iteration has a different order
        self.seedCounter = self.seedCounter + 1
        
        # Return the randomised Dataset
        return randomisedDataset
    
## MARK - Method to apply perceptron algorithm to train the weighting terms 
    
    def train(self, dataset, weights):
        # Intialise the number of reviews predicted correctly and incorrectly 
        # to zero. These values are used to calculate the training error
        numCorrect = 0
        numIncorrect = 0
        
        # Iterate through each review in the training dataset dictionary
        for polarityFilename, review in dataset.items():
            
            # Return the terms in the review text and the exact same terms found
            # In the weights
            keys = review.keys() & weights.keys()
            
            # Calculate the dot product of the occurances of term in the review 
            # and the weight for that term.
            # IE if the term "good" appears 4 times in the review and has a weight of 
            # +3 in the weights the product is 12.
            # These products for each term in the review are summed to generate the dot 
            # product.
            sentimentSum = sum(review[key] * weights[key] for key in keys)
            
            # Use the sign step input function to turn the sum into sentiments, -1 for any sum 
            # less than 0, 0 for any sum = 0 and +1 for any sum greater than 0 
            sentiment = np.sign(sentimentSum)
                   
            # A sentiment prediction of 0 is classed as postive so turn any 0 sentiments to
            # positive ones
            if sentiment == 0:
                sentiment = 1
            
            # Check the sentiment prediction to the review label 
            # If the prediction is different to the actualy statement the weights
            # for the term in that review must be updated
            if sentiment != polarityFilename[0]:
                # Update the counter that a review was incorrectly predicted
                numIncorrect = numIncorrect + 1
                
                # Iterate through each term in the review and update that 
                # corresponding term in the weights.
                # If the model has predicted a postive sentiment when the review is negative
                # take away -1 from each term in the weights. 
                # If the model predicts negative sentiment when the review is postive 
                # add +1 to each of term weights .
                for key in keys:
                    weights[key] = weights[key] + polarityFilename[0]
            
            # Else the predicted sentiment is the same as the actual review sentiment
            # The weights do NOT need to be updated so update the correct guess counter
            else:
                numCorrect = numCorrect + 1
        
        # Calculate the error for the training iteration 
        error = 1 - (numCorrect / (numCorrect + numIncorrect))    
        
        # Return the updated trained weights and iteration error
        return weights, error
    
    
#==============================================================================
## MARK - Class to test the learnt weights against unseen data 

class Testing:
    
## MARK - Initialisation method, begin testing with testing dataset and learnt weights
    
    def __init__(self, dataset, weights):
        self.testingError = self.test(dataset, weights) 
        
## MARK - Method to test the unseen dataset with the learnt weights
        
    def test(self, dataset, weights):
        # Intialise the number of reviews predicted correctly and incorrectly 
        # to zero. These are used to calculate system accuracy
        numCorrect = 0
        numIncorrect = 0
        
        # Iterate through each review in the test dataset dictionary
        for polarityFilename, review in dataset.items():
            
            # Return the terms in the review text and the exact same terms found
            # in the weights
            keys = review.keys() & weights.keys()
            
            # Calculate the dot product of the occurances of the terms in the review 
            # and the weights for those review terms.
            sentimentSum = sum(review[key] * weights[key] for key in keys)
            
            # Use the sign step input function to turn the sum into a sentiment prediction
            sentiment = np.sign(sentimentSum)
            
            # A sentiment prediction of 0 is classed as postive
            if sentiment == 0:
                sentiment = 1
            
            # Check the sentiment prediction to the review label, if they differ
            # the prediction was incorrect
            if sentiment != polarityFilename[0]: 
                numIncorrect = numIncorrect + 1
            # Else the sentiment prediction and class label are the same, 
            # the review has been correctly predicted
            else:
                numCorrect = numCorrect + 1
        
        # Calculate the error the the test data
        error = 1 - (numCorrect / (numCorrect + numIncorrect))
        # Return the model test error
        return error 


#==============================================================================
## MARK - Used to run the script from the commandline
        
if __name__ == '__main__':
    
    print("Program Initalised")
    
    # A curated array of stopwords, words that would not be beneficial in calculating sentiments
    stopwords = ["i", "ll", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "between", "into", "through", "during", "before", "after", "to", "from", "up", "down", "in", "out", "on", "off", "over", "further", "then", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "s", "t", "can", "will", "just", "don", "should", "now"]
    
    # Intialise the code file variables using the Commandline Class
    config = CommandLine()
    
    # Build the datasets from the full data corpus
    datasets = Datasets(config.negDirPath, config.posDirPath, stopwords)
    
    # Initialise the blank weights for each model type
    uniGramWeights = Weights(datasets.uniGramTrainingSet)
    print("Initialised Uni-Gram Weights")
    biGramWeights = Weights(datasets.biGramTrainingSet)
    print("Initialised Bi-Gram Weights")
    triGramWeights = Weights(datasets.triGramTrainingSet)
    print("Initialised Tri-Gram Weights")
    
    # Train each model type
    print("Commenced Training Uni-Gram Model")
    uniGramTraining = Training(datasets.uniGramTrainingSet, uniGramWeights.weights)
    print("Commenced Training Bi-Gram Model")
    biGramTraining = Training(datasets.biGramTrainingSet, biGramWeights.weights)
    print("Commenced Training Tri-Gram Model")
    triGramTraining = Training(datasets.triGramTrainingSet, triGramWeights.weights)
    
    # Test each model type
    print("Commenced Testing Uni-Gram Model")
    uniGramTesting = Testing(datasets.uniGramTestingSet, uniGramTraining.weights)
    print("Commenced Testing Bi-Gram Model")
    biGramTesting = Testing(datasets.biGramTestingSet, biGramTraining.weights)
    print("Commenced Testing Bi-Gram Model")
    triGramTesting = Testing(datasets.triGramTestingSet, triGramTraining.weights)
    
    # Calculate the accuracy of each model
    print("Uni-Gram Testing Accuracy: " + str(round((1 - uniGramTesting.testingError), 4)))
    print("Bi-Gram Testing Accuracy: " + str(round((1 - biGramTesting.testingError), 4)))
    print("Tri-Gram Testing Accuracy: " + str(round((1 - triGramTesting.testingError), 4)))
    
    # Plot a graph of the change in prediction error during training
    #plt.xlabel('Iteration')
    #plt.ylabel('Prediction Error')
    #plt.plot(range(1, uniGramTraining.seedCounter), uniGramTraining.errors)
    #plt.plot(range(1, biGramTraining.seedCounter), biGramTraining.errors)
    #plt.plot(range(1, triGramTraining.seedCounter), triGramTraining.errors)
    #plt.legend(['Uni-Gram', 'Bi-Gram', 'Tri-Gram'], loc='lower right')
    #plt.show(block = False)

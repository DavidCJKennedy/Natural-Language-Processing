#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, random, time
import numpy as np
from heapq import nlargest
from collections import Counter
from sklearn.metrics import f1_score


#==============================================================================
## MARK - Class to handle the initialisation of the script in commandline

class CommandLine:
    
## MARK - Initialisation method, get the file path of the datasets, format datasets correctly
    
    def __init__(self):
        # Retrieve command line arguments
        self.testType = sys.argv[1]
        trainFileName = sys.argv[2]
        testFileName = sys.argv[3]
        
        self.trainDataset = self.load_dataset_sents(trainFileName)
        self.testDataset = self.load_dataset_sents(testFileName)
        
## MARK - Method to import the sentences with corresponding labels 

    def load_dataset_sents(self, file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
        targets = []
        inputs = []
        zip_inps = []
        with open(file_path) as f:
            for line in f:
                sent, tags = line.split('\t')
                words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
                ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
                inputs.append(words)
                targets.append(ner_tags)
                zip_inps.append(list(zip(words, ner_tags)))
        return zip_inps if as_zip else (inputs, targets)


#==============================================================================
## MARK - Class used to extract the Natural Language features from a data corpus

class FeatureExtraction:
        
    def __init__(self, corpus):
        self.cw_cl_counts = self.currentWordCurrentLabel(corpus)
        
    def currentWordCurrentLabel(self, corpus):
        currentWordCurrentLabel = Counter()
        for line in corpus:
            currentWordCurrentLabel.update(Counter(line))

        return Counter({ key : value for key, value in currentWordCurrentLabel.items() })
  
    
#==============================================================================
## MARK - Class used to train, test and evaluate the Perceptron
        
class Perceptron:
    
    def __init__(self, trainCorpus, testCorpus, epochs, cw_ct, beamWidth, isViterbi):
        if isViterbi:
            self.trainedViterbiWeights = self.trainPerceptronViterbi(trainCorpus, epochs, cw_ct)
            correct_tags, viterbiPredictions = self.testViterbiPerceptron(testCorpus, cw_ct, self.trainedViterbiWeights)
            self.evaluate(correct_tags, viterbiPredictions)
        else:
            self.trainedBeamSearchWeights = self.trainPerceptronBeamSearch(trainCorpus, epochs, cw_ct, beamWidth)
            correct_tags, beamPredictions = self.testBeamSearchPerceptron(testCorpus, cw_ct, self.trainedBeamSearchWeights, beamWidth)
            self.evaluate(correct_tags, beamPredictions)
   
## MARK - Method to return the feature representation of a sentence
        
    def phi_1(self, sent, cw_ct):
        phi_1 = Counter()
        # include features only if found in feature space
        phi_1.update([item for item in sent if item in cw_ct.keys()])
        return phi_1
          
## MARK - Method to return tag label predictions for a sentence using Viterbi
    
    def virterbiMatrix(self, sentence, weights, cw_ct):
        # Separate the words and labels in a sentence
        words, labels = list(zip(*sentence))
        allTags = ["O", "PER", "LOC", "ORG", "MISC"]
        # Create a blank matrix with correct dimensions to store the scores for each word/label pair
        matrix = [[0 for i in range(len(allTags))] for j in range(len(words))]
        # Create a blank array to store the backpointers, highlighting the most probable path
        backPointers = [(0, 0) for i in range(len(words) - 1)]
        prediction = []
        
        # Iterate through each word in the sentence
        for i in range(len(words)):
            terms = []
            # Gather all possible word/label pairs for a word
            for tag in allTags:
                terms.append((words[i], tag))
            
            # Return the feature representation
            phi = self.phi_1(terms, cw_ct)
            
            # Itterate through each of the possible tags
            for j in range(len(allTags)):
                highestScore = 0
                sumScore = 0
                currentTerm = (words[i], allTags[j])
                
                # Check the current term exists in the feature representation
                if currentTerm in phi:
                    # If it exists calculate the dot product
                    sumScore = weights[currentTerm] * phi[currentTerm]
                else:
                    sumScore = 0
                
                # Iterate through each previous score
                for k in range(len(allTags)):
                    # If i = is 0 then this is the beginning of a sentence and has 0 probability
                    if i == 0:
                        continue
                    else:
                        # Return and sum all previous scores
                        previousScore = matrix[i - 1][k]
                        sumScore += previousScore
                    
                    # Find and add the back pointers
                    if (k == 0 and i != 0):
                        highestScore = previousScore
                        backPointers[i - 1] = (i - 1, k)
                    elif (i != 0):
                        if previousScore > highestScore:
                            backPointers[i - 1] = (i - 1, k)
                
                # Add the score to the matrix
                matrix[i][j] = sumScore
                
        # Get the initial highest scoring tag label
        index = (len(words) - 1, np.argmax(matrix[-1], axis = 0))
        prediction.append((words[index[0]], allTags[index[1]]))
        
        # Iterate through the back pointers and find the predicted word/label combination
        for i in range(len(backPointers)):  
            index = backPointers[index[0] - 1]
            prediction.append((words[index[0]], allTags[index[1]]))
        
        # Flip the order of predictions
        prediction = prediction[:: -1]
        
        return prediction
    
## MARK - Method to return tag label predictions for a sentence using Viterbi and Beam Search
    
    def beamSearch(self, sentence, weights, cw_ct, beamWidth):
        # Separate the words and labels in a sentence
        words, labels = list(zip(*sentence))
        allTags = ["O", "PER", "LOC", "ORG", "MISC"]
        # Create a blank matrix with correct dimensions to store the scores for each word/label pair
        matrix = [[0 for i in range(len(allTags))] for j in range(len(words))]
        # Create a blank array to the reduced matrix holding the highest scoring index for each word
        beamMatrix = [[] for i in range(len(words))]
        prediction = []
        
        # Iterate through each word in the sentence
        for i in range(len(words)):
            terms = []
            # Gather all possible word/label pairs for a word
            for tag in allTags:
                terms.append((words[i], tag))
            
            # Return the feature representation
            phi = self.phi_1(terms, cw_ct)
            
            # Itterate through each of the possible tags
            for j in range(len(allTags)):
                sumScore = 0
                currentTerm = (words[i], allTags[j])
                
                # Check the current term exists in the feature representation
                if currentTerm in phi:
                    sumScore = weights[currentTerm] * phi[currentTerm]
                else:
                    sumScore = 0
                
                # Iterate through each of the previous highest scoring labels and add to the score
                if i != 0:
                    for index in beamMatrix[i - 1]:
                        sumScore += matrix[i - 1][index]
                
                # Add the score to the matrix
                matrix[i][j] = sumScore
            
            # Find the required number indexes of the biggest values in the full size matrix and
            # add to the beam matrix
            biggestIndexes = nlargest(beamWidth, range(len(matrix[i])), matrix[i].__getitem__)
            beamMatrix[i].extend(biggestIndexes)
        
        # Find the most probable word and tag label combination and return the prediction
        for i in range(len(words)): 
            prediction.append((words[i], allTags[beamMatrix[i][0]]))
            
        return prediction
        
## MARK - Method to train the perceptron using the Viterbi Algorithm  
        
    def trainPerceptronViterbi(self, corpus, epochs, cw_ct):
        weights = Counter()
        time.time()
    
        # Iterate through each training epoch
        for epoch in range(epochs):
            false = 0
            # Begin the timer
            now = time.time()
            # Shuffle the dataset using a known random seed for repeatability
            random.seed(epoch + 1)
            random.shuffle(corpus)
            
            for line in corpus:
                # Return the predicted labels for each word in the line 
                prediction = self.virterbiMatrix(line, weights, cw_ct)
                
                # Check whether the prediction is the same as the correct values
                if prediction != line:
                    # Update the weights accordingly if the prediction is incorrect
                    correct = Counter(line)
                    prediction = Counter(prediction)
                    predicted = Counter({key : -value for key, value in Counter(prediction).items()})
                                        
                    weights.update(correct)
                    weights.update(predicted)
                    
                    # Update the false guess counter
                    false += 1

            print("Epoch: ", epoch + 1, 
                  " / Time for epoch: ", round(time.time() - now, 2),
                 " / No. of false predictions: ", false)
        
        # return the trained weights
        return weights
    
## MARK - Method to train the perceptron using the Viterbi Algorithm with Beam Search included
    
    def trainPerceptronBeamSearch(self, corpus, epochs, cw_ct, beamWidth):
        weights = Counter()
        time.time()
        
        # Iterate through each training epoch
        for epoch in range(epochs):
            false = 0
            # Begin the timer
            now = time.time()
            # Shuffle the dataset using a known random seed for repeatability
            random.seed(epoch + 1)
            random.shuffle(corpus)
            
            for line in corpus:
                # Return the predicted labels for each word in the line 
                prediction = self.beamSearch(line, weights, cw_ct, beamWidth)
                
                # Check whether the prediction is the same as the correct values
                if prediction != line:
                    # Update the weights accordingly if the prediction is incorrect
                    correct = Counter(line)
                    prediction = Counter(prediction)
                    predicted = Counter({key : -value for key, value in Counter(prediction).items()})
                                        
                    weights.update(correct)
                    weights.update(predicted)
                    
                    false += 1

            print("Epoch: ", epoch + 1, 
                  " / Time for epoch: ", round(time.time() - now, 2),
                 " / No. of false predictions: ", false)
        
        # return the trained weights
        return weights
    
## MARK - Method to test the Viterbi Algorithm and return the predicted and correct labels
    
    def testViterbiPerceptron(self, corpus, cw_ct, weights):
        correct_tags = []
        predicted_tags = []
                
        for line in corpus:
    
            words, labels = list(zip(*line))
            correct_tags.extend(labels)
            max_scoring_seq = self.virterbiMatrix(line, weights, cw_ct)

            _, pred_tags = list(zip(*max_scoring_seq))
            predicted_tags.extend(pred_tags)
    
        return correct_tags, predicted_tags
        
## MARK - Method to test the Beam Search Algorithm and return the predicted and correct labels
    
    def testBeamSearchPerceptron(self, corpus, cw_ct, weights, beamWidth):
        correct_tags = []
        predicted_tags = []
                
        for line in corpus:
    
            words, labels = list(zip(*line))
            correct_tags.extend(labels)
            max_scoring_seq = self.beamSearch(line, weights, cw_ct, beamWidth)
            
            _, pred_tags = list(zip(*max_scoring_seq))
            predicted_tags.extend(pred_tags)
    
        return correct_tags, predicted_tags
    
## MARK - Method to evaluate the Perceptron
    
    def evaluate(self, correct_tags, predicted_tags):
        f1 = f1_score(correct_tags, predicted_tags, average='micro', labels = ["O", "PER", "LOC", "ORG", "MISC"])
        print("F1 Score: ", round(f1, 5))
    
    
    
if __name__ == '__main__':
    print("Program Initalised")
    config = CommandLine()
    features = FeatureExtraction(config.trainDataset)
    
    if config.testType == "-v":
        perceptron = Perceptron(config.trainDataset, config.testDataset, 10, features.cw_cl_counts, 12, True) 
    else:
        perceptron = Perceptron(config.trainDataset, config.testDataset, 10, features.cw_cl_counts, 12, False) 
    
     
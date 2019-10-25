#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, random
import numpy as np
from operator import itemgetter
from itertools import product
from sklearn.metrics import f1_score

#==============================================================================
## MARK - Class to handle the initialisation of the script in commandline

class CommandLine:
    
## MARK - Initialisation method, get the file path of the datasets, format datasets correctly
    
    def __init__(self):
        # Retrieve command line arguments
        trainFileName = sys.argv[1]
        testFileName = sys.argv[2]
        
        trainDataset = self.load_dataset_sents(trainFileName)
        testDataset = self.load_dataset_sents(testFileName)
        self.trainDataset = np.array(trainDataset).tolist()
        self.testDataset = np.array(testDataset).tolist()
        
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
#        each function returns a sorted dictionary of features and their counts in 
#        the corpus, {feature: count} where a threshold can be used to remove any
#        features that have a count of less than the threshold to speed up computation
#        time. 

class FeatureExtraction:
    
## MARK - Initialisation method, to generate the count dictionaries for each feature 
    
    def __init__(self, corpus, threshold):
        self.cw_cl_counts = self.currentWordCurrentLabel(corpus, threshold)
        self.pl_pl_cl_counts = self.trigramPL_PL_CL(corpus)
        self.pl_cl_counts = self.previousLabelCurrentLabel(corpus)
        self.pw_pl_cw_cl_counts = self.pw_pl_cw_cl(corpus)
        
## MARK - Method to return the feature counts for current word, current label pairs 
#        in the corpus. {'x1_y1': count, 'x2_y2': count}. Used to define Phi_1.
    
    def currentWordCurrentLabel(self, corpus, threshold):
        cw_cl_counts = {}
        
        for line in corpus:
            for pair in line:
                term = pair[0] + '_' + pair[1]
                if term in cw_cl_counts: 
                    cw_cl_counts[term] += 1
                else:
                    cw_cl_counts[term] = 1
         
        sortedDictionary = { key: value for key, value in sorted(cw_cl_counts.items(), key = itemgetter(1), reverse = True) if value >= threshold}
        return sortedDictionary
  
## MARK - Method to return the feature counts for previous label, current label pairs
#        in the corpus. {'y0_y1': count, 'y1_y2': count}. Used to define phi_2.

    def previousLabelCurrentLabel(self, corpus):
        pl_cl_counts = {}
        previousLabel = 'NONE'
        
        for line in corpus:
            previousLabel = 'NONE'
            for pair in line:
                term = previousLabel + '_' + pair[1]
                if term in pl_cl_counts: 
                    pl_cl_counts[term] += 1
                else:
                    pl_cl_counts[term] = 1
                
                previousLabel = pair[1]
                
        sortedDictionary = { key: value for key, value in sorted(pl_cl_counts.items(), key = itemgetter(1), reverse = True)}
        return sortedDictionary
    
## MARK - Method to return the feature counts for the label trigram, two previous 
#        label, previous label and current label. {'y0_y1_y2': count, 'y1_y2_y3': count}.
#        Used to define optional task of phi_3. This feature type is used to further 
#        examine the structure of labels within a sentence and whether previous labels
#        determine the current label. 

    def trigramPL_PL_CL(self, corpus):
        pl_pl_cl_counts = {}
        prevPrevLabel = 'NONE'
        previousLabel = 'NONE'
        
        for line in corpus:
            prevPrevLabel = 'NONE'
            previousLabel = 'NONE'
            for pair in line:
                term = prevPrevLabel + '_' + previousLabel + '_' + pair[1]
                if term in pl_pl_cl_counts: 
                    pl_pl_cl_counts[term] += 1
                else:
                    pl_pl_cl_counts[term] = 1
                
                prevPrevLabel = previousLabel
                previousLabel = pair[1]
                    
        sortedDictionary = { key: value for key, value in sorted(pl_pl_cl_counts.items(), key = itemgetter(1), reverse = True)}
        return sortedDictionary
    
## MARK - Method to return the feature counts for word/label bigram, previous 
#        word, previous label and current word, current label. 
#        {'x0_y0_x1_y1': count, 'x1_y1_x2_y2': count}. Used to define optional task of phi_4.
#        This feature has been selected to picture the relationship between the labels of 
#        specific previous words and the label of the current word. 

    def pw_pl_cw_cl(self, corpus):
        pw_pl_cw_cl_counts = {}
        previousWord = 'NONE'
        previousLabel = 'NONE'
        
        for line in corpus:
            previousWord = 'NONE'
            previousLabel = 'NONE'
            for pair in line:
                term = previousWord + '_' + previousLabel + '_' + pair[0] + '_' + pair[1]
                if term in pw_pl_cw_cl_counts:
                    pw_pl_cw_cl_counts[term] += 1
                else:
                    pw_pl_cw_cl_counts[term] = 1
                
                previousWord = pair[0]
                previousLabel = pair[1]
        
        sortedDictionary = { key: value for key, value in sorted(pw_pl_cw_cl_counts.items(), key = itemgetter(1), reverse = True)}
        return sortedDictionary
        
    
#==============================================================================
## MARK - Class used to generate the required phi terms, phi function takes as
#        the input the words, real labels of a sentence and corresponding counts
#        for that phi type in the whole data corpus and returns a dictionary with
#        the feature key counts in that sentence.

class FeatureTypes:
    
## MARK - Initialisation method, to determine the required phi and create define
#    that phi for a sentence. The phi is determined by phiType, where is this equals
#    1 then phi_1 is calculated. If phiType equals 1234, then all four phis are calculated 
#    and summed to give phi_1_2_3_4.
    
    def __init__(self, words, labels, phiType, cw_cl_counts, pl_cl_counts, pl_pl_cl_counts, pw_pl_cw_cl_counts):
        if phiType == 1:
            self.phi_1 = self.phi_1(words, labels, cw_cl_counts)
        elif phiType == 12:
            self.phi1 = self.phi_1(words, labels, cw_cl_counts)
            phi2 = self.phi_2(words, labels, pl_cl_counts)
            self.phi1_phi2 = {**self.phi1, **phi2}
        elif phiType == 1234:
            self.phi1 = self.phi_1(words, labels, cw_cl_counts)
            phi2 = self.phi_2(words, labels, pl_cl_counts)
            self.phi1_phi2 = {**self.phi1, **phi2}
            phi3 = self.phi_3(words, labels, pl_pl_cl_counts)
            phi4 = self.phi_4(words, labels, pw_pl_cw_cl_counts)
            self.phi1_phi2_phi3_phi4 = {**self.phi1, **phi2, **phi3, **phi4}
        
## MARK - Method to generate phi_1, which is based on current words and current labels
        
    def phi_1(self, words, labels, cw_cl_counts):
        phi_1 = {}
        
        for iteration in range(len(words)):
            term = words[iteration] + '_' + labels[iteration]
            if term in cw_cl_counts:
                if term not in phi_1:
                    phi_1[term] = 1
                else:
                    phi_1[term] += 1
        
        return phi_1
        
## MARK - Method to generate phi_2, which is based on the previous label and the current label
    
    def phi_2(self, words, labels, pl_cl_counts):
        phi_2 = {}
        previousLabel = 'NONE'
        
        for iteration in range(len(words)):
            term = previousLabel + '_' + labels[iteration]
            if term in pl_cl_counts:
                if term not in phi_2:
                    phi_2[term] = 1
                else:
                    phi_2[term] += 1
                
            previousLabel = labels[iteration]
            
        return phi_2
        
## MARK - Method to generate phi_3, which is based on the two prior labels and the current label
    
    def phi_3(self, words, labels, pl_pl_cl_counts):
        phi_3 = {}
        prevPrevLabel = 'NONE'
        previousLabel = 'NONE'
        
        for iteration in range(len(words)):
            term = prevPrevLabel + '_' + previousLabel + '_' + labels[iteration]
            if term in pl_pl_cl_counts:
                if term not in phi_3:
                    phi_3[term] = 1
                else:
                    phi_3[term] += 1
            prevPrevLabel = previousLabel
            previousLabel = labels[iteration]
            
        return phi_3
    
## MARK - Method to generate phi_4, which is based on the previous word/label pair and 
#        current word/label pair
    
    def phi_4(self, words, labels, pw_pl_cw_cl_counts):
        phi_4 = {}
        previousWord = 'NONE'
        previousLabel = 'NONE'
        
        for iteration in range(len(words)):
            term = previousWord + '_' + previousLabel + '_' + words[iteration] + '_' + labels[iteration]
            if term in pw_pl_cw_cl_counts:
                if term not in phi_4:
                    phi_4[term] = 1
                else:
                    phi_4[term] += 1
            previousWord = words[iteration]
            previousLabel = labels[iteration]
                
        return phi_4


#==============================================================================
## MARK - Class defining the Structured Perceptron used to define the weights for
#        a model, which are then used to predict a tag label sequence for a sentence

class Perceptron:
    
    weights = {}
    averageWeights = {}
    previousError = 1.0
    currentError = 0.0
    seedCounter = 0
    
## MARK - Method to initialise the Perceptron by generating the blank weights
#        dictionary and then training the perceptron over the desired number of iterations
#        and returning the averaged weights.
    
    def __init__(self, iterations, counts, dataset, phi):
        for count in counts:
            self.weights = self.generateBlankWeights(self.weights, count)
            
        for iteration in range(iterations):
            print("Training Iteration: " + str(self.seedCounter + 1))
            randomisedDataset = self.randomiseDataset(dataset)
            updatedWeights = self.train(randomisedDataset, self.weights, phi)
            
            for key, value in updatedWeights.items():
                self.weights[key] += value
        
        self.averageWeights = { key: round((value/self.seedCounter), 2) for key, value in sorted(self.weights.items(), key = itemgetter(1), reverse = True)}
        
## MARK - Method to return the blank weights from the counts of a feature type
    
    def generateBlankWeights(self, weights, counts):
        for term in counts:
            if term in weights:
                continue
            else:
                weights[term] = 0
        
        return weights
    
## MARK - Method to randomise the order of the dataset, to stop the perceptron 
#        becoming skewed to a specific order of terms
    
    def randomiseDataset(self, dataset):
        random.seed(self.seedCounter)
        random.shuffle(dataset)
        
        self.seedCounter += 1
        return dataset    
        
## MARK - Method to train the weights by searching through all possible combinations
#        of tag sequence for a sentence and returning the most probable sequence.
#        If this sequence is not equal to the real sequence then the weights need
#        to be altered accordingly. Subtracting from terms in the dictionary that are
#        false postives and adding to terms that are false negatives. 
        
        
    def train(self, trainingData, weights, phi):
        standardLabels = ["PER", "LOC", "ORG", "MISC", "O"]
        
        for line in trainingData:
            sentenceWords = []
            sentenceLabels = []
            labelSequences = product(standardLabels, repeat = len(line))
            
            for word, label in line:
                sentenceWords.append(word)
                sentenceLabels.append(label)
                
            bestPhi, bestLabels = self.predict(sentenceWords, sentenceLabels, labelSequences, weights, phi)
                
            truePhis = FeatureTypes(sentenceWords, sentenceLabels, phi, extraction.cw_cl_counts, extraction.pl_cl_counts, extraction.pl_pl_cl_counts, extraction.pw_pl_cw_cl_counts)
            
            if phi == 1:
                truePhi = truePhis.phi_1
            elif phi == 12:
                truePhi = truePhis.phi1_phi2
            elif phi == 1234:
                truePhi = truePhis.phi1_phi2_phi3_phi4
            
            if sentenceLabels != bestLabels:
                for term, count in truePhi.items():
                    if term in weights:
                        weights[term] += truePhi[term]
                    else:
                        weights[term] = truePhi[term]
            
                for term, count in bestPhi.items():
                    if term in weights:
                        weights[term] += 0 - bestPhi[term]
                    else:
                        weights[term] = 0 - bestPhi[term]     
       
        return weights
    
## MARK - Method to predict the label sequence for a sentence from the weights
    
    def predict(self, sentenceWords, sentenceLabels, labelSequences, weights, phi):
        greatestProb = 0
                
        for labels in labelSequences:
            y_label = 0
            phis = FeatureTypes(sentenceWords, labels, phi, extraction.cw_cl_counts, extraction.pl_cl_counts, extraction.pl_pl_cl_counts, extraction.pw_pl_cw_cl_counts)
            
            if phi == 1:
                self.phi_sentence = phis.phi_1
            elif phi == 12:
                self.phi_sentence = phis.phi1_phi2
            elif phi == 1234:
                self.phi_sentence = phis.phi1_phi2_phi3_phi4
            
            for key, value in self.phi_sentence.items():
                if key in weights:
                    y_label += weights[key] * value
            
            if y_label >= greatestProb:
                greatestProb = y_label
                bestPhi = self.phi_sentence
                bestLabels = labels
    
        return bestPhi, bestLabels
    
    
#==============================================================================
## MARK - Class used to test the model built by the Structured Perceptron, takes
#        an unseen dataset of sentences and predicts a tag label sequence. The 
#        models are evaluated using the F1 score because the dataset is imbalanced.
        
class Test:
    
## MARK - Initialisation method to return the correct and predicted labels for 
#        each sentence in a corpus and evaluate the predictions by calculating the f1 
#        score from these returned results.
    
    def __init__(self, weights, testData, phiType):
        y_true, y_predicted = self.test(weights, testData, phiType)
        self.score = self.scores(y_true, y_predicted)
        self.top = self.getMostPositiveTerms(weights)
        
## MARK - Method to test each model by predicting the labels of sentences using the 
#        pre-determined weights
    
    def test(self, weights, testData, phi):
        y_true = []
        y_predicted = []
        standardLabels = ['PER', 'LOC', 'ORG', 'MISC', 'O']
        
        for line in testData:
            sentenceWords = []
            sentenceLabels = []
            labelSequences = product(standardLabels, repeat = len(line))
        
            for word, label in line:
                sentenceWords.append(word)
                sentenceLabels.append(label)
                      
            prediction, labels = Perceptron.predict(self, sentenceWords, sentenceLabels, labelSequences, weights, phi) 
                        
            for index in range(len(labels)):
                y_true.append(sentenceLabels[index])
                y_predicted.append(labels[index])
                
        return y_true, y_predicted
        
## MARK - Method to return the f1 score of the label predictions.
    
    def scores(self, y_true, y_predicted):
        f1_micro = f1_score(y_true, y_predicted, average = 'micro', labels = ['ORG', 'MISC', 'PER', 'LOC', 'O'])
        return f1_micro

## MARK - Method to return the most positive terms for each label.

    def getMostPositiveTerms(self, weights):
        mostPosPER = []
        mostPosLOC = []
        mostPosORG = []
        mostPosMisc = []
        mostPosO = []
        mostPositive = {}
        
        for key, value in weights.items():
            if '_PER' in key and len(mostPosPER) < 10:
                mostPosPER.append(key)
            if '_LOC' in key and len(mostPosLOC) < 10:
                mostPosLOC.append(key)
            if '_ORG' in key and len(mostPosORG) < 10:
                mostPosORG.append(key)
            if '_MISC' in key and len(mostPosMisc) < 10:
                mostPosMisc.append(key)
            if '_O' in key and '_ORG' not in key and len(mostPosO) < 10:
                mostPosO.append(key)
                
            if (len(mostPosPER) and len(mostPosLOC) and len(mostPosORG) and len(mostPosMisc) and len(mostPosO)) == 10:
                mostPositive['PER'] = mostPosPER
                mostPositive['LOC'] = mostPosLOC
                mostPositive['ORG'] = mostPosORG
                mostPositive['MISC'] = mostPosMisc
                mostPositive['O'] = mostPosO
        
        return mostPositive
    
#==============================================================================
## MARK - Used to run the script from the commandline
        
if __name__ == '__main__':
    
    print("Program Initalised")
    config = CommandLine()
    extraction = FeatureExtraction(config.trainDataset, 3)
    phi_1_perceptron = Perceptron(10, [extraction.cw_cl_counts], config.trainDataset, 1)
    phi_1_test = Test(phi_1_perceptron.averageWeights, config.testDataset, 1)
    print("F1 Score for the Phi_1 model is: " + str(phi_1_test.score) + ", using a threshold of 3 and after 10 iterations.")
    print("The most positive terms are: ")
    print(phi_1_test.top)
    phi_2_perceptron = Perceptron(10, [extraction.cw_cl_counts, extraction.pl_cl_counts], config.trainDataset, 12)
    phi_2_test = Test(phi_2_perceptron.averageWeights, config.testDataset, 12)
    print("F1 Score for the Phi_1 + Phi_2 model is: " + str(phi_2_test.score) + ", using a threshold of 3 and after 10 iterations.")
    print("The most positive terms are: ")
    print(phi_2_test.top)
    phi_3_4_perceptron = Perceptron(10, [extraction.cw_cl_counts, extraction.pl_cl_counts, extraction.pl_pl_cl_counts, extraction.pw_pl_cw_cl_counts], config.trainDataset, 1234)
    phi_3_4_test = Test(phi_3_4_perceptron.averageWeights, config.testDataset, 1234)
    print("F1 Score for the Phi_1 + Phi_2 + Phi_3 + Phi_4 model is: " + str(phi_3_4_test.score) + ", using a threshold of 3 and after 10 iterations.")
    print("The most positive terms are: ")
    print(phi_3_4_test.top)
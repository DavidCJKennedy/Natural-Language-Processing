#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, inspect
import re, string
from operator import itemgetter
 

#==============================================================================
## MARK - Class to handle the initialisation of the script in commandline

class CommandLine:
    
## MARK - Initialisation method, get the file path of the datasets, format datasets correctly
    
    def __init__(self):
        # Retrieve command line arguments
        dataCorpusFileName = sys.argv[1]
        questionFileName = sys.argv[2]
        
        # Obtain the path to the current directory 
        # It is assumed that the data folders are also in this directory
        currentDirectory = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        
        # Open the Question and Data Corpus files
        dataCorpusFile = open(currentDirectory + '/' + dataCorpusFileName, 'r')
        questionFile =  open(currentDirectory + '/' + questionFileName, 'r')
        
        # Read the open Data corpus file and pre-process to remove punctuation and convert to lowercase
        preprocessedCorpus = self.preprocessCorpus(dataCorpusFile.read())
        # Replace all new line symbols with <s>, each line is a different sentence.
        # Adding <s> allows for the sentences to be joined into a long array with sentence structure
        # retained
        self.corpus = preprocessedCorpus.replace('\n', '<s> ').split()
                
        # Read in the questions, each question is on seperate lines. Hence readlines()
        questions = questionFile.readlines()     
        
        # Process the question file to obtain a malleable question used in testing 
        # the language models. And the raw question without the answer candidates used
        # to present the results of the language models
        self.questions, self.justQuestions = self.processQuestions(questions)
        
        # Close the files to reduce stress on computer memory
        dataCorpusFile.close()
        questionFile.close()
    
## MARK - Method the preprocess the corpus to lowercase and remove punctuation
        
    def preprocessCorpus(self, corpus):
        # Create the regular expression where if any punctuation is found 
        # the punctuation is replaced with an empty string
        regularExpression = re.compile('[%s]' % re.escape(string.punctuation))
        # Return the processed corpus in lowercase
        return regularExpression.sub('', corpus).lower()
    
## MARK - Method the preprocess the questions to malleable forms   

    def processQuestions(self, questions):
        # Init blank arrays for the new question forms
        processedQuestions = []
        justQuestions = []
        # As in the corpus processing, create the regular expression to remove 
        # punctuation
        regularExpression = re.compile('[%s]' % re.escape(string.punctuation))
        
        # Iterate through each question
        for question in questions:
            # Ignore any lines that are not questions
            if question != '/n':
                # Split the question into the query text and the possible answers
                split = question.rsplit(':', 1)
                # Add the query to the questions array
                justQuestions.append(split[0])
                
                # Replace the query symbol target with 'xxqqxx' as to avoid accidental
                # removal when replacing any punctuation
                # 'xxqqxx' used as likelyhood of being in legitimate query text
                # is as low as ____.
                split[0] = split[0].replace('____', 'xxqqxx')
                # Remove all punctuation from the query and convert to lowercase
                split[0] = regularExpression.sub('', split[0]).lower()
                # Remove the / between the two possible answers
                split[1] = split[1].replace('/', ' ')
                # Remove new line symbol
                split[1] = split[1].replace('\n', '')
                
                # Split query into individual symbols
                question = split[0].split()
                # Add the <s> begining and sentence end tokens
                question.insert(0, '<s>')
                question.append('<s>')
                
                # Split the possible answers into individual symbols
                possibleAnswers = split[1].split()
                # Add malleable question to array
                processedQuestions.append([question, possibleAnswers])
        
        # Return processed question types
        return processedQuestions, justQuestions


#==============================================================================
## MARK - Class to build the Unigram language model

class Unigram:
    # Initialise blank dictionary to store unigram in class
    unigram = {}
    
## MARK - Initialisation method, commence building unigram from data corpus
    
    def __init__(self, corpus):
        self.unigram, self.symbolCounts = self.buildUnigram(corpus)
    
## MARK - Build Unigram method 

    def buildUnigram(self, corpus):
        # Init the total corpus symbol count and blank unigram dictionary
        # a new unigram dictionary is created here to avoid global variable overheads
        # corpusSymbolCount is required to calculate probability of each symbol
        corpusSymbolCount = 0
        unigram = {}
        
        # Iterate through each symbol in the corpus
        for symbol in corpus:
            # If symbol represents start or end of a line, this is not account for in overall symbol count
            if symbol != '<s>':
                # Add one to symbol count
                corpusSymbolCount += 1
                # If symbol is not already in the unigram model
                if symbol not in unigram: 
                    # A new symbol has been found so add as new entry to the unigram
                    unigram[symbol] = 1
                else:
                    # Symbol already exists so add one to the count
                    unigram[symbol] += 1
            
            # The symbol represents the start or end of a sentence
            else:
                # Very likely that <s> already is in the unigram
                # Check if in unigram first to reduce overheads
                if '<s>' in unigram: 
                    # Update individual count for <s>
                    unigram['<s>'] += 1
                else:
                    unigram['<s>'] = 1
        
        # Create the symbol counts variable that stores every symbol from the corpus
        # including <s>, used in smoothedBigram
        symbolCounts = unigram
        # Create a new instance of the unigram as to stop all instances being updated when
        # <s> is removed
        newInstance = dict(unigram)
        # Remove <s> from unigram as <s> is not a valid answer in a query
        del newInstance['<s>']
        
        # Term symbol counts into the probability of a symbol occuring
        # Order symbols from most common to least common
        newInstance = { key: value / corpusSymbolCount for key, value in sorted(newInstance.items(), key = itemgetter(1), reverse = True)}
        
        # Return unigram and symbol count dictionaries
        return newInstance, symbolCounts


#==============================================================================
## MARK - Class to build the Bigram language model

class Bigram:
    # Initialise blank dictionary to store unigram in class
    bigram = {}
    
## MARK - Initialisation method, commence building bigram from data corpus and symbolcounts
    
    def __init__(self, corpus, symbolCounts):
        self.bigram = self.buildBigram(corpus, symbolCounts)

## MARK - Build bigram method 

    def buildBigram(self, corpus, symbolCounts):
        # Init the blank bigram dictionary
        # a new bigram dictionary is created here to avoid global variable overheads
        bigram = {}

        # Iterate through the number of symbols in the corpus, retrieving the index
        # len -1 as the final bigram takes up the final two symbols in the corpus
        for iteration in range((len(corpus) - 1)):
            # Get the first half of the bigram
            symbol = corpus[iteration]
            # Get the immediate following symbol to make the other half of the bigram
            nextSymbol = corpus[iteration + 1]
        
            # If symbol is not already in the bigram model
            if symbol not in bigram:
                # A new symbol has been found so add as new entry to the bigram
                # Where the first symbol is the key, the following symbol is added as 
                # a value with a count stored with it.
                # {key: firstSymbol, value: {key: secondSymbol, value: count}}
                bigram[symbol] = {nextSymbol: 1}
            else:
                # Symbol already exists, so check if the second symbol exists as 
                # a value for that bigram key symbol
                if nextSymbol in bigram[symbol]:
                    # If it does then update the count for the second symbol
                    bigram[symbol][nextSymbol] += 1
                else:
                    # Else add the new second symbol to the bigram
                    bigram[symbol].update({nextSymbol: 1})
        
        # Iterate through each key in the bigram
        for symbol, nextSymbols in bigram.items():
            # Retrieve the number of times that key symbol occurs in the full corpus
            symbolCount = symbolCounts[symbol]
            # Generate the probabilities for the second term occuring after the first in the bigram
            bigram[symbol] = { key: value / symbolCount for key, value in sorted(nextSymbols.items(), key = itemgetter(1), reverse = True)}
    
        # Return the complete bigram
        return bigram


#==============================================================================
## MARK - Class to Test the language models

class Test:
    # Initialise a question counter variable to keep track of which query is being using by the models
    # Required for displaying results
    questionCounter = 0
    
## MARK - Initialisation method, iterate through each question and apply the language models to guess
# the missing query term
    
    def __init__(self, processedQuestions, originalQuestions, unigramModel, bigramModel, symbolCounts):        
        for question in processedQuestions:
            unigramAnswer = self.testUnigram(question, unigramModel)
            bigramAnswer = self.testBigram(question, bigramModel)
            smoothedBigramAnswer = self.testSmoothedBigram(question, bigramModel, symbolCounts) 
            
            # Display the results for each question
            self.displayResults(unigramAnswer, bigramAnswer, smoothedBigramAnswer, originalQuestions, self.questionCounter)
            # Move the counter onto the next question
            self.questionCounter += 1
            
## MARK - Method to test the unigram model and return guess for missing query term
                
    def testUnigram(self, question, model):
        # Initialise probability to -1 because a guess will always have greater
        # probability than this
        probability = -1
        guess = ""
        
        # Get the potential answers for the missing term in the query
        answers = question[1]
        # Iterate through each potential answer
        for answer in answers:
            # Check whether the answer exists in the unigram, if not then it has 
            # 0 probability and is ignored
            if answer in model:
                # Retrieve the probability for the potential answer
                answerProbability = model[answer]
                # Determine if the probability is the highest achieved so far
                if answerProbability > probability:
                    # If the probability is the highest then set the possible answer as the
                    # current guess and update the current highest probability
                    probability = answerProbability
                    guess = answer
                else:
                    # Not the highest probability so continue to next potential answer
                    continue
        
        # Return highest probability guess for the missing query term
        return guess
                   
## MARK - Method to test the bigram model and return guess for missing query term
            
    def testBigram(self, question, model):
        # Initialise probability to -1 because a guess will always have greater
        # probability than this
        probability = -1
        # Intialise individual term probability to 0 and guess to blank
        termProbability = 0
        guess = "" 
        
        # Get the query and potential answers from the question
        query = question[0]
        answers = question[1]
        # Locate the index of the missing term in the query
        missingSymbolIndex = query.index('xxqqxx')
        
        # Retrieve the symbol before the missing term
        beforeSymbol = query[missingSymbolIndex - 1]
        # Retrieve the symbol after the missing term
        afterSymbol = query[missingSymbolIndex + 1]
        
        # Find the symbol before the missing term in the bigram
        if beforeSymbol in model:
            # Retrieve all possible following terms
            beforeSymbolDict = model[beforeSymbol]
            
            # Iterate through each of the potential answers
            for answer in answers:
                # Check if the answer is a value in the bigram, if it is not
                # then the answer is ignored because it has 0 probability
                if answer in beforeSymbolDict:
                    # Get the probability of this potential answer occuring in the
                    # bigram
                    termProbability = beforeSymbolDict[answer]
                
                # Now we need to determine whether the symbol following the missing term
                # in the query is possible given the potential answer.
                # The is done by consulting the next bigram in the sequence
                if answer in model:
                    # Retrieve all possible following terms
                    answerSymbolDict = model[answer]
                    # Check if the following symbol is a value in the bigram
                    # If it is not the answer is ignored as the bigram has 0 probability
                    if afterSymbol in answerSymbolDict:
                        # Calculate the probability score for the potential answer 
                        # Given that next symbol in the sentence is possible 
                        score = self.harmonicMean(termProbability, answerSymbolDict[afterSymbol])
                        # Determine if the score is the highest achieved so far
                        if score > probability:
                            # If the score is the highest then set the possible answer as the
                            # current guess and update the current highest score
                            probability = score
                            guess = answer
          
        # Return highest probability guess for the missing query term
        return guess
    
## MARK - Method to test the smoothed bigram model and return guess for missing query term.
#  The smoothed bigram model is also constucted here, not all terms in the bigram 
#  need to be smoothed, just the terms to be assessed this reduces computation time.
#  Smoothing is the process of reducing the weight of common terms and adding some 
#  probability to terms that were not encountered in the data corpus.

    ## NOTE ~ +1 laplace has not been used as this method is poor. Instead terms that 
    #  occur less frequently in overall dataset but do not appear in the bigram should
    #  be weighted higher. Terms that are common in the dataset should be weighted down.
    #  Doing this stops a symbol such as "the" overpowering other symbols, giving all 
    #  potential symbols even those not encounted in the fixed data corpus a fair chance.
    
    def testSmoothedBigram(self, question, model, symbolCounts):
        # Initialise probability to -1 because a guess will always have greater
        # probability than this
        probability = -1
        # Intialise individual term probability to 0 and guess to blank
        termProbability = 0
        guess = ""
        
        # Get the query and potential answers from the question
        query = question[0]
        answers = question[1]
        # Locate the index of the missing term in the query
        missingSymbolIndex = query.index('xxqqxx')
        
        # Retrieve the symbol before the missing term
        beforeSymbol = query[missingSymbolIndex - 1]
        # Retrieve the symbol after the missing term
        afterSymbol = query[missingSymbolIndex + 1]
        
        # Find the symbol before the missing term in the bigram
        if beforeSymbol in model:
            # Retrieve all possible following terms
            beforeSymbolDict = model[beforeSymbol]
            
            # Iterate through each of the potential answers
            for answer in answers:
                # Check if the answer is a value in the bigram
                if answer in beforeSymbolDict:
                    # Get the probability of this potential answer occuring in the
                    # bigram
                    termProbability = beforeSymbolDict[answer]
                else:
                    # If the potential answer has not been encountered in the bigram
                    # generate a probability for it. This ensures the probabilty is not 0.
                    # If probability is 0 then this potential answer would be immediately
                    # discounted.
                    termProbability = self.calculateKSeenValue(answer, symbolCounts)
                
                # Now we need to determine whether the symbol following the missing term
                # in the query is possible given the potential answer
                # Find the potential answer in the bigram
                if answer in model:
                    # Retrieve all possible following terms
                    answerSymbolDict = model[answer]
                    # Check if the following symbol is a value in the bigram
                    if afterSymbol in answerSymbolDict:
                        # Calculate the probability score for the potential answer 
                        # Given that next symbol in the sentence is possible 
                        score = self.harmonicMean(termProbability, answerSymbolDict[afterSymbol])
                        # Determine if the score is the highest achieved so far
                        if score > probability:
                            # If the score is the highest then set the possible answer as the
                            # current guess and update the current highest score
                            probability = score
                            guess = answer
                    else:
                        # The following symbol is not in the bigram so a probability must be generated for it
                        score = self.harmonicMean(termProbability, self.calculateKSeenValue(answer, symbolCounts))
                        # Determine if the score is the highest achieved so far
                        if score > probability:
                            # If the score is the highest then set the possible answer as the
                            # current guess and update the current highest score
                            probability = score
                            guess = answer
        
        # Return highest probability guess for the missing query term
        return guess
        
## MARK - Method to calculate the harmonic mean between two probabilities. This penalises a system
#  Where there is a big difference in the two probabilities. The best guess is one where the probability of both 
#  bigrams is good, not just one. The harmonic mean highlights this.
                
    def harmonicMean(self, probability1, probability2):
        if probability1 == 0:
            return 0
        elif probability2 == 0:
            return 0
        else:
            # Calculate the harmonic mean and return
            return ((2 * probability1 * probability2) / (probability1 + probability2))
        
## MARK - Method to calculate the probability of a symbol that is not present in a bigram 

    def calculateKSeenValue(self, specificSymbol, symbolCounts):
        # Retrieve total count for the symbol in the corpus
        countForSpecificSymbol = symbolCounts[specificSymbol]
        
        # Boosts the value of rare symbols and reduces value of common symbols
        kValue = (1 / (countForSpecificSymbol * countForSpecificSymbol))
        return kValue
    
## MARK - Method to display the results of the langauge models
    
    def displayResults(self, unigramAnswer, bigramAnswer, smoothedBigram, questions, counter):
        # Get the question that was used to test the models
        question = questions[counter]
        # Print out the question with highlighted missing query term
        print(question)
        # Print out the returned guess for the missing term from each model
        print("~ Unigram Answer: " + unigramAnswer + ". Bigram Answer: " + bigramAnswer + ". Smoothed Bigram Answer: " + smoothedBigram + ".")
        
        
#==============================================================================
## MARK - Used to run the script from the commandline
        
if __name__ == '__main__':
    
    print("Program Initalised")
    # Intialise the code file variables using the Commandline Class
    config = CommandLine()
    # Build the Unigram
    unigram = Unigram(config.corpus)
    print("Unigram created")
    # Build the Bigram
    bigram = Bigram(config.corpus, unigram.symbolCounts)
    print("Bigram created")
    # Test the language models
    test = Test(config.questions, config.justQuestions, unigram.unigram, bigram.bigram, unigram.symbolCounts)
    
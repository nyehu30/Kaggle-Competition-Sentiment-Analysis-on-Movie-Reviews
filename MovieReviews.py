##########################
###### READ IN FILE ######
##########################
# python


import csv as csv
import numpy as np
import os
import nltk

os.getcwd()
os.chdir('/n9e/Desktop')
os.listdir('.')
os.chdir('Desktop')
os.listdir('.')
os.chdir('Sentiment Analysis on Movie Reviews')

csvreader =  csv.reader(open('train.tsv', 'rb'), delimiter = '\t')
header = csvreader.next() 

data = []
for row in csvreader:
    data.append(row)  

print data[:10] 
   
data = np.array(data) # an array of string      


################################################
###### Multinomial Naive Bayes - Baseline ######
################################################

#
# P(Class?|Document) = P(Document|Class)(likelihood) * P(Class)(prior) / P(Document)(it's the same for each document, so we choose to ignore it)
# Class = 0, 1, 2, 3, 4
# Document = "Bag of words" in each phrase
# We use the word appears in the test set and then use the training set to calculate it
# So, we need to calculate #1 P(Class)
#                          #2 P(Document|Class) 
#                          #3 P(Class?|Document)
# 
# Step 1. P(Class = 0, 1, 2, 3, 4) = Count of a certain class / The total count of all classes 
#    
# Step 2. Assume the occurence of each word is independent of the other's 
# P(Document|Class = 0, 1, 2, 3, 4) = P(Word1|Class = 0, ..., 4) * P(Word2|Class 0, ... , 4) * ... P(Wordn|Class = 0, ..., 4)  
# 
# ex. P(Word1|Class = 0) = (count of word1 in class 0 + 1) / (count of all words + count of vocabulary)
#
# Adding 1 to the numerator and vocabulary size to the denominator is to avoid # zero count of new words in a training set
#
# We get a bunch of probabilities for each word from the training set
#
# Step 3. P(Class?|Document in Test Set) = Max (P(Dument|Class = 0, ... , 4) * P(Class = 0, ..., 4))
# 
   
######################    
###### P(Class) ######
######################

sentiment = data[0::, 3]
phrase = data[0::, 2]

total_count_of_class = sentiment.size


classDist = nltk.FreqDist(word for word in sentiment)
classProbability = {}

# Probabilities of different classes
for key in classDist:
     classProbability[key] = float(classDist[key]) / float(total_count_of_class)  
     
print classProbability
     
classDist.plot(5, cumulative=False)
       
##########################
###### Tokenization ######
##########################

tokens = []

for i in range(phrase.size):
    tokens.append(nltk.word_tokenize(phrase[i].lower()))
    
# Flatten a nested list
def flatten(l):  
    for el in l:  
        if hasattr(el, "__iter__") and not isinstance(el, basestring):  
            for sub in flatten(el):  
                yield sub  
        else:  
            yield el 

######### Another way to flatten a list

def flatten(lists):
    results = []
    for lst in lists:
        for thing in lst:
            results.append(thing)
    return results        
            
new_token_list = [x for x in flatten(tokens)]  
print new_token_list[0:100] 

# Frequency distributions for all words
wordDist = nltk.FreqDist(word for word in new_token_list)
print wordDist   

# Vocalbulary size 

vocalbularySize = len(wordDist)
wordDist.items()[1:50]

#########################################################
###### P(Word|Class = 0, ..., 4) from the test set ######  
#########################################################

negative_review = []
smwhat_negative_review = []
neutral_review = []
smwhat_positive_review = []
positive_review = []

# Assign different phrases to different classes
for review in data:
    if review[3] == "0": 
        negative_review.append(nltk.word_tokenize(review[2].lower()))
    elif review[3] == "1":
        smwhat_negative_review.append(nltk.word_tokenize(review[2].lower()))
    elif review[3] == "2":
        neutral_review.append(nltk.word_tokenize(review[2].lower()))
    elif review[3] == "3":
        smwhat_positive_review.append(nltk.word_tokenize(review[2].lower()))
    else: 
        positive_review.append(nltk.word_tokenize(review[2].lower()))

######### nested list version #########
classLists = [[1],[2],[3],[4],[5]]

for review in data:
    if review[3] == "0": 
        classLists[0].append(nltk.word_tokenize(review[2].lower()))
    elif review[3] == "1":
        classLists[1].append(nltk.word_tokenize(review[2].lower()))
    elif review[3] == "2":
        classLists[2].append(nltk.word_tokenize(review[2].lower()))
    elif review[3] == "3":
        classLists[3].append(nltk.word_tokenize(review[2].lower()))
    else: 
        classLists[4].append(nltk.word_tokenize(review[2].lower()))

for list in classLists:
    list.pop([0])
     
#######################################
        
for i in range(6):
     print negative_review[i]    
     
for i in range(6):
     print smwhat_negative_review[i]  
     
for i in range(6):
     print neutral_review[i] 
     
for i in range(6):
     print smwhat_positive_review[i]    
     
for i in range(6):
     print positive_review[i] 
 
negative_review_tokens = [x for x in flatten(negative_review)] 
smwhat_negative_review_tokens = [x for x in flatten(smwhat_negative_review)] 
neutral_review_tokens = [x for x in flatten(neutral_review)] 
smwhat_positive_review_tokens = [x for x in flatten(smwhat_positive_review)] 
postive_review_tokens = [x for x in flatten(positive_review)] 

######### nested list version #########

classListsCopy = [[1],[2],[3],[4],[5]]

for i in range(5):  
        tokens = [word for word in flatten(classLists[i])]
        classListsCopy[i].append(tokens)
     
# Word frequency distributions for different classes         

wordDist_neg_review = nltk.FreqDist(negative_review_tokens)
wordDist_neg_review.items()[0:200]

wordDist_smNeg_review = nltk.FreqDist(smwhat_negative_review_tokens)
wordDist_smNeg_review.items()[0:200]

wordDist_neutral_review = nltk.FreqDist(neutral_review_tokens)
wordDist_neutral_review.items()[0:200]

wordDist_smPos_review = nltk.FreqDist(smwhat_positive_review_tokens)
wordDist_smPos_review.items()[0:200]

wordDist_pos_review = nltk.FreqDist()
wordDist_pos_review.items()[0:200]        

######### nested version ##########

#???
         
# Probabilities for each word in different classes         
# Negative Class
total_count_token_negReview =  len(negative_review_tokens)

prob_token_negReview = wordDist_neg_review.copy()
for key in wordDist_neg_review:
     probability = float(wordDist_neg_review[key] + 1)/ float(total_count_token_negReview + vocalbularySize) 
     prob_token_negReview[key] = probability      
print prob_token_negReview.items()[0:10]
                 

# Somewhat Negative Class
total_count_token_smNeg_review = len(smwhat_negative_review_tokens)

prob_token_smNeg_review = wordDist_smNeg_review.copy()
for key in wordDist_smNeg_review:
     probability = float(wordDist_smNeg_review[key] + 1)/ float(total_count_token_smNeg_review + vocalbularySize) 
     prob_token_smNeg_review[key] = probability      
print prob_token_smNeg_review.items()[0:10]
                 
 
# Neutral Class 
total_count_neutral_review = len(neutral_review_tokens)                  
prob_token_neutral_review = wordDist_neutral_review.copy()
for key in wordDist_neutral_review:
     probability = float(wordDist_neutral_review[key] + 1)/ float(total_count_neutral_review + vocalbularySize) 
     prob_token_neutral_review[key] = probability  
print prob_token_neutral_review.items()[0:10]       


# Somewhat Positive Class
total_count_smPos_review = len(smwhat_positive_review_tokens)
prob_token_smPos_review = wordDist_smPos_review.copy()
for key in wordDist_smPos_review:
     probability = float(wordDist_smPos_review[key] + 1)/float(total_count_smPos_review + vocalbularySize) 
     prob_token_smPos_review[key] = probability   
print prob_token_smPos_review.items()[0:10]


# Positive Class
total_count_posReview = len(postive_review_tokens)
prob_token_pos_review = wordDist_pos_review.copy()
for key in wordDist_pos_review :
     probability = float(wordDist_pos_review[key] + 1)/ float(total_count_posReview + vocalbularySize) 
     prob_token_pos_review[key] = probability   
print prob_token_pos_review.items()[0:10]      
  
################################    
###### P(Class?|Test set) ######
################################  
         
         
csvreader =  csv.reader(open('test.tsv', 'rb'), delimiter = '\t')
header = csvreader.next() 

testData = []
for row in csvreader:
    testData.append(row)  

print testData[:10] 
   
testData = np.array(testData) 
         
testPhrase = testData[0::,2]     


testTokens = []
for i in range(testPhrase.size):
    testTokens.append(nltk.word_tokenize(testPhrase[i].lower()))
                       
new_testTokens = [word for word in flatten(testTokens)]

test_wordDist = nltk.FreqDist(new_testTokens)            

test_keys = test_wordDist.keys()                 
len(test_keys) # = 10030

##################################         
######### Negative Class #########
##################################
         
total_count_token_negReview =  len(negative_review_tokens)
denominatorNeg = float(total_count_token_negReview + vocalbularySize)

prob_token_negReview = test_wordDist.copy()
for key in test_keys:
     if key in wordDist_neg_review.keys():
        prob_token_negReview[key] = float(wordDist_neg_review[key] + 1)/denominatorNeg    
     else:
        prob_token_negReview[key] = 1.0/denominatorNeg  
         
print prob_token_negReview.items()[0:10]         

 
keys_training = []
keys_test = []         
for key in test_keys:
     if key in wordDist_neg_review.keys():
        keys_training.append(key)   
     else:
        keys_test.append(key)
                  
###########################################         
######### Somewhat negative class #########
###########################################

total_count_token_smNeg_review =  len(smwhat_negative_review_tokens)
denominatorSmNeg = float(total_count_token_smNeg_review + vocalbularySize)

prob_token_smNeg_review = test_wordDist.copy()
for key in test_keys:
     if key in wordDist_smNeg_review.keys():
        prob_token_smNeg_review[key] = float(wordDist_smNeg_review[key] + 1)/denominatorSmNeg    
     else:
        prob_token_smNeg_review[key] = 1.0/denominatorSmNeg  
         
print prob_token_smNeg_review.items()[0:100]         

 
smNeg_training_keys = []
smNeg_test_keys = []   
      
for key in test_keys:
     if key in wordDist_smNeg_review.keys():
        smNeg_training_keys.append(key)   
     else:
        smNeg_test_keys.append(key)        


#################################
######### Neutral Class #########
#################################

total_count_neutral_review = len(neutral_review_tokens)
dnominatotNeutral = float(len(neutral_review_tokens) + vocalbularySize) 
                 
prob_token_neutral_review = test_wordDist.copy()
for key in test_keys:
     if key in wordDist_neutral_review.keys():
        prob_token_neutral_review[key] = float(wordDist_neutral_review[key] + 1)/ dnominatotNeutral
     else:
        prob_token_neutral_review[key] = 1.0 / dnominatotNeutral  


print prob_token_neutral_review.items()[0:100] 
 
neutral_training_keys = []
neutral_test_keys = []

for key in test_keys:
    if key in wordDist_neutral_review.keys():
        neutral_training_keys.append(key)
    else:
        neutral_test_keys.append(key)
            


###########################################
######### Somewhat positive class #########
###########################################

total_count_token_smPos_review = len(smwhat_positive_review_tokens)
denominatorSmPos = float(total_count_token_smPos_review + vocalbularySize)

prob_token_smPos_review = test_wordDist.copy()
for key in test_keys:
    if key in wordDist_smPos_review.keys():
        prob_token_smPos_review[key] = float(wordDist_smPos_review[key] + 1) / denominatorSmPos 
    else:
        prob_token_smPos_review[key] = 1.0 / denominatorSmPos 
          
print prob_token_smPos_review.items()[0:100]

smPos_training_keys = []
smPos_test_keys = []

for key in test_keys:
    if key in wordDist_smPos_review.keys():
        smPos_training_keys.append(key)
    else:
        smPos_test_keys.append(key)    


##################################
######### Positive Class #########        
##################################         
         
total_count_posReview = len(postive_review_tokens)
denominatorPos = float(total_count_posReview + vocalbularySize) 

prob_token_pos_review = test_wordDist.copy()
for key in test_keys:
     if key in wordDist_pos_review.keys():
        prob_token_pos_review[key] = float(wordDist_pos_review[key] + 1)/ denominatorPos 
     else:
        prob_token_pos_review[key] = 1.0 / denominatorPos   
print prob_token_pos_review.items()[0:10]           
         
         
pos_training_keys = []
pos_test_keys = []

for key in test_keys:
    if key in wordDist_pos_review.keys():
        pos_training_keys.append(key)
    else:
        pos_test_keys.append(key)      
     
    
    
  
testTokenArray = np.array(testTokens)
 
# Negative Class
prob_phrases_negClass = []
for row in testTokenArray:
    prob_phrase = 1.0
    for word in row:
        prob_phrase *= prob_token_negReview[word]
    prob_phrases_negClass.append(prob_phrase*0.04531590413943355)     
 
# Somewhat negative class
prob_phrases_smNegClass = []
for row in testTokenArray:
    prob_phrase = 1.0
    for word in row:
        prob_phrase *= prob_token_smNeg_review[word]
    prob_phrases_smNegClass.append(prob_phrase*0.1747597078046905)


# Neutral class
prob_phrases_neutralClass = []
for row in testTokenArray:
    prob_phrase = 1.0
    for word in row:
        prob_phrase*= prob_token_neutral_review[word]
    prob_phrases_neutralClass.append(prob_phrase*0.5099448929898757)

       
# Somewhat positive class
prob_phrases_smPosClass = []
for row in testTokenArray:
    prob_phrase = 1.0
    for word in row:
        prob_phrase*= prob_token_smPos_review[word]
    prob_phrases_smPosClass.append(prob_phrase*0.21098936306548763)    
    
# Postive class
prob_phrases_posClass = []
for row in testTokenArray:
    prob_phrase = 1.0
    for word in row:
        prob_phrase*= prob_token_pos_review[word]
    prob_phrases_posClass.append(prob_phrase*0.058990132000512625)    
    
prob_phrases_negClass    
prob_phrases_smNegClass
prob_phrases_neutralClass
prob_phrases_smPosClass
prob_phrases_posClass

prob_all_classes = []
prob_all_classes.append(prob_phrases_negClass)
prob_all_classes.append(prob_phrases_smNegClass)
prob_all_classes.append(prob_phrases_neutralClass)
prob_all_classes.append(prob_phrases_smNegClass)
prob_all_classes.append(prob_phrases_posClass)

    
classAssignment = []        
for j in range(len(prob_all_classes[0])):
    greatestProb = 0 
    for i in range(5):
        if prob_all_classes[i][j] > greatestProb:
            greatestProb = prob_all_classes[i][j]
            index = i
    classAssignment.append(index)           


with open("MultinomialNaiveBayesBaseline.csv", 'wb') as archive_file:
    prediction_file = csv.writer(archive_file)
    for number in classAssignment:
        prediction_file.writerow(['', number]) 


prediction_file = csv.writer(open("sampleSubmission", "wb"))
for number in classAssignment:
    prediction_file.writerow([, number]) 

>>> len(classAssignment)
66292



prediction_file = csv.writer(open("MultinomialNaiveBayesBaseline.csv", "wb"))
prediction_file.writerow(["PhraseId", "Sentiment"]) 

classAssignment = []        
for j in range(len(prob_all_classes[0])):
    greatestProb = 0 
    for i in range(5):
        if prob_all_classes[i][j] > greatestProb:
            greatestProb = prob_all_classes[i][j]
            index = i
    prediction_file.writerow([row[0], index]) 
    

            
 for j in range(len(prob_all_classes[0])):
    greatestProb = 0 
    for i in range(5):
        if prob_all_classes[i][j] > greatestProb:
            greatestProb = prob_all_classes[i][j]
            index = i
    print("Row:  %d, index: %d") % (j, index)     



       
    
for j in range(10):
    print(" ")
    for i in range(5):
        print prob_all_classes[i][j]
        
        
for i in range(10):
    print classAssignment[i]         
                       
#####################################
######## Boolean Naive Bayes ########
#####################################        
# P(Class?|Document) = P(Document|Class) * P(Class)
# Class = 0, 1, 2, 3, 4
# Document = "Bag of non redundent word" in each phrase
#                   

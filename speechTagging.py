# Implementation of natural language processing speech tagging
# Author: Andrew Van Hoveln

import numpy as np
import matplotlib.pyplot as plt

# Method to process input file into usable data
def processfile():
    file = open("tagged_sentences.txt")

    tags = {}

    corpus = []

    wordsAndTags = {}

    for line in file:
        corpus.append([])
        for word_tag in line.split():
            last_slash_index = word_tag.rfind('/')
            word = word_tag[:last_slash_index]
            tag = word_tag[last_slash_index + 1:]

            corpus[len(corpus)-1].append((word,tag))

            if tag not in tags:
                tags[tag] = 1
            else:
                tags[tag] += 1
            
            if word not in wordsAndTags:
                wordsAndTags[word] = {tag:1}
            else:
                if tag not in wordsAndTags[word]:
                    wordsAndTags[word][tag] = 1
                else:
                    wordsAndTags[word][tag] += 1
    
    # Split data into dictionary of tags with keys as frequencies,
    #   wordsAndTags which is a dictionary of every word followed by a subdictionary of each tag followed by its frequency
    #   corpus is just the sentences processed into list
    return wordsAndTags, tags, corpus
                    

# Function to plot data
def plotWordsAndTags(words, tags):
    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Displaying the words graph
    word2 = ['family', 'guy', 'peter', 'griffin']
    values = []

    for word in word2:
        total = 0
        if word in words:
            for tag in words[word]:
                total += words[word][tag]
        values.append(total)
    
    ax1.bar(word2, values, label='Words')
    ax1.set_xlabel('Words')
    ax1.set_ylabel('Counts')    

    # Adding numbers on top of each bar
    for i, value in enumerate(values):
        ax1.text(i, value + 1, str(value), ha='center', va='bottom')

    tagNames = []
    tagCounts = []

    for tag in tags:
        if tag != '##' and tag != '$$':
            tagNames.append(tag)
            tagCounts.append(tags[tag])

    tagCounts, tagNames = zip(*sorted(zip(tagCounts, tagNames), reverse=True))

    ax2.barh(tagNames, tagCounts, align = 'center')

    ax2.set_xlabel('Counts')
    ax2.set_ylabel('Tags')    

    # Displaying the numbers to the right of each bar
    for i, value in enumerate(tagCounts):
        ax2.text(value + 0.5, i, str(value), ha='left', va='center')

    plt.tight_layout()

    plt.show()

# Here we return a dictionary of counts
# It puts counts each combination of tags into a dictionary
def countFollowing(corpus, tags):
    myDict = {}
    for tag1 in tags:
        myDict[tag1] = {}
        for tag2 in tags:
            myDict[tag1][tag2] = 0

    for sentence in corpus:
        for i in range(1, len(sentence)):
            myDict[sentence[i][1]][sentence[i-1][1]] += 1
    
    return myDict

# Here we create our transition probability table for future use
def createTransitionProbTable(corpus, tags):
    table = countFollowing(corpus, tags)

    for tag1 in table:
        for tag2 in table[tag1]:
            table[tag1][tag2] = np.log((table[tag1][tag2] + 1) /(tags[tag2] + len(tags)))

    return table
            
# Emission probability does not need a table.
# Can be quickly calculated using dictionaries of word and tags
def emission_prob(word, tag, wordsAndTags, tags):
    numTimesLabeled = 0
    numTagsIn = 0
    if word in wordsAndTags and tag in wordsAndTags[word]:
        numTimesLabeled = wordsAndTags[word][tag]
    if tag in tags:
        numTagsIn = tags[tag]
    probability = (numTimesLabeled + 1) / (numTagsIn + len(tags))

    return np.log(probability)

# Score is emission + transition
def score(word, tag, prevTag, transitionProbTable, wordsAndTags, tags):
    return emission_prob(word, tag, wordsAndTags, tags) + transitionProbTable[tag][prevTag]

# implementation of the viterbi algorithm to find the best tagging sequence
def viterbi(sentence, wordsAndTags, tags, transitionProbTable):
    # Here we intilize viterbi variable and back pointer
    V = [{}]
    backptr = [{}]

    # split the sentence
    sentence = sentence.split()

    # find the intial viterbi variables for starting two tags
    for tag in tags:
        V[0][tag] = score(sentence[0], tag, "##", transitionProbTable, wordsAndTags, tags)

    # find the sentence tags for word 2 - M
    for m in range(1, len(sentence)):
        # Add empty to back pointers
        V.append({})
        backptr.append({})

        # Loop through every tag
        for tag in tags:
            # Find the best prevous tag if the current position is tagged as "tag"
            # This does the argMax and the max
            (tagScore, bestTag) = max(
                (V[m-1][prevTag] + score(sentence[m], tag, prevTag, transitionProbTable, wordsAndTags, tags), prevTag) for prevTag in tags)

            # Set the viterbi variable to the max score
            V[m][tag] = tagScore
            # Set the back pointer to the best previous tag
            backptr[m][tag] = bestTag
    
    # Here we need to find the best final tag which will begin our reverse into the back pointers
    (tagScore, bestTag) = max((V[len(sentence) - 1][final_state] + score('$$', '$$', final_state, transitionProbTable, wordsAndTags, tags), final_state) for final_state in tags)
    
    # Create a list to hold answer
    flist = []
    flist.append(bestTag)  # Add last tag to the answer list

    # Prevous tag is the best tag
    prevTag = bestTag

    # Loop through back pointers
    for m in range(len(sentence) - 1, 0, -1):
        flist = [backptr[m][prevTag]] + flist
        prevTag = backptr[m][prevTag]

    # Return tag list
    return flist
    





# Here we do our initilization
wordsAndTags, tags, corpus = processfile()
plotWordsAndTags(wordsAndTags, tags)
table = createTransitionProbTable(corpus, tags)

# Run viterbi on various test cases
# All test cases match the values in the word document

# Test 1
sentence = "nice !"
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 2
sentence = "good lord !"
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 3
sentence = "how are you ?"
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 4
sentence = "they can fish ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 5
sentence = "she is on diet ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 6
sentence = "we got kicked out ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 7
sentence = "there is no free food ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 8
sentence = "no need to worry about it at all !"
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 9
sentence = "how come the professor let him pass ?"
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 10
sentence = "maybe we shouldn't tell our parents that we didn't go ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 11
sentence = "how many people are there in the conference room ?"
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 12
sentence = "not sure if we can get there on time or not , so we should leave early ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 13
sentence = "speaking of that , we'll let you handle it ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 14
sentence = "remember : we are not here to kill time !"
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 15
sentence = "no offense but the result is not what we want . . ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 16
sentence = "2 examples are far from enough , you should try , at least , 3 more ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 17
sentence = "who would have thought we had to do remote learning for the past 2 years ? !"
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 18
sentence = "had the ball found the net , the goal would have been ruled out ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 19
sentence = "and the result of the analysis shows that people are currently not interested in buying their products ."
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)

# Test 20
sentence = "since the final project is hard , we should ask the professor for help !"
path = viterbi(sentence, wordsAndTags, tags, table)
print(sentence)
print(path)




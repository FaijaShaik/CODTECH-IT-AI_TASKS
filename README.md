# CODTECH-IT AI_TASKS

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SHAIK FAIJA

*INTERN ID*: CTIS05SU

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

# TASK1_Text_Summarization_Tool:

An extractive text summarization tool built in Python. This tool uses natural language processing (NLP) techniques to identify the most critical sentences in a given text and generates a concise summary based on frequency analysis.

# Requirements & Tools Used:

To run this tool, you need  

Python 3.x and the following libraries:

Python Libraries:nltk (Natural Language Toolkit) – For tokenization and text preprocessing.

heapq – A built-in Python module used to efficiently retrieve the highest-scoring sentences.

string – A built-in module used to handle and filter out punctuation.

NLTK Data Packages:punkt & punkt_tab – Models used for dividing text into sentences and words.

stopwords – A list of common filler words (e.g., "the", "is", "and") to ignore during analysis.

# Concept & Core Logic:

This script implements Extractive Summarization. Instead of generating completely new sentences, it ranks existing sentences from the original text and extracts the most important ones.The tool processes text through a four-step pipeline:
1. Text Preprocessing & Tokenization:The raw text is split into individual sentences using sent_tokenize() and individual words using word_tokenize(). All words are converted to lowercase to ensure consistency during evaluation.
2. Word Frequency Calculation:The tool filters out "stop words" and punctuation marks because they do not carry significant semantic meaning. It then counts how often each remaining word appears. To prevent longer texts from skewing the results, frequencies are normalized by dividing each word's count by the frequency of the most common word: {Normalized Frequency} = word count/maximum word count
4. Sentence Scoring:The script iterates through each sentence and calculates a total importance score. The score of a sentence is the sum of the normalized frequencies of the words it contains. Sentences containing highly frequent, meaningful keywords naturally receive higher scores.
5. Sentence Selection & Sorting:Based on the user-defined summary_ratio (e.g., 0.5 for 50% of the original length), the heapq.nlargest function extracts the top-scoring sentences. Finally, these selected sentences are re-sorted back into their original order of appearance to maintain narrative flow and readability before being stitched back into a final string.

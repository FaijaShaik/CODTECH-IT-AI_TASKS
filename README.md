# CODTECH-IT AI_TASKS

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SHAIK FAIJA

*INTERN ID*: CTIS05SU

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

# TASK1_Text_Summarization_Tool:

An extractive text summarization tool built in Python. This tool uses natural language processing (NLP) techniques to identify the most critical sentences in a given text and generates a concise summary based on frequency analysis.

## Requirements & Tools Used:

To run this tool, you need  

Python 3.x and the following libraries:

Python Libraries:nltk (Natural Language Toolkit) – For tokenization and text preprocessing.

heapq – A built-in Python module used to efficiently retrieve the highest-scoring sentences.

string – A built-in module used to handle and filter out punctuation.

NLTK Data Packages:punkt & punkt_tab – Models used for dividing text into sentences and words.

stopwords – A list of common filler words (e.g., "the", "is", "and") to ignore during analysis.

## Concept & Core Logic:

This script implements Extractive Summarization. Instead of generating completely new sentences, it ranks existing sentences from the original text and extracts the most important ones.The tool processes text through a four-step pipeline:
1. Text Preprocessing & Tokenization:The raw text is split into individual sentences using sent_tokenize() and individual words using word_tokenize(). All words are converted to lowercase to ensure consistency during evaluation.
2. Word Frequency Calculation:The tool filters out "stop words" and punctuation marks because they do not carry significant semantic meaning. It then counts how often each remaining word appears. To prevent longer texts from skewing the results, frequencies are normalized by dividing each word's count by the frequency of the most common word: {Normalized Frequency} = word count/maximum word count
4. Sentence Scoring:The script iterates through each sentence and calculates a total importance score. The score of a sentence is the sum of the normalized frequencies of the words it contains. Sentences containing highly frequent, meaningful keywords naturally receive higher scores.
5. Sentence Selection & Sorting:Based on the user-defined summary_ratio (e.g., 0.5 for 50% of the original length), the heapq.nlargest function extracts the top-scoring sentences. Finally, these selected sentences are re-sorted back into their original order of appearance to maintain narrative flow and readability before being stitched back into a final string.

#TASK2_SPEECH_TO_TEXT:

*Audio Transcription Tool*
An automated speech-to-text application built in Python. This tool processes local audio files, optimizes them by filtering out background noise, and utilizes cloud-based deep learning models to convert spoken audio into written text.

## Technologies Used:
Python: The underlying runtime environment and programming language.

SpeechRecognition (Library): A comprehensive Python wrapper that interfaces with multiple speech-to-text APIs.

Google Speech Recognition API: The cloud-based engine used to process, analyze, and decode the audio data into text.

OS Module (Python Standard Library): Utilized for system-level file verification to prevent application crashes.

## Requirements & Installation:
To run this tool locally, you need Python 3.x and the SpeechRecognition library installed.
pip install SpeechRecognition

## Concept & Core Logic:
This script operates as a Speech-to-Text Pipeline. It works by digitizing a static audio file, cleaning up its acoustic profile, and forwarding it to a remote acoustic machine learning model for linguistic translation.

The tool processes audio through a three-step pipeline:

1. Environment Verification & Initialization
The tool instantiates a Recognizer class, which serves as the control center for configuring and running speech recognition tasks. Before touching the file, the script explicitly verifies that the specified audio path exists, preventing standard runtime path errors.

2. Audio Conditioning & RecordingThe audio file is safely opened and read using sr.AudioFile. The pipeline then applies two critical adjustments:adjust_for_ambient_noise: Reads the first $0.5$ seconds of the file to gauge background static, dynamically recalibrating the energy threshold to filter out noise.record: Extracts the actual vocal wavelengths from the source file and saves them into a specific object model memory buffer (audio_data).

3. Cloud-Based Transcription & Error Handling
The finalized audio data is transmitted securely to Google's acoustic engine via recognize_google(), which returns the corresponding string output.

The entire process is wrapped inside an exception block to gracefully manage common real-world failures:

UnknownValueError: Triggers if the file is completely silent, warped, or missing discernable human speech structures.

RequestException: Triggers if your machine loses internet connectivity or if the remote cloud API server is temporarily unreachable.


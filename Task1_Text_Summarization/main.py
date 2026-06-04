#TEXT SUMMARIZATION TOOL
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
from heapq import nlargest


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def summarize_with_nltk(text, summary_ratio=0.3):
    # 1. Tokenize the text into sentences and words
    raw_sentences = sent_tokenize(text)
    
   
    # This replaces all hidden line breaks, tabs, and double spaces with a single space
    sentences = [" ".join(sent.split()) for sent in raw_sentences]
    words = word_tokenize(text.lower())
    
    # 2. Calculate Word Frequencies (excluding stop words and punctuation)
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}
    
    for word in words:
        if word not in stop_words and word not in punctuation:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
    # Normalize the frequencies
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
        
    # 3. Score Sentences based on the words they contain
    sentence_scores = {}
    for sent in sentences:
        # Tokenize individual words in this sentence to match frequencies
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
                    
    # 4. Select Top Sentences
    select_length = int(len(sentences) * summary_ratio)
    select_length = max(1, select_length)
    
    summary_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    
    summary_sentences = sorted(summary_sentences, key=sentences.index)
    # Return the final stitched summary
    return " ".join(summary_sentences)

if __name__ == "__main__":
    article = """
    Artificial Intelligence (AI) is transforming the world at an unprecedented pace. 
    From healthcare to finance, industries are leveraging machine learning algorithms to automate 
    complex tasks and make data-driven decisions. In medicine, AI models can detect anomalies in 
    X-rays with accuracy matching or exceeding human radiologists. Meanwhile, autonomous vehicles 
    are navigating complex city streets, promising a future with fewer traffic accidents. 
    However, this rapid technological shift raises significant ethical concerns. 
    Issues surrounding algorithmic bias, data privacy, and workforce displacement require urgent attention. 
    As AI systems become more integrated into daily life, developing robust regulatory frameworks 
    is crucial to ensure these technologies benefit humanity safely and equitably.
    """
    
    print(summarize_with_nltk(article, summary_ratio=0.5))
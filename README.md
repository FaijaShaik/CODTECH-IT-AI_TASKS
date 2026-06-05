# CODTECH-IT AI_TASKS

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SHAIK FAIJA

*INTERN ID*: CTIS05SU

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

## 📁 Project Structure

```text
CODTECH-IT/
│
├── Task1_Text_Summarization/
│   ├── main.py
│   └── output.txt
│
├── Task2_Speech_to_text/
│   ├── main.py
│   ├── output.txt
│   └── test.wav
│
├── Task3_Neural_style_transfer/
│   ├── art_style.jpg
│   ├── main.py
│   ├── my_photo.jpg
│   └── styled_output.jpg
│
└── Task4_Text_Generation_Model/
    └── main.py
```

# 📝 Task 1: Text Summarization Tool

### Concept

An extractive summarization tool that processes long blocks of text, analyzes keyword importance, and extracts the most high-value sentences to form a shortened summary without altering the original structural context.

### Requirements & Tools
Python 3.8+

nltk corporate datasets (punkt, punkt_tab, stopwords)

heapq module (Python standard library)

### Technologies Used
NLTK (Natural Language Toolkit): Utilized for text processing operations, specifically sentence and word tokenization.

### Key Features
Frequency Scoring Engine: Strips out English stop words and punctuation to count raw word frequencies, normalizing them against the most frequent token.

Sentence Ranking: Dynamically scores individual sentences by compiling their inner word weights, picking top results cleanly via heapq.nlargest

# 📝 Task 2: Speech to Text
### Concept
An Automated Speech Recognition (ASR) framework designed to take local audio track file inputs (.wav) containing spoken English and translate acoustic frequencies into readable text strings.

### Requirements & Tools
Python 3.8+

Target audio track asset (test.wav)

Active network connection (for cloud API validation)

### Technologies Used
SpeechRecognition Library: A flexible Python interface for interacting with popular speech-to-text decoding suites.

Google Speech Recognition API: Cloud-hosted, pre-trained deep learning networks that handle speech decoding.

### Key Features
Ambient Noise Reduction: Employs adjust_for_ambient_noise to sample background static before parsing, maximizing transcription accuracy.

Error Handling Blocks: Uses robust try-catch mechanisms to handle blurry audio or sudden service dropouts gracefully.

# 📝 Task 3: Neural Style Transfer
### Concept
An advanced computer vision application that blends a content photograph with an artistic style painting. It extracts the global layout shapes of your photo and overlays them with the colors and brushstroke patterns of the artwork.

### Requirements & Tools
Python 3.8+

Input assets (my_photo.jpg and art_style.jpg)

CUDA-capable GPU (Optional, falls back to CPU automatically)

### Technologies Used
PyTorch & Torchvision: Core deep learning framework used for image transformations and tensor mathematical calculations.

VGG-19 Network: A foundational convolutional neural network trained on millions of images, used to extract deep artistic traits.

### Key Features
Gram Matrix Formula: Isolates artistic textures and style details from the reference artwork while discarding layout boundaries.

In-Place Memory Patch: Disables VGG-19's default in-place ReLU operations (inplace=False) to secure memory pipelines and avoid execution runtime crashes.

# 📝 Task 4: Text Generation Model
### Concept
An autoregressive natural language generation system that reads open-ended conversational text prompts from the user and automatically predicts a high-quality continuation paragraph.

### Requirements & Tools
Python 3.8+

CLI interactive loop terminal

Hardware-agnostic layout (runs reliably on a standard laptop CPU)

### Technologies Used
Hugging Face Transformers: Used to fetch and execute pre-trained transformer pipelines easily.

GPT-2 (Generative Pre-trained Transformer 2): OpenAI’s language transformer model trained for sequential context prediction.

### Key Features
Advanced Decoding Parameters: Couples Top-K (50) and Top-p (0.92 / Nucleus) sampling with a calibrated temperature=0.8. This prevents repetitive phrases and keeps text outputs creative yet coherent.

🚀 Installation & Quick Start
To configure your environment and run these modules, execute the following commands in your terminal:

1. Install All Dependencies
  ```bash
3. pip install torch torchvision nltk transformers SpeechRecognition Pillow
```
4. Run an Application
Navigate into the respective task directory and execute the main Python entry point:
 ```bash
cd Task4_Text_Generation_Model
python main.py
```

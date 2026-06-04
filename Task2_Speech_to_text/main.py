import speech_recognition as sr
import os

def transcribe_audio(audio_file_path):
    # 1. Initialize the Recognizer instance
    recognizer = sr.Recognizer()
    
    # Check if the file actually exists before processing
    if not os.path.exists(audio_file_path):
        return f"Error: The file at '{audio_file_path}' was not found."
        
    try:
        # 2. Load and open the audio file
        with sr.AudioFile(audio_file_path) as source:
            print("Reading audio file and reducing background noise...")
            # Adjust for ambient noise to make the transcription cleaner
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Record the full audio content from the file
            audio_data = recognizer.record(source)
            
        print("Transcribing audio using pre-trained model...")
        # 3. Use the pre-trained Google model to recognize speech
        text_output = recognizer.recognize_google(audio_data)
        
        return text_output
        
    except sr.UnknownValueError:
        return "Speech Recognition could not understand the audio (Audio might be too blurry or silent)."
    except sr.RequestException as e:
        return f"Could not request results from the speech recognition service; {e}"

# --- Example Usage ---
if __name__ == "__main__":
    # Replace this path with the path to your own short .wav audio clip
    sample_audio = "test.wav" 
    print("--- SPEECH RECOGNITION SYSTEM ---")
    result = transcribe_audio(sample_audio)
    print("\n--- TRANSCRIPTION RESULT ---")
    print(result)
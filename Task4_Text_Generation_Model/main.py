import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text():
    print("Loading pre-trained GPT-2 model (this may take a minute on the first run)...")
    
    # 1. Load the pre-trained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Set the model to evaluation mode
    model.eval()
    
    # Use CPU since we want it to run reliably on any laptop layout
    device = torch.device("cpu")
    model.to(device)
    
    print("\n--- GPT-2 TEXT GENERATION SYSTEM READY ---")
    print("Type 'exit' to stop the program.\n")
    
    while True:
        # 2. Get prompt input from user
        prompt = input("Enter a prompt/topic: ")
        if prompt.lower() == 'exit':
            print("Stopping system. Goodbye!")
            break
            
        if not prompt.strip():
            print("Prompt cannot be empty! Try again.")
            continue
            
        print("\nGenerating coherent paragraph...")
        
        # 3. Convert input text into mathematical tokens (numbers)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # 4. Generate the continuation using Top-K and Top-p sampling for high quality
        with torch.no_grad():
            output_beam = model.generate(
                input_ids,
                max_length=100,          
                num_return_sequences=1,   
                no_repeat_ngram_size=2,  
                do_sample=True,           
                top_k=50,                 
                top_p=0.92,                
                temperature=0.8,          
                pad_token_id=tokenizer.eos_token_id
            )
            
        # 5. Decode the mathematical numbers back into readable English words
        generated_text = tokenizer.decode(output_beam[0], skip_special_tokens=True)
        
        print("\n--- GENERATED PARAGRAPH ---")
        print(generated_text)
        print("-" * 30 + "\n")

if __name__ == "__main__":
    generate_text()
    
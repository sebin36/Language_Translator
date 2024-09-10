import streamlit as st
from transformers import MarianMTModel, MarianTokenizer


# Define available language codes (MarianMT uses language pairs)
language_codes = {
    'English': 'en',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Italian': 'it',
    'Dutch': 'nl',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Hindi': 'hi'
}

def load_model(src_lang, tgt_lang):
    # Load the appropriate MarianMT model and tokenizer based on source and target languages
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate(text, model, tokenizer):
    # Tokenize and translate the input text
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
    return translated_text

def main():
    st.title("Language Translator")
    
    # Language selection
    src_lang = st.selectbox("Select source language:", list(language_codes.keys()))
    tgt_lang = st.selectbox("Select target language:", list(language_codes.keys()))
    
    # Ensure source and target languages are different
    if src_lang == tgt_lang:
        st.error("Source and target languages must be different.")
        return
    
    # Get input text
    text = st.text_area("Enter text to translate:")
    
    if st.button("Translate"):
        if text.strip() == "":
            st.warning("Please enter some text to translate.")
        else:
            src_code = language_codes[src_lang]
            tgt_code = language_codes[tgt_lang]
            
            try:
                # Load model and tokenizer for the selected language pair
                model, tokenizer = load_model(src_code, tgt_code)
                translated_text = translate(text, model, tokenizer)
                st.success(f"Translated Text ({tgt_lang}):")
                st.write(translated_text)
            except Exception as e:
                st.error(f"Error in translation: {str(e)}")

if __name__ == "__main__":
    main()
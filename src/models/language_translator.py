import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def translate_en_to_fr(text, model, tokenizer):
    """
    Translates English text to French using facebook/mbart-large-50-many-to-many-mmt
    """
    # 1. Set the source language code (English)
    tokenizer.src_lang = "en_XX"

    # 2. Tokenize the input text
    encoded_input = tokenizer(text, return_tensors="pt")

    # 3. Generate translation
    # We set forced_bos_token_id to the French language code in the model's vocab
    generated_tokens = model.generate(
        **encoded_input, 
        forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"],
        max_length=100
    )

    # 4. Decode the tokens
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text


if __name__ == "__main__":
    # 1) Load the model & tokenizer for MBart50
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # 2) Example English text
    english_text = "set green in o six soon"

    # 3) Translate from English to French
    french_translation = translate_en_to_fr(english_text, model, tokenizer)

    print("English Text:", english_text)
    print("French Translation:", french_translation)

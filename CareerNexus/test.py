import spacy
print(spacy.__version__)  # Check the installed version
nlp = spacy.load("en_core_web_sm")  # Load a small English model
doc = nlp("Hello, SpaCy is working!")  
print([(token.text, token.pos_) for token in doc])  # Print tokenized output

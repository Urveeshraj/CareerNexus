import fitz  # PyMuPDF
import re
import nltk
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

job_descriptions = [
    {"category": "Data Science", "description": "Looking for a Data Scientist skilled in Python, Machine Learning, and NLP."},
    {"category": "HR", "description": "HR specialist with experience in recruitment, employee relations, and payroll."},
    {"category": "Operations Manager", "description": "Expert in process optimization, supply chain management, and leadership."},
    {"category": "Sales", "description": "Sales Executive skilled in business development, lead generation, and client relations."},
    {"category": "Automation Testing", "description": "Automation Tester with expertise in Selenium and automated frameworks."},
    {"category": "DevOps Engineer", "description": "DevOps Engineer with knowledge of CI/CD, Kubernetes, and cloud infrastructure."},
    {"category": "ETL Developer", "description": "ETL Developer proficient in data pipelines, ETL tools, and databases."}
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\b(yes|basic|beginner|proficiency|experience)\b', '', text)
    text = re.sub(r'[\r\n]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT"]]
    return " ".join(skills)

def process_resume(file_path):
    resume_text = extract_text_from_pdf(file_path)
    cleaned_resume = clean_text(resume_text)
    extracted_skills = extract_skills(cleaned_resume)

    job_texts = [job['description'] for job in job_descriptions]
    job_embeddings = model.encode(job_texts, convert_to_tensor=True)
    resume_embedding = model.encode([cleaned_resume], convert_to_tensor=True)

    similarity_scores = util.pytorch_cos_sim(resume_embedding, job_embeddings).numpy().flatten()
    best_match_index = np.argmax(similarity_scores)
    best_match_category = job_descriptions[best_match_index]['category']
    best_match_score = similarity_scores[best_match_index]

    return extracted_skills, similarity_scores, best_match_category, best_match_score

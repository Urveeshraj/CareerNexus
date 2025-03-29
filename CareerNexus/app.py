from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF for PDF handling
import re
import nltk
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

# Create uploads folder if not exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Load NLP models
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Job descriptions
job_descriptions = [
    {"category": "Data Science", "description": "Looking for a Data Scientist skilled in Python, Machine Learning, and NLP."},
    {"category": "HR", "description": "HR specialist with experience in recruitment, employee relations, and payroll."},
    {"category": "Operations Manager", "description": "Expert in process optimization, supply chain management, and leadership."},
    {"category": "Sales", "description": "Sales Executive skilled in business development, lead generation, and client relations."},
    {"category": "Automation Testing", "description": "Automation Tester with expertise in Selenium and automated frameworks."},
    {"category": "DevOps Engineer", "description": "DevOps Engineer with knowledge of CI/CD, Kubernetes, and cloud infrastructure."},
    {"category": "ETL Developer", "description": "ETL Developer proficient in data pipelines, ETL tools, and databases."}
]

from model import process_resume  # Import NLP function

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    extracted_skills, similarity_scores, best_match_category, best_match_score = process_resume(file_path)

    return jsonify({
        "skills": extracted_skills,
        "scores": {job_descriptions[i]['category']: float(similarity_scores[i]) for i in range(len(job_descriptions))},
        "best_match": best_match_category,
        "best_match_score": float(best_match_score)
    })

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

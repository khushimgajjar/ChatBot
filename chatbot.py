from flask import Flask, request, render_template, jsonify
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
        preprocessed_sentences.append(' '.join(words))
    return preprocessed_sentences, sentences

# Function to create TF-IDF matrix
def create_tfidf_matrix(sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix, vectorizer

# Function to answer a question
def answer_question(question, tfidf_matrix, original_sentences, vectorizer, top_n=5):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    question = ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(question) if word.isalpha() and word.lower() not in stop_words])
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()
    
    if np.max(similarities) == 0:
        return "Sorry, I couldn't find an answer to that question."
    
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_sentences = [original_sentences[i] for i in top_indices]
    
    return " ".join(top_sentences)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(file.filename)
    text = extract_text_from_pdf(file.filename)
    
    preprocessed_sentences, original_sentences = preprocess_text(text)
    tfidf_matrix, vectorizer = create_tfidf_matrix(preprocessed_sentences)
    
    # Save these to use in the question answering route
    app.config['tfidf_matrix'] = tfidf_matrix
    app.config['original_sentences'] = original_sentences
    app.config['vectorizer'] = vectorizer
    
    return "File uploaded and processed successfully."

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    tfidf_matrix = app.config['tfidf_matrix']
    original_sentences = app.config['original_sentences']
    vectorizer = app.config['vectorizer']
    
    response = answer_question(question, tfidf_matrix, original_sentences, vectorizer)
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)

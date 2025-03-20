import streamlit as st
import fitz  # PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re

nlp = spacy.load("en_core_web_sm")
JOB_DESCRIPTION = """
Looking for a skilled Python developer with experience in NLP, machine learning, and data analysis.
Must be proficient in Python, Pandas, Scikit-learn, and Streamlit. Should have experience working with text data.
Should know the language Java.Must be skilled in frontend like html, css, javascript.Must have basic knowledge of Database Management System(DBMS).
"""

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = " ".join([page.get_text("text") for page in doc])
    return text

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    doc = nlp(text)  # Process text with SpaCy
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def main():
    st.title("AI-powered Resume Screening and Ranking System")
    st.write("Upload a single resume (PDF) to check its relevance for a job description.")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    if uploaded_file is not None:
        # Extract text
        resume_text = extract_text_from_pdf(uploaded_file)
        processed_resume = preprocess_text(resume_text)
        processed_job_desc = preprocess_text(JOB_DESCRIPTION)
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_job_desc])
        
        # Cosine Similarity
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] * 100
        
        # Display results
        st.subheader("Resume Screening Result:")
        st.write(f"Relevance Score: **{similarity_score:.2f}%**")
        if similarity_score > 50:
            st.success("✅ Resume is a good match for the job description!")
            def main():
                st.title("AI-powered Resume Screening and Ranking System")
                st.write("Upload a single resume (PDF) to check its relevance for a job description.")
    
        else:
            st.warning("⚠️ Resume may not be a strong match. Consider improving it!")
if __name__ == "__main__":
    main()

    
    
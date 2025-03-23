# ML-Resume-Screening-Model
Project Overview:
This project is an AI-powered Resume Screening and Ranking System developed using Python, Streamlit, and Machine Learning techniques. It aims to simplify and automate the recruitment process by analyzing resumes and ranking them based on their relevance to a given job description.

Features
1)Extracts text data from PDF resumes using the PyMuPDF library.        
2)Uses spaCy for advanced Natural Language Processing (NLP) tasks like tokenization and 
  lemmatization.       
3)Implements TF-IDF Vectorization to convert textual data into numerical form.         
4)Calculates Cosine Similarity to measure the relevance of resumes against the job description.
5)Displays ranked resumes using an interactive Streamlit interface.

Technologies Used
1)Python: Core programming language
2)Streamlit: For building an interactive web interface
3)PyMuPDF (fitz): For extracting text from PDF files
4)spaCy: For NLP tasks
5scikit-learn: For TF-IDF Vectorization and Cosine Similarity

Installation:
1.Clone this repository:
  git clone https://github.com/your-username/ai-resume-screening.git
  cd ai-resume-screening
2.Install the required dependencies:
  pip install -r requirements.txt
3.Download the spaCy English model:
  python -m spacy download en_core_web_sm

How to Run the Application:
 streamlit run finalapp.py
Upload resumes in PDF format and provide the job description. The system will rank the resumes based on relevance.

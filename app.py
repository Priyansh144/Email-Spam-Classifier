import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Download NLTK data
nltk.download('punkt')

# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Add custom CSS for styling
st.markdown(
    """
    <style>
    /* Brownish gradient background */
    .stApp {
        background: linear-gradient(to bottom right, #8B4513, #D2B48C, #F4A460);
        background-size: 300% 300%;
        animation: gradientBG 10s ease infinite;
        font-family: 'Arial', sans-serif;
        color: white;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Header styling */
    .header {
        text-align: center;
        color: #ffffff;
        padding: 10px;
        border-radius: 15px;
        border: 3px solid #ffffff;
        background-color: rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    /* Custom label styling */
    .input-label {
        font-size: 18px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
        display: block;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }

    /* Text area styling with black background */
    .stTextArea textarea {
        background-color: #000000 !important;
        color: white !important;
        border: 2px solid #ffffff !important;
        border-radius: 15px !important;
        padding: 10px !important;
        font-size: 16px !important;
        font-family: 'Arial', sans-serif !important;
    }

    /* Button styling */
    .stButton button {
        background-color: #A0522D !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
        border: 2px solid #ffffff !important;
    }
    .stButton button:hover {
        background-color: #8B4513 !important;
    }

    /* Result box styling */
    .result {
        text-align: center;
        font-size: 24px;
        margin-top: 20px;
        padding: 10px;
        border-radius: 15px;
        border: 3px solid white;
        background-color: rgba(255, 255, 255, 0.2);
    }
    .spam {
        color: #e74c3c;
    }
    .not-spam {
        color: #2ecc71;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App header
st.markdown('<div class="header"><h1>Email/SMS Spam Classifier</h1></div>', unsafe_allow_html=True)

# Custom input label
st.markdown('<label class="input-label">Enter the message</label>', unsafe_allow_html=True)

# Input area for the message
input_sms = st.text_area("", help="Type your message here", key="text", placeholder="Your message...")

# Button to trigger the prediction
if st.button('Predict'):
    # Step 1: Preprocess the input text
    transformed_sms = transform_text(input_sms)
    
    # Step 2: Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])
    
    # Step 3: Predict the result
    result = model.predict(vector_input)[0]
    
    # Step 4: Display the result with styling
    if result == 1:
        st.markdown('<div class="result spam">Spam</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result not-spam">Not Spam</div>', unsafe_allow_html=True)

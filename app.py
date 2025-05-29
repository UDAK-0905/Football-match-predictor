import streamlit as st
import pandas as pd
from src.predictor import load_data, preprocess, train_model, predict_match

@st.cache_data
def load_and_train():
    df = load_data('data/matches.csv')  # adjust path if needed
    X, y, le_home, le_away = preprocess(df)
    model = train_model(X, y)  # Train on all data (or split/train if you want)
    return model, le_home, le_away

def main():
    st.title("Football Match Result Predictor ⚽️")

    model, le_home, le_away = load_and_train()

    st.write("### Select Teams to Predict Match Result")

    home_team = st.selectbox("Home Team", options=le_home.classes_)
    away_team = st.selectbox("Away Team", options=le_away.classes_)

    if st.button("Predict Result"):
        if home_team == away_team:
            st.error("Home and Away teams cannot be the same.")
        else:
            prediction = predict_match(model, home_team, away_team, le_home, le_away)
            result_map = {'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'}
            st.success(f"Predicted Result: {result_map.get(prediction, 'Unknown')}")

if __name__ == "__main__":
    main()

import streamlit as st

# Custom CSS styling
st.markdown("""
    <style>
    /* Background and overall layout */
    .main {
        background-color: #e0f7fa;
        padding: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Page title */
    .stApp h1 {
        color: #1b5e20;
        text-align: center;
        font-weight: bold;
        font-size: 3rem;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #a5d6a7 !important;
    }

    /* Input widgets */
    .stSelectbox, .stButton>button {
        background-color: #66bb6a;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #388e3c;
        transition: 0.3s;
    }

    /* Prediction output */
    .prediction-box {
        background-color: #ffffff;
        padding: 1rem 2rem;
        border: 2px solid #388e3c;
        border-radius: 12px;
        margin-top: 2rem;
        font-size: 1.2rem;
        text-align: center;
        color: #2e7d32;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    body, .stApp {
        background-color: #e0f7fa !important;
    }
    </style>
""", unsafe_allow_html=True)


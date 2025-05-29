## ⚽ Football Match Predictor
Predict the outcome of football matches using machine learning! This app leverages historical football match data and team statistics to forecast whether a match will end in a Home Win, Away Win, or Draw.

### 🚀 Features
- Uses team-specific rolling averages of goals scored and conceded for more accurate predictions.

- Encodes team names and dynamically updates statistics based on historical match results.

- Intuitive web app UI built with Streamlit for easy match result predictions.

- Input home and away teams to get an instant prediction on match outcome.

- Lightweight and fast — no heavy models, just smart feature engineering!

### 🛠️ How It Works
The model calculates each team’s average goals scored and conceded before every match.

These stats are used as features for training a **Logistic Regression Model**.

When you enter teams in the app, it predicts the match outcome based on learned patterns from past data.

### 📈 Usage
#### Clone the repository:

git clone https://github.com/yourusername/Football-match-predictor.git
cd Football-match-predictor

#### Install dependencies:

pip install -r requirements.txt
#### Run the app locally:

streamlit run app.py
#### Or access the live app [here](https://football-match-predictor-hs2d4kprdjaqhpzatygxhk.streamlit.app/).

### 📊 Dataset
#### The model uses historical match data with features like:

- Full time and half time goals (FTHG, FTAG, HTHG, HTAG)

- Shots on target, corners, fouls, yellow and red cards

- Team names encoded as categorical features

**⚠️ Note**: The raw dataset is not included here due to size and licensing, but you can use your own historical football match dataset.

**🤝 Contributions**: 
Contributions and suggestions are welcome! Feel free to open issues or submit pull requests.

**📜 License**: 
MIT License © 2025 Uday Wawage

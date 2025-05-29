import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def load_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    return df

def preprocess(df):
    # Convert DateTime to datetime type and sort by date
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)

    # Create columns to hold rolling avg stats before each match
    df['HomeTeamAvgGoalsScored'] = 0.0
    df['HomeTeamAvgGoalsConceded'] = 0.0
    df['AwayTeamAvgGoalsScored'] = 0.0
    df['AwayTeamAvgGoalsConceded'] = 0.0

    # Initialize dictionaries to hold running sums and counts
    team_goals_scored = {}
    team_goals_conceded = {}
    team_matches_played = {}

    # Function to update rolling stats
    def update_stats(team, goals_scored, goals_conceded):
        if team not in team_goals_scored:
            team_goals_scored[team] = 0
            team_goals_conceded[team] = 0
            team_matches_played[team] = 0
        team_goals_scored[team] += goals_scored
        team_goals_conceded[team] += goals_conceded
        team_matches_played[team] += 1

    # Loop through matches in chronological order
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']

        # Calculate average goals scored/conceded before this match for home team
        if team_matches_played.get(home, 0) > 0:
            df.at[idx, 'HomeTeamAvgGoalsScored'] = team_goals_scored[home] / team_matches_played[home]
            df.at[idx, 'HomeTeamAvgGoalsConceded'] = team_goals_conceded[home] / team_matches_played[home]
        else:
            # No previous data, set zero or some default
            df.at[idx, 'HomeTeamAvgGoalsScored'] = 0
            df.at[idx, 'HomeTeamAvgGoalsConceded'] = 0

        # Same for away team
        if team_matches_played.get(away, 0) > 0:
            df.at[idx, 'AwayTeamAvgGoalsScored'] = team_goals_scored[away] / team_matches_played[away]
            df.at[idx, 'AwayTeamAvgGoalsConceded'] = team_goals_conceded[away] / team_matches_played[away]
        else:
            df.at[idx, 'AwayTeamAvgGoalsScored'] = 0
            df.at[idx, 'AwayTeamAvgGoalsConceded'] = 0

        # Now update stats with the current match result AFTER recording averages
        update_stats(home, row['FTHG'], row['FTAG'])
        update_stats(away, row['FTAG'], row['FTHG'])

    # Encode team names
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df['HomeTeamEnc'] = le_home.fit_transform(df['HomeTeam'])
    df['AwayTeamEnc'] = le_away.fit_transform(df['AwayTeam'])

    # Define features to use for prediction
    features = df[['HomeTeamEnc', 'AwayTeamEnc', 'HomeTeamAvgGoalsScored', 'HomeTeamAvgGoalsConceded',
                   'AwayTeamAvgGoalsScored', 'AwayTeamAvgGoalsConceded']]

    # Target is Full Time Result (FTR): 'H', 'A', or 'D'
    target = df['FTR']

    return features, target, le_home, le_away

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def predict_match(model, home_team, away_team, le_home, le_away):
    # Encode teams
    try:
        home_enc = le_home.transform([home_team])[0]
        away_enc = le_away.transform([away_team])[0]
    except ValueError:
        return "Unknown team name(s). Please check spelling."

    # Since rolling averages need historical data, here we use zero or average values as placeholders
    # You can improve by passing rolling stats from your app if available
    input_df = pd.DataFrame([[home_enc, away_enc, 0, 0, 0, 0]],
                            columns=['HomeTeamEnc', 'AwayTeamEnc', 'HomeTeamAvgGoalsScored',
                                     'HomeTeamAvgGoalsConceded', 'AwayTeamAvgGoalsScored', 'AwayTeamAvgGoalsConceded'])
    prediction = model.predict(input_df)[0]
    return prediction

if __name__ == "__main__":
    df = load_data('../data/matches.csv')
    X, y, le_home, le_away = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate(model, X_test)

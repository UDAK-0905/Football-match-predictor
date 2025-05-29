import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def load_data(filepath):
    df = pd.read_csv('../data/matches.csv')
    return df

def preprocess(df):

    def get_result(row):
        if row['FTHG'] > row['FTAG']:
            return 'HomeWin'
        elif row['FTHG'] < row['FTAG']:
            return 'AwayWin'
        else:
            return 'Draw'
        
    df['Result'] = df.apply(get_result, axis=1)

    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df['HomeTeamEnc'] = le_home.fit_transform(df['HomeTeam'])
    df['AwayTeamEnc'] = le_away.fit_transform(df['AwayTeam'])

    features = df[['HomeTeamEnc', 'AwayTeamEnc', 'FTHG', 'FTAG']]
    target = df['Result']

    return features, target, le_home, le_away

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    df = load_data('../data/matches.csv')
    X, y, le_home, le_away = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate(model, X_test)
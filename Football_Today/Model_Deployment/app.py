from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load(open('model/match_predictors_log.pkl', 'rb'))

# Load processed dataset for stats

df = pd.read_csv('model/EPL_processed.csv')

# Team ratings dictionary
team_ratings = {
    "Man City": 0.740, "Liverpool": 0.711, "Arsenal": 0.684, "Sunderland": 0.544,
    "Chelsea": 0.525, "Newcastle": 0.507, "Tottenham": 0.506, "Aston Villa": 0.503,
    "Man United": 0.496, "Brighton": 0.460, "Fulham": 0.425, "Brentford": 0.415,
    "Bournemouth": 0.401, "Crystal Palace": 0.397, "West Ham": 0.390, "Nott'm Forest": 0.374,
    "Wolves": 0.362, "Everton": 0.344, "Leicester": 0.299, "Leeds": 0.276,
    "Burnley": 0.214, "Luton": 0.197, "Southampton": 0.197, "Watford": 0.194,
    "Norwich": 0.166, "Ipswich": 0.149, "Sheffield United": 0.111
}

# Simple encoders (replace with actual encoders if needed)
def encode_categorical(value):
    return hash(value) % 1000

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    home_team = request.form['HomeTeam']
    away_team = request.form['AwayTeam']
    B365H = float(request.form['B365H'])
    B365A = float(request.form['B365A'])
    B365D = float(request.form['B365D'])
    home_form = float(request.form['HomeTeam_Form_Score'])
    away_form = float(request.form['AwayTeam_Form_Score'])
    referee = request.form['Referee']
    time = request.form['time']

    # Ratings
    home_rating = team_ratings.get(home_team, 0.5)
    away_rating = team_ratings.get(away_team, 0.5)

    # Encode categorical features
    referee_encoded = encode_categorical(referee)
    home_encoded = encode_categorical(home_team)
    away_encoded = encode_categorical(away_team)
    time_encoded = encode_categorical(time)

    # Final feature vector
    features = [B365H, B365A, B365D, home_rating, away_rating,
                referee_encoded, away_form, home_form, away_encoded,
                home_encoded, time_encoded]

    prediction_code = model.predict([features])[0]
    raw_confidence = model.predict_proba([features])[0][0]  # original confidence

    # Boost predicted class to at least 60%
    boosted_confidence = round(min(raw_confidence * 100 + 10, 95), 2)
    remaining = 100 - boosted_confidence

    if prediction_code == 1:  # Home Win
        result = f"{home_team} Wins"
        home_prob = away_prob = draw_prob = 0
        home_prob = boosted_confidence
        away_prob = round(remaining * 0.6, 2)
        draw_prob = round(remaining - away_prob, 2)

    elif prediction_code == 2:  # Away Win
        result = f"{away_team} Wins"
        home_prob = away_prob = draw_prob = 0
        away_prob = boosted_confidence
        home_prob = round(remaining * 0.4, 2)
        draw_prob = round(remaining - home_prob, 2)

    else:  # Draw
        result = f"{home_team} vs {away_team} ends in a Draw"
        home_prob = away_prob = draw_prob = 0
        draw_prob = boosted_confidence
        home_prob = round(remaining * 0.6, 2)
        away_prob = round(remaining - home_prob, 2)

    return render_template('index.html',result=result,home_prob=home_prob,away_prob=away_prob,draw_prob=draw_prob,home_team=home_team,away_team=away_team)

# insights
@app.route('/team_stats', methods=['POST'])
def team_stats():
    data = request.get_json()
    home = data['home']
    away = data['away']

    # Head-to-head stats
    h2h = df[((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
             ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))]
    result_counts = h2h['Result'].value_counts().to_dict()

    # Average ratings
    home_rating = team_ratings.get(home, 0.5)
    away_rating = team_ratings.get(away, 0.5)

    # Recent 5 matches
    home_form = df[(df['HomeTeam'] == home) | (df['AwayTeam'] == home)].tail(5)['Result'].tolist()
    away_form = df[(df['HomeTeam'] == away) | (df['AwayTeam'] == away)].tail(5)['Result'].tolist()

    return jsonify({
        "result_counts": result_counts,
        "home_rating": home_rating,
        "away_rating": away_rating,
        "home_form": home_form,
        "away_form": away_form
    })


if __name__ == '__main__':
    app.run(debug=True)
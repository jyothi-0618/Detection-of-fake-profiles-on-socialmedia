from flask import Flask, render_template, request
import numpy as np
import joblib
import instaloader

app = Flask(__name__)

# Load the trained model and scaler
rfc = joblib.load("rfc_model.pkl")
sc = joblib.load("scaler.pkl")

# Prediction function
def prediction(username):
    try:
        # Create an Instaloader instance
        loader = instaloader.Instaloader()

        # Fetch profile information
        profile = instaloader.Profile.from_username(loader.context, username)

        # Extract features from the profile
        num_digits_in_username = sum(char.isdigit() for char in profile.username)
        num_words_in_username = len(profile.username.split())
        num_digits_in_full_name = sum(char.isdigit() for char in profile.full_name)
        num_words_in_full_name = len(profile.full_name.split())

        # Define profile features
        profile_data = {
            'profile_pic': int(bool(profile.profile_pic_url)),
            'num_by_num': num_digits_in_username / max(num_words_in_username, 1),
            'full_name': num_words_in_full_name,
            'num_by_char': num_digits_in_full_name / max(len(profile.full_name), 1),
            'name_username': int(profile.full_name.lower() == profile.username.lower()),
            'bio_len': len(profile.biography),
            'url': int(bool(profile.external_url)),
            'private': int(profile.is_private),
            'post': profile.mediacount,
            'followers': profile.followers,
            'follows': profile.followees,
        }

        # Convert the profile features to a NumPy array
        profile_features = np.array([
            profile_data['profile_pic'], profile_data['num_by_num'], profile_data['full_name'],
            profile_data['num_by_char'], profile_data['name_username'], profile_data['bio_len'],
            profile_data['url'], profile_data['private'], profile_data['post'],
            profile_data['followers'], profile_data['follows']
        ]).reshape(1, -1)

        # Standardize the profile features
        profile_features_scaled = sc.transform(profile_features)

        # Make predictions
        prediction = rfc.predict(profile_features_scaled)
        prob = rfc.predict_proba(profile_features_scaled)

        # Get prediction result and probability
        if prediction == 0:
            result = 'Genuine account'
        else:
            result = 'Spam account'

        probability = prob[:, prediction] * 100
        percentage_value = probability.item()

        return result, percentage_value, profile_data

    except instaloader.exceptions.ProfileNotExistsException:
        return 'Profile username does not exist.', None, None
    except Exception as e:
        return 'Profile not found', None, None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form['username']
        result, probability, details = prediction(username)
        return render_template("result.html", result=result, probability=probability, details=details)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

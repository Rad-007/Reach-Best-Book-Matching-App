import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = {
    "Conscientiousness": [1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 9, 8],
    "Openness": [2, 6, 5, 5, 3, 4, 1, 2, 4, 5, 1, 7, 8, 3, 5, 6, 7, 9, 4, 6, 2],
    "Genre": ["Manga", "Comics", "Satire", "Young Adult", "Youth", "Graphic Novel", 
              "Girl's Fiction", "Romance", "Drama", "Kids", "Asian", "Horror", 
              "Sci-Fi", "Humour", "Journeys", "Classics", "Plays", "Philosophy", 
              "Mystery", "Self Improvement", "Religion"]
}

df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df[['Conscientiousness', 'Openness']]
y = df['Genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost model
catboost_model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, loss_function='MultiClass')
catboost_model.fit(X_train, y_train, verbose=100)

# Evaluate the model
y_pred_catboost = catboost_model.predict(X_test)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)

print(f"Accuracy (CatBoost): {accuracy_catboost}")

# Create a function for genre prediction using CatBoost
def predict_genre_catboost(conscientiousness, openness, model):
    new_data = pd.DataFrame({'Conscientiousness': [conscientiousness], 'Openness': [openness]})
    prediction = model.predict(new_data)
    genre = prediction[0]
    return genre

# Example usage of the prediction function with CatBoost
new_conscientiousness = 4
new_openness = 6
predicted_genre_catboost = predict_genre_catboost(new_conscientiousness, new_openness, catboost_model)

print(f"Predicted Genre (CatBoost): {predicted_genre_catboost}")

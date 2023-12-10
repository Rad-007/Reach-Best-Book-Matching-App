import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import r2_score

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

# Encode categorical target variable
genre_mapping = {genre: i for i, genre in enumerate(df['Genre'].unique())}
df['Genre'] = df['Genre'].map(genre_mapping)

# Split the data into features (X) and target variable (y)
X = df[['Conscientiousness', 'Openness']]
y = df['Genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Train SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Evaluate the models
y_pred_logistic = logistic_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)

r2_logistic = r2_score(y_test, y_pred_logistic)
r2_svm = r2_score(y_test, y_pred_svm)

print(f"R^2 Score (Logistic Regression): {r2_logistic}")
print(f"R^2 Score (SVM): {r2_svm}")

# Create a function for genre prediction
def predict_book_genre(conscientiousness, openness):
    new_data = pd.DataFrame({'Conscientiousness': [conscientiousness], 'Openness': [openness]})
    prediction = svm_model.predict(new_data)
    genre = next(key for key, value in genre_mapping.items() if value == prediction[0])
    return genre

'''
# Example usage of the prediction function
new_conscientiousness = 4
new_openness = 6
predicted_genre_logistic = predict_genre(new_conscientiousness, new_openness, logistic_model)
predicted_genre_svm = predict_genre(new_conscientiousness, new_openness, svm_model)

print(f"Predicted Genre (Logistic Regression): {predicted_genre_logistic}")
print(f"Predicted Genre (SVM): {predicted_genre_svm}")
'''

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

file_path = 'data/playlist.csv'
data = pd.read_csv(file_path) 

data = data.dropna(axis=0)

le_y = LabelEncoder()
data['name_of_artists'] = le_y.fit_transform(data['name_of_artists'])

y = data['name_of_artists']  

# Convert 'last_updated' to datetime
data['track_add_date'] = pd.to_datetime(data['track_add_date'])

# Extract components from 'last_updated'
data['year'] = data['track_add_date'].dt.year
data['month'] = data['track_add_date'].dt.month
data['day'] = data['track_add_date'].dt.day
data['hour'] = data['track_add_date'].dt.hour

features = ['year', 'month', 'day', 'hour']
X = data[features]
print(X.head())

# le_X = LabelEncoder()
# X.track_name = le_X.fit_transform(X.track_name)

print(X.head())
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

forest_model = RandomForestClassifier()
forest_model.fit(train_X, train_y)
y_pred = forest_model.predict(train_X)

# Use inverse_transform to get original 'condition_text'
y_pred = le_y.inverse_transform(y_pred)
print (y_pred)

joblib.dump(le_y, 'model/random_forest/le_y.joblib')
joblib.dump(forest_model, 'model/random_forest/spotify.joblib')
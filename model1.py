import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
#loading data
song_data = pd.read_csv("E:\data science projects\Project_2_Song_Popularity\data.csv")
#pre-processing
song_data = song_data.drop(['id','name'], axis=1)
song_data.loc[song_data['popularity'] < 65, 'popularity'] = 0
song_data.loc[song_data['popularity'] >= 65, 'popularity'] = 1
features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness", "loudness",
            "mode", "speechiness", "tempo", "valence"]
X = song_data[features]
y = song_data.popularity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#model training(random forest classiier)
randf = RandomForestClassifier()
#testing Random Forest Classifier Model with training data
randf.fit(X_train, y_train)
with open('model1.pkl','wb') as f:
    pickle.dump(randf,f)

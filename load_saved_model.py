from sklearn.metrics import r2_score
import keras
import numpy as np
import pandas as pd

model = keras.saving.load_model('models/model_2500.keras')


dataset_df = pd.read_csv('preprocessed_data.csv')


dataset = dataset_df.to_numpy()

y_train = dataset[:, 11]
x1 = dataset[:, 0:11]
x2 = dataset[:, 12:]
X_train = np.concatenate((x1, x2), axis=1)


model.summary()
model.evaluate(X_train, y_train)
predictions = model.predict(X_train)
print(r2_score(y_train, predictions))
import pandas as pd
from sklearn.utils import shuffle
df = pd.read_csv('CreditPrediction.csv')

df = df.drop(['Unnamed: 19'], axis=1)
df = df.drop(['CLIENTNUM'], axis=1)

credit = df.pop('Credit_Limit')
df['Credit_Limit'] = credit


df['Gender'].replace(['M', 'F'], [0, 1], inplace=True)
df['Gender'].fillna(value=0, inplace=True)

df['Education_Level'].replace(['Uneducated', 'High School', 'College','Graduate', 'Post-Graduate', 'Doctorate', 'Unknown'],
                              [0, 1, 2, 3, 4, 5, 3], inplace=True)

df['Marital_Status'].replace(['Single', 'Married', 'Divorced', 'Unknown'], [0, 1, 2, 1], inplace=True)
df['Marital_Status'].fillna(value=0, inplace=True)

df['Income_Category'].replace(['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown'],
                              [0, 1, 2, 3, 4, 5], inplace=True)

df['Card_Category'].replace(['Blue', 'Silver', 'Gold', 'Platinum'],
                            [0, 1, 2, 3], inplace=True)

df['Card_Category'].fillna(value=0, inplace=True)



for column in df.columns:
    # Calculate the mean, ignoring NaN values
    mean_value = df[column].mean()
    # Replace NaN values with the mean
    df[column].fillna(mean_value, inplace=True)

df = df[df['Customer_Age'] <= 75]
df = shuffle(df)
train_df = df.iloc[:9000, :]
test_df = df.iloc[9000:, :]

max_value = train_df['Customer_Age'].max()
train_df['Customer_Age'] = train_df['Customer_Age'] / max_value
test_df['Customer_Age'] = test_df['Customer_Age'] / max_value

max_value = train_df['Months_on_book'].max()
train_df['Months_on_book'] = train_df['Months_on_book'] / max_value
test_df['Months_on_book'] = test_df['Months_on_book'] / max_value



for column in train_df.columns[11:-1]:
    max_value = train_df[column].max()
    train_df[column] = train_df[column] / max_value
    test_df[column] = test_df[column] / max_value


train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)


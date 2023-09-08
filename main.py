import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import RandomForestClassifier

# Load datasets
# 891 rows, 12 columns
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Data preprocessing (modify data by relevance)
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
train = train.drop(['PassengerId'], axis=1)

# Create new feature Deck
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train, test]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)

# Drop cabin feature since it is not needed anymore
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
# Fill values for Embarked if NULL
train = train.fillna({'Embarked': 'S'})

data = [train, test]
for dataset in data:
    mean = train["Age"].mean()
    std = test["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random age between the mean and std values
    rand_age = np.random.randint(mean - std, mean + std, size= is_null)
    # fill NaN values in Age column with random age values
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train["Age"].astype(int)
    train["Age"].isnull().sum()


# Convert features to improve usability

# Convert float to int
data = [train, test]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

# Convert string to int
data = [train, test]
genders = {'male': 0, 'female': 1}

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)

# Convert Embarked to int
ports = {'S': 0, 'C': 1, 'Q': 2}
data = [train, test]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

# Convert Age to int and make categories per age
data = [train, test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


# Convert names
data = [train, test]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into int
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0. Just in case.
    dataset['Title'] = dataset['Title'].fillna(0)
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)

# print(train)

# Training the algorithm
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test = test.drop("PassengerId", axis=1).copy()
submission = test['PassengerId'].copy()
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

submission = pd.DataFrame(data={'PassengerId': submission})
submission['Survived'] = pd.DataFrame(data={'Survived': Y_prediction})

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

# Write outcome to csv file
submission.to_csv('submission.csv', index=False)


# Init figure and axis objects
f, ax = plt.subplots()

# Plot first visualization of survivors
train['Survived'].value_counts().plot.pie(
    explode=[0, 0.1], autopct='%1.1f%%'
)
ax.set_title('Survivors (1) and deceased (0)')
plt.show()
plt.clf()

# Plot: Survivors and deceased by sex
f, ax = plt.subplots()
sns.countplot(data=train, x='Sex', hue='Survived')
ax.set_title('Number of people that survived (1) or deceased (0) by sex')
ax.set_ylabel('Number of people')


# Plot all figures
plt.show()
plt.clf()








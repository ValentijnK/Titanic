'''
case 1 voorspellen over titanic
'''

import pandas as pd
import copy

train = pd.read_csv(r'C:\Users\Chong\Desktop\train.csv')
test = pd.read_csv(r'C:\Users\Chong\Desktop\test.csv')
gendersummision = pd.read_csv(r'C:\Users\Chong\Desktop\gender_submission.csv')

'''aantal na van per kolom kijken'''
count_na_train = train.isna().sum()
print(count_na_train)
count_na = test.isna().sum()
print(count_na)


'''
scatter plot van leeftijd vs survival
'''

import matplotlib.pyplot as plt


x_age = train.groupby(['Age','Sex'])['Survived'].mean().reset_index()
colors = {'male': 'red', 'female': 'blue'}
fig, ax = plt.subplots()
ax.scatter(x_age['Age'], x_age['Survived'], color=[colors[gender] for gender in x_age['Sex']])
ax.set_title('Survived vs Age')
ax.set_xlabel('Age')
ax.set_ylabel('Survived')
male_patch = plt.bar(0, 0, color='red', label='Male')
female_patch = plt.bar(0, 0, color='blue', label='Female')
plt.legend(handles=[male_patch, female_patch])
plt.show()


'''scatter plot prijs vs surived'''
x_prijs = train.groupby(['Fare','Sex'])['Survived'].sum().reset_index()
fig, ax = plt.subplots()
colors = {'male': 'red', 'female': 'blue'}

ax.scatter(x_prijs['Fare'], x_prijs['Survived'], color=[colors[gender] for gender in x_prijs['Sex']])
ax.set_title('Survived vs prijs')
ax.set_xlabel('prijs')
ax.set_ylabel('Survived')
male_patch = plt.bar(0, 0, color='red', label='Male')
female_patch = plt.bar(0, 0, color='blue', label='Female')
plt.legend(handles=[male_patch, female_patch])
plt.show()


'''
histogram van leeftijd vs survival
'''
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# Filter de data voor vrouwen en mannen
women = train[train['Sex'] == 'female']
men = train[train['Sex'] == 'male']

# Plot overlevenden (Survived=1) voor vrouwen
axes[0].hist(women[women['Survived'] == 1].Age.dropna(), bins=10, label='survived', alpha=0.5)
# Plot niet-overlevenden (Survived=0) voor vrouwen
axes[0].hist(women[women['Survived'] == 0].Age.dropna(), bins=10, label='not survived', alpha=0.5)
axes[0].legend()
axes[0].set_title('Female')

# Plot overlevenden (Survived=1) voor mannen
axes[1].hist(men[men['Survived'] == 1].Age.dropna(), bins=10, label='survived', alpha=0.5)
# Plot niet-overlevenden (Survived=0) voor mannen
axes[1].hist(men[men['Survived'] == 0].Age.dropna(), bins=10, label='not survived', alpha=0.5)
axes[1].legend()
axes[1].set_title('Male')

plt.suptitle('Leeftijdsverdeling van Overlevenden en Niet-overlevenden per Geslacht')
plt.show()


'''
boxplot van prijs vs survival
'''
y_1 = train[train['Survived'] == 1]
y_0 = train[train['Survived'] == 0]

plt.boxplot([y_1['Fare'], y_0['Fare']], labels=['y_1', 'y_0'])
'''
gemiddelde aangeven
'''

plt.axhline(y_1['Fare'].mean(), color='red', linestyle='dashed', linewidth=2, label='Gemiddelde y_1')
plt.axhline(y_0['Fare'].mean(), color='blue', linestyle='dashed', linewidth=2, label='Gemiddelde y_0')

plt.xlabel('Overleefd (Survived)')
plt.ylabel('Ticket Prijs')
plt.title('Boxplot van ticket prijs op basis van survived')
plt.legend()
plt.show()


'''uitbijter van boxplot verwijderen'''
Q1 = y_1['Fare'].quantile(0.25)
Q3 = y_0['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Verwijder uitschieters uit de dataset
y_1_c = y_1[(y_1['Fare'] >= lower_bound) & (y_1['Fare'] <= upper_bound)]
y_0_c = y_0[(y_0['Fare'] >= lower_bound) & (y_0['Fare'] <= upper_bound)]

plt.boxplot([y_1_c['Fare'], y_0_c['Fare']], labels=['y_1', 'y_0'])

'''
gemiddelde aangeven
'''

plt.axhline(y_1_c['Fare'].mean(), color='red', linestyle='dashed', linewidth=2, label='Gemiddelde y_1')
plt.axhline(y_0_c['Fare'].mean(), color='blue', linestyle='dashed', linewidth=2, label='Gemiddelde y_0')

plt.xlabel('Overleefd (Survived)')
plt.ylabel('Ticket Prijs')
plt.title('Boxplot van ticket prijs op basis van survived')
plt.legend()
plt.show()


'''
barplot van embarked vs survived
'''
x_embarked = train.groupby(['Embarked','Sex'])['Survived'].mean().reset_index()
colors = {'male': 'red', 'female': 'blue'}

# Maak een barplot met matplotlib
plt.bar(x_embarked['Embarked'], x_embarked['Survived'],color=[colors[gender] for gender in x_embarked['Sex']])
plt.xlabel('Instapplaats')
plt.ylabel('Gemiddeld Overlevingspercentage')
plt.title('Gemiddeld Overlevingspercentage per Instapplaats')

male_patch = plt.bar(0, 0, color='red', label='Male')
female_patch = plt.bar(0, 0, color='blue', label='Female')
plt.legend(handles=[male_patch, female_patch])

plt.show()




'''
barplot class vs survived
'''
x_Pclass = train.groupby(['Pclass','Sex'])['Survived'].mean().reset_index()
colors = {'male': 'red', 'female': 'blue'}

plt.bar(x_Pclass['Pclass'], x_Pclass['Survived'],color=[colors[gender] for gender in x_Pclass['Sex']])
plt.xlabel('class')
plt.ylabel('Gemiddeld Overlevingspercentage')
plt.title('Gemiddeld Overlevingspercentage per class')

male_patch = plt.bar(0, 0, color='red', label='Male')
female_patch = plt.bar(0, 0, color='blue', label='Female')
plt.legend(handles=[male_patch, female_patch])

plt.show()



'''niet relevant kolom verwijderen'''

x_train = copy.deepcopy(train)
x_train = x_train.drop(['PassengerId'],axis = 1)
x_train = x_train.drop(['Name'],axis = 1)
x_train = x_train.drop(['Ticket'],axis = 1)
x_train = x_train.drop(['Cabin'],axis = 1)



x_test = copy.deepcopy(test)
x_test = x_test.drop(['PassengerId'],axis = 1)
x_test = x_test.drop(['Name'],axis = 1)
x_test = x_test.drop(['Ticket'],axis = 1)
x_test = x_test.drop(['Cabin'],axis = 1)




'''bedrag afronden, van float naar int'''
data = [x_train, x_test]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

'''data verwerken van str naar int'''
data = [x_train, x_test]
sex = {'male': 0, 'female': 1}
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(sex)

ports = {'S': 0, 'C': 1, 'Q': 2}
data = [x_train, x_test]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
    


#Decision tree bouwen

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

import numpy as np
'''na van age invullen'''
'''test set'''
train_mean = x_train["Age"].mean()
train_std = x_train["Age"].std()
train_is_null = x_train["Age"].isnull().sum()
# compute random numbers between the mean, std and is_null
rand_age_test = np.random.randint(train_mean - train_std, train_mean + train_std, size = train_is_null)
# fill NaN values in Age column with random values generated
age_slice = x_train["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age_test
x_train["Age"] = age_slice
x_train["Age"] = x_train["Age"].astype(int)
x_train["Age"].isnull().sum()

'''test set'''
test_mean = x_test["Age"].mean()
test_std = x_test["Age"].std()
test_is_null = x_test["Age"].isnull().sum()
# compute random numbers between the mean, std and is_null
rand_age_train = np.random.randint(test_mean - test_std, test_mean + test_std, size = test_is_null)
# fill NaN values in Age column with random values generated
age_slice = x_test["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age_train
x_test["Age"] = age_slice
x_test["Age"] = x_test["Age"].astype(int)
x_test["Age"].isnull().sum()


x_train = x_train.dropna(subset=['Embarked'])
y_train = x_train['Survived']
x_train = x_train.drop(['Survived'],axis = 1)

#data splisten 70% trainen en 30% testen
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state = 1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

    








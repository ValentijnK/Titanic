# Titanic Challenge
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

## Data
In this competition, you’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc. One dataset is titled train.csv and the other is titled test.csv.

Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.

The test.csv dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s your job to predict these outcomes.

Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.

| Variable    | Definition              | Levels                                         |
|-------------|-------------------------|------------------------------------------------|
| PassengerID | Unique ID passenger     |                                                |
| Survival    | Survival                | 0 = no, 1 = yes                                |
| Pclass      | Ticket Class            | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| Name        | Name of passenger       |                                                |
| Sex         | Sex of passenger        | 'male' or 'female'                             |
| Age         | Age in years            |                                                |
| Sibsp       | # of siblings / spouses |                                                |
| Parch       | # of parents / children |                                                |
| Ticket      | Ticket number           |                                                |
| Fare        | Fare of ticket          |                                                |
| Cabin       | Cabin number            |                                                |
| Embarked    | Port of embarkation     | C = Cherbourg, Q = Queenstown, S = Southampton |

## Goal
It is your job to predict if a passenger survived the sinking of the Titanic or not.
For each in the test set, you must predict a 0 or 1 value for the variable.

Your score is the percentage of passengers you correctly predict. This is known as accuracy.

## Leaderboard
| Scores     |
|------------|
| 1. 72,248% |
| 2. ...     |
| 3. ...     |

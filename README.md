# COGS-189-Final-Project-
This is the final project for COGS 189 WI23. Author Jinyi Zhao. 
# COGS 118A Final Project - Popularity of Games Predictions

# Names

- Chenxi Li (A16341810)
- Jialong Guo (A15851883)
- Gege Bei(A16356724)
- Sikai Liang(A16839298)
- Jinyi Zhao (A16315154)

# Abstract 

The problem of our project is: ***does a specific game have high popularity?*** The popularity of a game could be quite abstract, since there are plenty of criteria that could be used to define whether a game has high popularity or low popularity. In our project, we will be using game ratings as the indicator of whether a game has high or low popularity, and the goal of our project is to create a relatively objective universal model to predict the popularity status of a new game that is not included in our acquired datasets below. The **algorithms** that we will be using are **logistic regression, KNN, decision tree, and random forest**. We will use feature selections to choose the variables that are most important. The game ratings represent the variables that we will predict, since we use the game rating to define high/low popularity of games. Moreover, the game ratings and the supported systems(e.g. windows) will be represented as binary variables for our models. In addition to these variables, the variables such as positive ratio, user reviews, and discount will be statistical variables, and the statistical variables and the supported systems will be the data that we used to predict the gaming ratings(popularity). We will divide this data into training sets (validation set), and test sets to create and train our model to fit as closely as possible to the dataset. Finally, the performance will be measured using the f1 score, accuracy, recall, cross-validation, and ROC-AUC curve.

# Background

Personal Computer (PC) video games, as one of the primary forms of entertainment of the public, have long been drawing people’s attention due to their highly engaging and versatile nature. According to the latest gaming statistics, the global video game industry has accumulated a tremendous consumer base (ie. approximately 38% of the worldwide population has video gaming experiences), forming a total market size, measured by yielded revenues, of approximately $106.8 billions in 2023.<sup>[1](#IndustryNote)</sup>

Moreover, the global video game industry has still been undergoing a developmental stage of rapid growth due to high market demand. The market size of the global video game industry, as the statistics indicated, is estimated to increase 9.4% over the year of 2023 and reach approximately 135 billion dollars by 2025, in which the PC gaming sector alone will accumulate around $46.7 billion.<sup>[2](#KaiNote1)</sup> Through this tremendous consumer base and the capability of creating immersive, captivating worlds that enable individuals to explore various cultures and experiences, the video game industry undeniably has crucial impacts on people’s entertainment and social & cultural trends in the modern era.
Unlike the video game market back in early 2000s where the majority sales occurred at retail stores, the digital distribution platforms, such as Steam, GamersGate, Microsoft Store, EA Origin, and so forth, have been contributing to the majority (approximately 83%) of the global video game sales nowadays <sup>[3](#KaiNotes2)</sup>. Steam, as the top digital distribution platform in terms of video game sales and number of users, for instance, accounts for approximately 50% to 70% of the entire PC video game downloads as well as 75% of the global vertical market <sup>[4](#EladNotes)</sup>. Such digital distribution platforms employ a series of marketing and sale strategies/practices that satisfy their active users’ demand as well as maximize their revenues and market shares. Among such business practices, building and employing a reliable, efficient model, which is capable of yielding sound estimations of popularity of video games, play an essential role in optimizing business operations. Therefore, in order to not only acquire a complete knowledge of the future development of the video game industry but also generate accurate recommendations of the products (ie. games), which ensures promising sales, it is absolutely necessary to establish an effective evaluation mechanism as mentioned above. However, predicting the popularity of the games published can be a vague and extremely difficult task due to the fact that various quantitative factors, such as the game prices, users’ playtimes, number of downloads, etc., could potentially have significant impacts on the popularity of games. Therefore, it is pivotal to predict the popularity of games based on a specified, subjective metric and incorporate the aforementioned quantitative factors in the analysis. 
Quite a few prior studies have gone on in the research area of the popularity prediction of  PC video games. The “Machine Learning for Predicting Success of Video Games,” for instance, investigates & analyzes data, which are acquired through Steam Spy (“a service that tracks the number of owners of each game” and Steam Charts (“a database that collects data about concurrent players”), including game price, supported language, price overview, reviews, and so forth (numerous data are being analyzed in this research, and those that are also involved in the analysis of this project is listed here). <sup>[5](#MichalNotes1)</sup> According to the author, success of the game is defined as  “the number of owners after two months since release for the data from Steam Spy” as well as “the average number of concurrent players in two months after release for Steam Charts”. <sup>[5](#MichalNotes1)</sup> In total, data of 9780 games are employed in the research, which are then cleaned and restructured through raw data processing. Besides, comparison between the data acquired from Steam Charts and Steam Spy is performed for further analysis. As discussed in the experiment section of this paper, the data is split into training, validation, and test set, and preprocessing dependent on these splits is then conducted individually. It is worth mentioning that cross-validation is not employed in the experiment, since it, according to the author, will lead to the issue of “predicting old games from new ones in numerous cases”.  <sup>[5](#MichalNotes1)</sup> When constructing the predictive models, the author employed regression and binary classification with three different settings “depending on the threshold where games are split (ie. >1, >10, >100 players)”: regression aims to predict a numeric value of the exact number of average concurrent players, and binary classification separates games into two categories based on the original value of “Players” (ie. a continuous class attribute running from 0 to 135300). <sup>[5](#MichalNotes1)</sup>  Five algorithms are employed in the regression stage: linear model, recursive partitioning and regression tree, random forest, Gaussian process, and support vector machine. According to the author, data splits contain 214 attributes, and Chi-squared is utilized to evaluate the importance of all attributes. Based on the average players prediction results via regression, random forest scores the best results with approximately 72% root relative squared error and 0.7 correlation, closely followed by support vector machine, which yields 74% root relative squared error and 0.7 correlation.  <sup>[5](#MichalNotes1)</sup> However, both aforementioned algorithms encounter difficulty with under-estimating games with higher numbers of average concurrent players, yet this is “ an understandable error as games may gain popularity by investing into promotions around release, attracting the interest of content creators after release etc.”  <sup>[5](#MichalNotes1)</sup>. On the other hand, classification aims to detect games, which attained more average concurrent players than a certain threshold. Also, five algorithms are employed in the regression stage: recursive partitioning and regression tree, general linear model, random forest, support vector machine, and naive bayes. Based on the results, the baseline accuracy for predicting games with more than 1, 10, and 100 players on average are respectively as follows: 59.3%, 79.1%, and 95%. In predicting games with more than 100 players on average, the support vector machine scores the highest precision, closely followed by random forest, which has relatively higher recall than SVM.<sup>[5](#MichalNotes1)</sup> In conclusion, regression demonstrates a strong correlation between the average number of concurrent players and core game features. Furthermore, detecting the more successful game is “possible with relatively high precision but low recall.” However, the accuracy of predictions is higher for certain games yet lower for others. As a result, a subset of games that covers one third of the test data is employed through limiting it to games from publishers who released at least two games previously. The prediction of games in this subset is shown to be relatively more reliable and accurate. 
Even though this prior research focuses on the “success” of PC games released on Steam, it is still highly relevant to our project and provides us with quite a few critical insights as it employs certain data and machine learning techniques that will also be incorporated in our project. Besides, based on numerous real-world instances, “success” and “popularity” of games are highly associated with each other: those that gain high popularity among the customers often tend to be the “successful” games as defined in the aforementioned research. Our project employs a similar approach (ie. establishing an evaluation mechanism that accurately and objectively predicts the popularity of PC video games on Steam based on based on the estimated rating of the game), examining a collection of statistical information of the game, including the game's supportive system, required age, sale information (ie. price & discount), and customers' review information & playtime as well as the number of users who downloaded the game and employing the machine learning techniques of logistic regression, random forest, decision tree, and KNN. Through this project, we aim to facilitate the customer targeting process of the digital PC video game distribution platforms, aiding them with more time-saving, accurate user-specific video game popularity prediction algorithms, as well as provide future potential customers with objective popularity feedback. 


Here is an example of inline citation. After government genocide in the 20th century, real birds were replaced with surveillance drones designed to look just like birds. Use a minimum of 2 or 3 citations, but we prefer more. You need enough citations to fully explain and back up important facts. 

Remeber you are trying to explain why someone would want to answer your question or why your hypothesis is in the form that you've stated. 

# Problem Statement

How can we accurately predict the rating of a new game by constructing algorithm models like logistic regression/KNN/random forest etc. based on metrics of the game's operational system, discount, player’s average playtime, average owner numbers and the price it listed?


We plan to quality the popularity of a game as reaching to the level of mostly positive in its Steam game rating record. The result should be discrete instead of continuous, as shown by the separate meanings behind each individual level.  The result should also be measurable by graphing out using the classification model. The ROC curve/F1 score/precision and other metrics will likely show us the accuracy of prediction and indicate the effectiveness of our classification model. Lastly, the problem will likely be replicable since the model is dependent on time and any change on the dataset (e.g. the appearance of a new factor will likely alter the regression of our model). Predicting the popularity of a game can have several significant implications for game developers, publishers, and marketers. Creating games requires a significant investment of both time and money. By predicting the popularity of a new game, game developers and publishers can make informed choices regarding how much to spend on advertising and which features to prioritize during the development process. Estimating the potential success of a new game can offer game developers and publishers an edge over their competitors. By projecting which features or gameplay elements will be favored by players, they can design a game that distinguishes itself from other games in the market and draws in a substantial player base.

# Data

Link to the dataset: 1. https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam             
                     2.  www.kaggle.com/datasets/nikdavis/steam-store-games?resource=download&select=steam.csv

We are using two datasets for our analysis: Game Recommendations on Steam and Steam Store Games. The first dataset contains 46,068 unique observations and 13 variables, while the second dataset contains 27,033 unique observations and 18 variables. Both datasets collected data from games on Steam but include different variables. We combined the two datasets by filtering the same IDs using the app ID column. The newly formed dataset has around 18,000 data points and 25 variables. To avoid redundancy, we dropped several overlapping variables, such as release date, leaving us with approximately 15 usable variables.

An observation in our dataset consists of:

- the name of a specific game
- date released
- which PC system supports the game(win/mac/linux): it can be one/two/all and stores as Boolean variable
- game rating(categorical variable such as positive/very positive/negative)
- positive ratio: percentage of positive feedbacks
- user reviews: amount of user reviews left
- price of a specific game
- average and median play time of a game
- game developer/game publisher/game categories/ game genres

Feature selection: During the data cleaning and wrangling, we will select a subset of the most relevant features. The critical variables would be rating for games, user reviews, price, average play time, average/median play time of a game, discount, supported system, average owners. For categorical variables like game developers, and game genres, we will drop those datapoints, since they have too many different values stored in one column(over 100). It may not be practical to use one hot encoding, which can lead to a very large number of columns and cause problems such as high computational cost. Also, the supported system variables(win, mac, linux) have already been one-hot-coding with ‘True’ and ‘False’, we will change it into 0/1 for each column.


Our dataset includes a categorical variable representing game ratings, which has 9 different categories ranging from overwhelmingly positive to overwhelmingly negative. To simplify the data and make it more manageable, we plan to convert this variable into a binary variable with values of 0 or 1. Specifically, we will set games with overwhelmingly positive or very positive ratings as 1, and all others as 0. Our decision to use these ratings as a threshold is based on research we conducted on the Steam scoring system, which revealed that those two ratings are typically given to games with positive feedback ratings of 85% or higher. We believe that this threshold is a relatively accurate way to assess a game's popularity based on user reception.

To address the class imbalance issues, we will use advanced algorithms, such as Cost-Sensitive Learning, Adaptive Boosting, and Random Forest, and use appropriate evaluation metrics to better evaluate the model's performance on both classes like precision, recall, and F1 score.

# Proposed Solution

We will split the data into a training set and test set, and the percentage of each set would be based on the default value which is 20% test set, and 80% training set, then the model will be trained on the selected training set, and the performance of the model will be evaluated based on the metric systems. In addition, we will use k-fold validation as the cross-validation method for our models. The reason for this is k-fold-validation is more reliable than single-split method due to the fact that k-fold validation would take each fold to be the test set, so we will get a more complete result, however, k-fold validation could be computational expensive and it could be subject to overfitting. We will be using feature selections to filter out the variables that are considered to be less important, therefore, the variables that are left will be the variables that are considered to be more related to what we will predict. Our proposed solution for the problem of our project is to use 4 different algorithms, including logistic regression, KNN, decision tree, and random forest to create 4 different models, we will transform the game rating variables into binary variables, and we will be using “very positive” which means if a game has gaming rating of or above “very positive”, it will be 1, and if a game has game rating below “very positive”, it will be -1, and this will be the y-axis of our models. The variables other than the game rating will be on the x-axis, since they are the variables that we need to use to predict the popularity of games. We will use the metric systems to evaluate the models, and select the best model among the 4 models. Finally, we will use the selected best model to predict the popularity status of a new game that is not included in our acquired datasets.
 - Logistic regression: logistic regression could be expressed as f(x;w,b)={+1 if 1/(1+e^-(wTx+b))>=0.5, -1 otherwise), just like what the formula illustrates, logistic regression will output a binary result, which is 1 or -1. We will distribute the data into training sets and test sets, and we will use the training set to fit into our logistic regression algorithm, and then we will be using the test set to evaluate the performance of our logistic regression algorithm. After the logistic regression model is trained and the performance has been evaluated, we can make predictions on the popularity status of a new game by calculating the probability using the logistic function, and finally we could compare the results with the threshold, which is typically 0.5, to classify the final prediction, which would be 1 or -1, 1 represents the game has high popularity, and -1 represents the game has low popularity.
 - KNN: KNN is an algorithm that could be used for classification problems, it is a non-parametric model. We will split the dataset into training sets and test sets, then we would use cross-validation to determine the hyperparameter, which is the number of neighbors K. Moreover, we will determine the k nearest neighbors by calculating the distance(e.g.euclidean distance, manhattan distance) between the data point in training sets and the data point in test sets. Then the datapoint in the test set will be classified to the class that is most common among the K neighbors. KNN is quite a simple algorithm, however, it has drawbacks when encountering a large dataset, since it needs to calculate the distance between all pairs of the data points in the training sets.
 - Decision tree: Decision tree is a tree-like model that contains nodes, representing the decision points; branches,representing the variety of decision paths; and leaves, representing the final outcome. Decision trees could be used for classification problems. We will split the dataset into training sets and test sets, and we will create the tree by using the training data. Firstly, we will need to do the selection of a root node, which is our project problem, “does a specific game have high popularity?”, and the tree will be built recursively through dividing the data into subsets based on the selected root node, then the process will continue until the final outcome is predicted. Decision tree is quite intuitive, however, it is also prone to overfit, so proper tree pruning might be necessary.
 - Random forest: Random forest is an ensemble learning algorithm that contains multiple decision trees. It could also be used for classification problems. We will split the dataset into training sets and test sets, and we will select random samples from the dataset to create the subsets of the data, and then we will construct decision trees based on each subset of the data. Moreover, we will be using random forest which is summing up the predictions of the decision trees to predict the output for new data points, hence the popularity status of the games. Then we will evaluate the performance of our random forest algorithm by comparing the predicted output values to the actual output values.


# Evaluation Metrics

The evaluation metric that we chose is the confusion matrix. First, let's define the terms used in a confusion matrix:
- True Positive (TP): The model correctly predicts that the game is popular.
- False Positive (FP): The model incorrectly predicts that the game is popular.
- True Negative (TN): The model correctly predicts that the game is not popular.
- False Negative (FN): The model incorrectly predicts that the game is not popular.
Using the confusion matrix, we can calculate several performance metrics to evaluate the model.


MSE: MSE stands for Mean Squared Error, it measures the average squared difference between the predicted values and the true values, and it is suitable for evaluating the performance of a regression model.
Accuracy: The proportion of correct predictions made by the model. It is calculated as (TP + TN) / (TP + FP + TN + FN).
Precision: The proportion of positive predictions that are actually true. It is calculated as TP / (TP + FP).
Recall (or Sensitivity): The proportion of actual positives that are correctly identified by the model. It is calculated as TP / (TP + FN).
F1 Score: The harmonic mean of precision and recall. It is calculated as 2 * (Precision * Recall) / (Precision + Recall).

From these metrics, if the model has MSE of x, then the predictions are off by √x if  the model has an accuracy of x%, which means that x% of the predictions made by the model are correct. The precision of the model is x%, which means that when the model predicts that a game is popular, it is correct x% of the time. The recall (or sensitivity) of the model is x%, which means that the model correctly identifies x% of the popular games. The F1 Score is x%, which is the harmonic mean of precision and recall, and provides a balanced measure of the model's performance.

In this project, we specifically focus on the F1 score, as we learned in the class, the F1 score is used to evaluate the performance of a binary classification model, which fits our model(the intended solution) perfectly, since we are trying to generate a model that could classify either a game is popular or not, and unlike other metrics, F1 scores takes both precision and recall into account, which means it is the most balanced metric among them. This is important in our case, since the false positives and false negatives are equally important. False positives in this case represents the game that is actually not popular but is classified as a popular game, which would increase the potential buyers of the game, and also increase the potential financial loss to the buyers since the game might not be worth buying; and the false negatives indicate that the game is actually a popular game but the model classified it as an unpopular game, which would potentially cause financial loss to the companies since the number of potential buyers would decrease. Therefore, the F1 score would be the best option since it could tell us how the model is performing based on these two factors(FN and FP). The mathematical representations of this metric is F1 = 2 * (precision * recall) / (precision + recall), and the metric is derived from the harmonic mean of precision and recall.


In general, a good model should have lower MSE, high accuracy, precision, and recall, and a high F1 Score. However, the choice of which metric to prioritize depends on the specific problem and the costs associated with false positives and false negatives. For example, if the cost of a false positive is higher than the cost of a false negative, then precision may be more important than recall.


# Results

## Data cleaning and wrangling 

# import the library and packages that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read and name the csv files
data1 = pd.read_csv('games.csv')
data2 = pd.read_csv('steam.csv')
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

After we load the dataset, we will first drop all the missing data points

df1 = df1.dropna()
df2 = df2.dropna()
df2 = df2.rename(columns={'appid': 'app_id'})

Since both datasets are collected from Steam and have the app ID variable, we will merge the overlapping data into a single dataset

# merge two datasets
merged_df = pd.merge(df1, df2, on='app_id')

We will drop the columns that are not relevant to our research questions or cannot be one-hot encoded.

df = pd.DataFrame(merged_df)
df = df.drop(['app_id','positive_ratio','english','categories','release_date','publisher','developer','genres','title','platforms','achievements','steam_deck','steamspy_tags','price_original','price_final','positive_ratings','negative_ratings'],axis=1)


df = df.dropna()

name = df.columns[8]

# Move the 'name' column to the first position
df.insert(0, name, df.pop(name))

df['win'] = df['win'].astype(int)
df['mac'] = df['mac'].astype(int)
df['linux'] = df['linux'].astype(int)

Since the dataset only provides a range of owners, we will split the values stored as minimum and maximum in this column and calculate the average number of owners using these values

df[['min_value', 'max_value']] = df['owners'].str.split('-', expand=True)

# Convert columns to numeric type
df['min_value'] = pd.to_numeric(df['min_value'])
df['max_value'] = pd.to_numeric(df['max_value'])

# Calculate average for each row
df['owners average'] = (df['min_value'] + df['max_value']) / 2

#drop the columns that we do not need
df = df.drop(['name','date_release',"owners"],axis=1)

# check how many different values stored in "rating" column
print(df['rating'].value_counts())

df['rating'] = df['rating'].apply(lambda x: 1 if x in ["Very Positive", "Overwhelmingly Positive"] else 0)

# Print the first 10 rows of the dataframe with the new binary target variable
df

## EDA

sns.countplot(data=df, x="rating")

Use pairplot to identify any patterns or relationships between the different variables in the dataset. 

scatter = df[['rating', 'user_reviews', 'discount', 'average_playtime', 'price','owners average']]
sns.pairplot(scatter, hue="rating")

Use heatmap to show any patterns or relationships between the different variables in the dataset. 

features = list(df.columns)
plt.figure(figsize = (12, 10))
sns.heatmap(data = df[features].corr(), annot = True, cmap='Blues')

# Model Selection

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import validation_curve


- Train/Test Split

X = df.drop('rating',axis=1) # input features
y = df["rating"] # target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Logistic Regression Classifier With Evaluation Metrics

*Some of the solvers with different penalties have the exact same model score, which would make the evaluation metrics to generate the exact same results. Therefore, we only implement the first ones of the same models.

- K-Fold Cross-Validation & Grid search

lr = LogisticRegression(random_state=0)
# Define the range of hyperparameters to test
param_grid = {'C': [0.001, 0.01, 0.1, 1], 'penalty': ['l1', 'l2']}

# Define the cross-validation method
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Use grid search to find the best hyperparameters
grid_search = GridSearchCV(lr, param_grid, cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Use the best hyperparameters to fit the logistic regression model
lr_best = LogisticRegression(**grid_search.best_params_, random_state=0)
lr_best.fit(X_train, y_train)


1.Logistic Regression method with solver 'lbfgs' and penalty 'l2'

lr = LogisticRegression(C=1, random_state=0, solver = 'lbfgs',penalty='l2')
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print("Model score: {:.3f}".format(score))

- Confusion Matrix

# Generate the confusion matrix

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred_test)


disp = ConfusionMatrixDisplay(cm);
disp.plot()

accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy:', accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred_test)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred_test)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_test)
print('F1 score:', f1)

- ROC-AUC Curve

y_scores = lr.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print('ROC AUC score:', roc_auc)

plt.plot(fpr, tpr, color='pink', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' ROC Curve')
plt.legend(loc="lower right")
plt.show()

- Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

auc_score = auc(recall, precision)

# Plot the precision-recall curve
plt.plot(recall, precision, color='skyblue', lw=2, label='Precision-Recall Curve (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()


2.Logistic Regression method with solver 'lbfgs' and penalty 'none'

lr = LogisticRegression(C=1, random_state=0, solver = 'lbfgs',penalty = 'none')
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print("Model score: {:.3f}".format(score))

3.Logistic Regression method with solver 'saga' and penalty 'none'

lr = LogisticRegression(C=1, random_state=0, solver = 'saga',penalty = 'none')
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print("Model score: {:.3f}".format(score))

- Confusion Matrix

# Generate the confusion matrix

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred_test)


disp = ConfusionMatrixDisplay(cm);
disp.plot()

accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy:', accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred_test)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred_test)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_test)
print('F1 score:', f1)

- ROC-AUC Curve

y_scores = lr.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print('ROC AUC score:', roc_auc)

plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' ROC Curve')
plt.legend(loc="lower right")
plt.show()

- Precison-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

auc_score = auc(recall, precision)

# Plot the precision-recall curve
plt.plot(recall, precision, color='pink', lw=2, label='Percision-Recall Curve (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()


4.Logistic Regression method with solver 'saga' and penalty 'l1'

lr = LogisticRegression(C=1, random_state=0, solver = 'saga',penalty = 'l1')
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print("Model score: {:.3f}".format(score))

5.Logistic Regression method with solver 'saga' and penalty 'elasticnet'

lr = LogisticRegression(C=1, random_state=0, solver = 'saga',penalty = 'elasticnet',l1_ratio=0.5)
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print("Model score: {:.3f}".format(score))

6.Logistic Regression method with solver 'liblinear' and penalty 'l1'

lr = LogisticRegression(C=1, random_state=0, solver = 'liblinear',penalty = 'l1')
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print("Model score: {:.3f}".format(score))

- Confusion Matrix

# Generate the confusion matrix

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred_test)


disp = ConfusionMatrixDisplay(cm);
disp.plot()

accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy:', accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred_test)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred_test)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_test)
print('F1 score:', f1)

- ROC-AUC Curve

y_scores = lr.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print('ROC AUC score:', roc_auc)

plt.plot(fpr, tpr, color='yellow', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' ROC Curve')
plt.legend(loc="lower right")
plt.show()

- Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

auc_score = auc(recall, precision)

# Plot the precision-recall curve
plt.plot(recall, precision, color='skyblue', lw=2, label='Precision-Recall Curve (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()


7.Logistic Regression method with solver 'liblinear' and penalty 'l2'

lr = LogisticRegression(C=1, random_state=0, solver = 'liblinear',penalty = 'l2')
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print("Model score: {:.3f}".format(score))

- Confusion Matrix

# Generate the confusion matrix

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred_test)


disp = ConfusionMatrixDisplay(cm);
disp.plot()

accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy:', accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred_test)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred_test)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_test)
print('F1 score:', f1)

- ROC-AUC Curve 

y_scores = lr.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print('ROC AUC score:', roc_auc)

plt.plot(fpr, tpr, color='pink', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' ROC Curve')
plt.legend(loc="lower right")
plt.show()

- Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

auc_score = auc(recall, precision)

# Plot the precision-recall curve
plt.plot(recall, precision, color='black', lw=2, label='Precision-Recall Curve (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()


# Random Forest Classifier With Evaluation Metric

*For Random Forest, We would alter the n-estimators which is the amount of trees in the Random Forest, and we will observe the differences.

rf = RandomForestClassifier(n_estimators=100, random_state=42)
# fit Random Forest on the training sets
rf.fit(X_train, y_train)
# Make predictions based on the test set
y_pred = rf.predict(X_test)

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy:', accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred_test)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred_test)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_test)
print('F1 score:', f1)

- ROC-AUC Curve

y_scores = rf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print('ROC AUC score:', roc_auc)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

- Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

auc_score = auc(recall, precision)

# Plot the precision-recall curve
plt.plot(recall, precision, color='purple', lw=2, label='Precision-Recall Curve (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()


#In this case, we changed n_estimators to 400, which means there would be 400 decision trees inside our Random Forest
rf = RandomForestClassifier(n_estimators=400, random_state=42)
# fit Random Forest on the training sets
rf.fit(X_train, y_train)
# Make predictions based on the test set
y_pred = rf.predict(X_test)

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy:', accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred_test)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred_test)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_test)
print('F1 score:', f1)

- ROC-AUC Curve

y_scores = rf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print('ROC AUC score:', roc_auc)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

- Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

auc_score = auc(recall, precision)

# Plot the precision-recall curve
plt.plot(recall, precision, color='red', lw=2, label='Percision-Recall Curve (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()


#  k-Nearest Neighbors (kNN) algorithm

knn = KNeighborsClassifier(n_neighbors=9)
#fit KNN to the training set
knn.fit(X_train, y_train)
# prediction based on the test set
y_pred = knn.predict(X_test)

y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy:', accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred_test)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred_test)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_test)
print('F1 score:', f1)

- ROC-AUC Curve

y_scores = knn.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print('ROC AUC score:', roc_auc)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

- KNN Validation Curve

p_range = [1,5,10,100]
train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, 
                                             param_name="n_neighbors", 
                                             param_range= p_range)

plt.figure(figsize=(10, 6))
plt.title("KNN Validation Curve")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(p_range)
plt.plot(p_range, np.median(train_scores, 1), color='blue', label='training score')
plt.plot(p_range, np.median(test_scores, 1), color='red', label='validation score')
plt.legend(loc="best")
plt.show()

# Decision Tree

dtree = DecisionTreeClassifier()

# fit the decision tree to the training set
dtree.fit(X_train, y_train)

# predicting based on the test set
y_pred = dtree.predict(X_test)

# evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

y_pred_train = dtree.predict(X_train)
y_pred_test = dtree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy:', accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred_test)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred_test)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_test)
print('F1 score:', f1)

- Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

auc_score = auc(recall, precision)

# Plot the precision-recall curve
plt.plot(recall, precision, color='orange', lw=2, label='Precision-Recall Curve (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()


- ROC-AUC Curve

y_scores = dtree.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print('ROC AUC score:', roc_auc)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Support Vector Machine (SVM)

# Determine the SVM by using SVC() function, also the reason for probability = True is to ensure we could use roc-auc curve to evaluate the performance of the model.
svm = SVC(probability=True)
# fit the SVM to the training set
svm.fit(X_train, y_train)
# predicting based on the test set
y_pred = svm.predict(X_test)

# evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy:', accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred_test)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred_test)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_test)
print('F1 score:', f1)

- ROC-AUC Curve

y_scores = svm.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print('ROC AUC score:', roc_auc)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

- Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

auc_score = auc(recall, precision)

# Plot the precision-recall curve
plt.plot(recall, precision, color='pink', lw=2, label='Precision-Recall Curve (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()

# Discussion

### Interpreting the result

Throughout the comparison of all the algorithms that were implemented above, we decided that Random Forest with smaller but efficient n-estimators(the amount of decision trees) is the most appropriate algorithm in our model. We will explain this conclusion in the aspect of comparing random forest to other models and comparing within the Random Forest algorithm. *Our metric systems include Accuracy, Precision, Recall, F1-score, ROC-AUC score, Precision-Recall score.
- Comparing with other models: By comparing the performance of each algorithm through multiple metric systems, it is obvious that Random Forest has significantly higher score across all metric systems than all other algorithms besides KNN algorithm, they have similar performance regarding ROC-AUC Score, however, the random forest model has better accuracy (0.778 compared to 0.768) and F-1 score (0.647 compared to 0.641), despite the other results being quite similar. Therefore, Random Forest in general has a better performance than KNN algorithm.
- Comparing within Random Forest: By increasing the number of n-estimators within the random forest function, we included more decision trees inside our random forest algorithm. However, the result across the metric systems illustrated that increasing the number of decision trees would not increase the performance of the model. While the computation cost of the model would increase due to the increasing number of decision trees.


### Limitations

Are there any problems with the work?  For instance would more data change the nature of the problem? Would it be good to explore more hyperparams than you had time for?   

### Ethics & Privacy

Dataset choice may lead to privacy concerns. The Steam dataset may contain personally identifiable information (PII) about users, such as their names, email addresses, and payment information. It is important to ensure that this data is properly anonymized and that users' privacy is protected. In order to achieve this, we carefully use the data that initially not include the private information and we also removed or anonymized them from the dataset to protect users' privacy.

The output of our model maybe biased and unfair. For biasness, the Steam datasets may contain biased data, such as games that are popular only among certain demographic groups. This could lead to biased predictions, which could have negative implications for users. For fairness, our machine learning model may have the potential to unfairly discriminate against certain groups of people. For example, if the model is trained on a dataset that is biased against women, it may make unfair predictions about games that are popular among women. In order to avoid those questions, we especially remove all the variables objectivly relate to the reason that will infuence the otput of the model. For example, we removed the genre and category of the game, the nationality of the producers, the name of the producers, and the gender of the chief producers.

It is important to be transparent about how the machine learning model works and how it makes predictions. Users should be informed about how their data is being used and how the model works. Users should also be given the opportunity to provide informed consent for their data to be used in the machine learning project. They should be informed about the purpose of the project and how their data will be used. Those are problems solved by Steam platform noticing the User by their comment annoucement and solved also by Kaggle cookies.

We do hve a potential misuse that is hard to solve. There is a risk that the machine learning model could be misused for malicious purposes, such as predicting the popularity of games that promote hate speech or other harmful content.

### Conclusion

Reiterate your main point and in just a few sentences tell us how your results support it. Mention how this work would fit in the background/context of other work in this field if you can. Suggest directions for future work if you want to. 

# Footnotes
<a name="IndustryNote">1</a>: Industry market research, reports, and Statistics (no date) IBISWorld. Available at: https://www.ibisworld.com/industry-statistics/market-size/video-games-united-states/#:~:text=The%20market%20size%2C%20measured%20by,to%20increase%209.4%25%20in%202023 (Accessed: February 22, 2023).

<a name="KaiNote1">2</a>: Kaisersays:, Milagros. “Gaming Statistics 2023.” TrueList, 9 Jan. 2023, https://truelist.co/blog/gaming-statistics/ (Accessed: February 22, 2023).

<a name="KaiNotes2">3</a>: Kaisersays:, Milagros. “Gaming Statistics 2023.” TrueList, 9 Jan. 2023, https://truelist.co/blog/gaming-statistics/ (Accessed: February 22, 2023).

<a name="EladNotes">4</a>: Elad, Barry. “25+ Steam Statistics 2022 Users, Most Played Games and Market Share.” Enterprise Apps Today, 15 Aug. 2022, https://www.enterpriseappstoday.com/stats/steam-statistics.html#:~:text=Steam%20accounts%20for%2050%25%20to,20%20million%20gamers%20every%20day (Accessed: February 22, 2023)

<a name="MichalNotes1">5</a>: Michal Trněný, "Machine Learning for Predicting Success of Video Games." Spring 2017, 
https://is.muni.cz/th/k2c5b/diploma_thesis_trneny.pdf (Accessed: March 8, 2023)

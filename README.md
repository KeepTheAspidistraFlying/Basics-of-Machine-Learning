# Basics-of-Machine-Learning
very basic but essential info on ML

Hey Guyz;

i am going to update this file with new info everytime i find something interesting, so it might get more detailed.
if u have any question or suggestion, plz contact me via my email; Hajiahmadiparisa@yahoo.com

consider these 2 types of learning;

1] supervised learning: known target based on past data

2] unsupervised learning: no known past answer

now, another type of categorizing;

a] classification: what class something belongs to

b] regression: predicting a numerical value

***************************************************************************************************************************

every algorithm is specified for a type of learning. 

for instance; "Logistic Regression", it is used for classification problems. (dont mix the name of the algorithm with its type of problem.) I used this algorithm in Titanic Survival Prediction, Breast Cancer Prediction & Bob the Builder. u can find these projects in my repos by the same name.

***************************************************************************************************************************

u can use different programming languages for implementation of these algorithms but I prefer Python.
I find Python packages very useful; 

Pandas; reading big data & data manipulation

numpy; computation of numerical data/ manipulating lists & tables of numerical data

matplotlib; graphing data

scikit-learn; machine learning models (one of the best documented Python modules)

***************************************************************************************************************************

there are 2 commonly metrics for classification; Precision and Recall

Precision; TP/(TP+FP)

Recall; TP/(TP+FN)

TP:True Positive/// FP:False Positive/// FN:False Negative/// TN:True Negative

the values of Precision & Recall that we are aiming for depends on the dataset and the application.

in the Titanic example, assume we consider those whose survival possibility is more than 80% will survive. in this case we will have less survivals but the certainty is much higher. in this case, precision is high.

now assume we consider those whose survival possibility is more than 45% will survive. in this case there is a lower chance of getting FN, but the certainty of our guess is also lower. in this case, recall is high.

F1= 2*(Precision*Recall)/(Precision+Recall)

we use a matrix called "confusion matrix" to show the 4 values of confusion (TP,FP,TN,FN).
we can import confusion_matrix from sklearn.metrics to see these 4 values.
pay attention the matrix shown by python is in this order:

TN      FP

FN      TP

alright, now remember the titanic survival, we assumed if the possibility of survival is over 50%, show 1 and if its less show 0. here, 50% is our threshold.
if we use 75% instead of 50%, we actually made the threshold higher.
what does it mean??
it means our 1s are gonna have a higher certainty but fewer 1s. (precision higher, recall lower).

each choice of a threshold is a different model.
an ROC (=Receiver Operating Characteristic) curve is a graph showing all of the possible models and their performances.
an ROC curve is a graph of "sensivity" vs. "specificity".

sensivity= recall= TP/(TP+FN)

specificity=TN/(TN+FP)

our goal is to maximize these 2 values. (generally when one goes up the other gets lower)

to calculate the value of specifity you can use the first value of the second array of "precision_recall_fscore_support" imported from sklearn.metrics

(specificity is the recall of negative class)

check Titanic Survival Prediction repo,"Titanic_Survival_Prediction_Sensitivity&Specificity.zip"

you can change the value of threshold by changing model.predict(X_test) to model.pridict_proba(X_test)[:,1]> A number between 0 and 1 (your new threshold).

as mentioned each threshold is a new model. we can build a ROC curve by roc_curve function from sckit_learn.
this function takes the true target values and the predicted probabilities from our model.

step1: use predict_proba method

step2: call roc_curve function

         it returns:

               1. an array of FP rates: 1-Specificity (X_axis)

               2. an array of TP rates: Sensitivity (Y_axis)

               3. the threshold (wont be needed in graph)

As we don’t use the threshold values to build the graph, the graph does not tell us what threshold would yield each of the possible models.

The ROC curve shows the performance, not of a single model, but of many models. Each choice of threshold is a different model.

How to choose between these models will depend on the specifics of our situation.

The closer the curve gets to the upper left corner, the better the performance.

The line should never fall below the diagonal line as that would mean it performs worse than a random model.

now, we know the curve that is above the other one is better. so the area under the better curve is higher.

we can use AUC (area under curve) to decide which model is better. use (roc_auc_score(y_test, y_pred_proba[:,1]).

the AUC does not measure the performance of a single model. It gives a general sense of how well the Logistic Regression model is performing.

u can see Titanic repo AUC_SCORE to better understand the concept of auc and its code.(python)

in test and train we devided our whole set into 2sets and then built the model with the train set and evaluated it with test set. the best model is the one that is built with whole datapoints and the whole purpose of test and train set is evaluation. as we saw before, if we use different train and test sets we find a new value for accuracy, precision, recall and f1, which is not an ideal evaluation. so we do the train and test k times and report the mean of the metrics measures for evaluation. we call this "K-fold cross validation".

note that this is ONLY for evaluation purposes. NOT building the model.

coding k-fold cross validation is straightforward. u can use "from sklearn.model_selection import KFold".

kf=KFold(n_splits= 5, shuffle=True)

list(kf.split(X))

use list() to convert generator to list. u can use any number instead of 5 but we usually use 5.

***************************************************************************************************************************

so, we've talked about logistic regression in the previous section.
we used data to build the model and then we checked the model with the same data and it did well.
but how well our model will do on new data?? if it does poorly, then "overfitting" happend.
note that the more "features" we have there's a higher chance of overfitting.

so what to do??

simple!! we devide our dataset into 2 sets; we use the first set to build a model, meaning; we train with this set. then we test the model with the second set that our model haven't seen yet. the first set is called "training set" and the second is "test set".
(we usually use 70%-80% of data for training and 20%-30% for test.)

u can use scikit-learn in Python to do training and testing.

u can also check Titanic Survival Prediction repo, I explained how to use Python for train and test. (Titanic_Survival_Prediction_Train&Test.zip)

***************************************************************************************************************************

***************************************************************************************************************************

so far we talked about logistic regression which is defined by the coefficients that define the line. These coefficients are called parameters. Since the model is defined by these parameters, Logistic Regression is a parametric machine learning algorithm.

now we wanna talk about " Decision Trees", which are an example of a nonparametric machine learning algorithm.

It is basically a flow chart of questions that you answer about a datapoint until you get to a prediction.

Gini impurity is a measure of how pure a set is.

for example in Titanic survival prediction, we calculate the gini impurity on a subset of our data based on how many datapoints in the set are passengers that survived and how many are passengers who didn’t survive. it will be a value between 0 and 0.5 where 0.5 is completely impure (50% survived and 50% didn’t survive) and 0 is completely pure (100% in the same class).

(p is the percent of passengers who survived. Thus (1-p) is the percent of passengers who didn’t survive.)

GINI=2(p)(1-p)

Entropy is another measure of purity.
for example in Titanic survival prediction, it will be a value between 0 and 1 where 1 is completely impure (50% survived and 50% didn’t survive) and 0 is completely pure (100% the same class).

entropy= -[ p log(p)[2] + (1-p) log(1-p)[2] ]

it’s not obvious whether gini or entropy is a better choice.it often won’t make a difference, but you can always cross validate to compare a Decision Tree with entropy and a Decision Tree with gini to see which performs better.

Information Gain= H(S) - H(A)*|A|/|S| - H(B)*|B|/|S|

H:impurity measure(entropy or gini)

S:original dataset

A&B:2 splitting sets from S

the greater info. gain the better it is.

To determine how to do the first split, we go over every possible split and calculate the information gain if we used that split.

in Titanic example, for numerical features like Age, PClass and Fare, we try every possible threshold. Splitting on the Age threshold of 50 means that datapoints with Age<=50 are one group and those with Age>50 are the other. so since there are 89 different ages in our dataset, we have 88 different splits to try for the age feature!

Just like with Logistic Regression, scikit-learn has a Decision Tree class. The code for building a Decision Tree model is very similar to building a Logistic Regression model, fit (to train the model), score (to calculate the accuracy score) and predict (to make predictions).

The default impurity criterion in scikit-learn’s Decision Tree algorithm is the Gini Impurity. However, they’ve also implemented entropy and you can choose which one you’d like to use when you create the DecisionTreeClassifier object.

for creating a png image of your graph, use scikit-learn's export_graphviz function.

remember overfitting?? Decision Trees are incredibly prone to overfitting. 

In order to solve these issues, we do what’s called pruning the tree. This means we make the tree smaller with the goal of reducing overfitting.

There are two types of pruning: pre-pruning & post-pruning.

1. In pre-pruning, we stop building before the tree is too big.

2. In post-pruning we build the whole tree and then we review the tree and decide which leaves to remove to make the tree smaller.

Pre-Pruning techniques:

1. Max depth

2. Leaf size

3.   Number of leaf nodes


There’s no hard science as to which pre-pruning method will yield better results. In practice, we try a few different values for each parameter and cross validate to compare their performance.

Scikit-learn has implemented quite a few techniques for pre-pruning. In particular, we will look at three of the parameters: max_depth, min_samples_leaf, and max_leaf_nodes. In order to decide on which to use, we use cross validation and compare metrics. scikit-learn has a grid search class built in that will help us.

from sklearn.model_selection import GridSearchCV

now let us talk about computation; Decision Trees are slow to train and fast to predict. Interpretability is the biggest advantage of Decision Trees.

***************************************************************************************************************************

***************************************************************************************************************************

Random Forest


A random forest is an example of an ensemble because it uses multiple machine learning models to create a single model.

 The goal of random forests is to take the advantages of decision trees while mitigating the variance issues.

BOOTSTRAPPING
 
 A bootstrapped sample is a random sample of datapoints where we randomly select with replacement datapoints from our original dataset to create a dataset of the same size. Randomly selecting with replacement means that we can choose the same datapoint multiple times. We use bootstrapping to mimic creating multiple samples.

 Bootstrap Aggregation (or Bagging) is a technique for reducing the variance in an individual model by creating an ensemble from multiple models built on bootstrapped samples. Bagging Decision Trees is a way of reducing the variance in the model.

 With bagged decision trees, the trees may still be too similar to have fully created the ideal model. They are built on different resamples, but they all have access to the same features. Thus we will add some restrictions to the model when building each decision tree so the trees have more variation. We call this decorrelating the trees. If we bag these decision trees, we get a random forest.

Each decision tree within a random forest is probably worse than a standard decision tree. But when we average them we get a very strong model!

now lets get to coding;  from sklearn.ensemble import RandomForestClassifier

Since a random forest is made up of decision trees, we have all the same tuning parameters for prepruning as we did for decision trees: max_depth, min_samples_leaf, and max_leaf_nodes. but in Random Forest generally generally overfitting isnt an issue.

lets talk about 2 new parameters; : n_estimators (the number of trees) and max_features (the number of features to consider at each split).

Increasing the number trees will increase performance until a point where it levels out. The more trees, however, the more complicated the algorithm. A more complicated algorithm is more resource intensive to use. Generally it is worth adding complexity to the model if it improves performance but we do not want to unnecessarily add complexity. we use Elbow Graph to optimize performance without adding unnecessary complexity.




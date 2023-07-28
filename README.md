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

note that this is ONLY for avaluation purposes. NOT building the model.

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

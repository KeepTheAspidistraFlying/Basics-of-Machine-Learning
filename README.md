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

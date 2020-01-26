# Fortune-Cookie-Classifier

A Fortune cookie classifier is given input as bunch of messages and this classifier will predict whether the message is
a wise saying or future prediction. We are given with 3 types of data: traning data, testing data and a list of stop words.


Step 1: Generate a vocabulary by removing stop words from all messages. Here we use the bag of words representation.(Alphabetical Order)
Step 2: Convert each message in the training set into features of size M which is the size of vocabulary. 
Step 3: Place 1 if the word is present in message and 0 if not. 
Step 4: Train the classifier using following classifiers. 



1) Naive Bayes classifier with Laplace Smoothing (Weka tools)
2) Logistic Regression Classifier (Weka Tools) 



Results: 

Training Accuracy Naive Bayes = 96.8944099378882 %
Training Accuracy Naive Bayes SK LEARN = 97.82608695652173 %
Training Accuracy Logistic Regression  SK Learn= 94.40993788819875 %
Testing Accuracy Naive Bayes= 76.23762376237624 %
Testing Accuracy Naive Bayes SK LEARN = 65.34653465346535 %
Testing Accuracy using Logistic Regression SK Learn = 79.20792079207921 %

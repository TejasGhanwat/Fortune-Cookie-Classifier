import pandas as pnd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#Declaration
train = open("traindata.txt", "r")
stop_words = open("stoplist.txt","r")
train_labels = open("trainlabels.txt", "r")
stop_list = list()
train_list = list()
vocabulary_dummy = list()
vocabulary = list()
feature_list = []
trainer_list = list()
features = []
trainlabel_list = list()
flicker = []

#Generate lists
for word in train:
	item = word.split()
	for item_more in item:
		train_list.append(item_more)
		trainer_list.append(item_more)
for word_count in stop_words:
	token = word_count.split()
	for lasun in token:
		stop_list.append(lasun)

#Construct Vocabulary list:
for word in train_list:
	if word not in stop_list:
		vocabulary_dummy.append(word)
vocabulary_dummy.sort()
for war in vocabulary_dummy:
	if war not in vocabulary:
		vocabulary.append(war)

for label in train_labels:
		trainlabel_list.extend(label)
for bug in trainlabel_list:
	if bug == "\n":
		trainlabel_list.remove(bug)

train_new = open("traindata.txt", "r")

#Create Feature List
feature_set = []
for item_1 in train_new:
	word = item_1.split()
	feature = [0] * len(vocabulary)
	for w1 in word:
		if w1 in vocabulary:
			ind = vocabulary.index(w1)
			feature[ind] = 1
	feature_set.append(feature)
#print(feature_set)
df = pnd.DataFrame(feature_set, columns=vocabulary)

#Function for calculating probablitites of labels
def labels_pred(labels):
	label_0 = 0
	label_1 = 0
	for count in labels:
		#print(count)
		if count== "0":
			label_0 += 1

		else:
			label_1 += 1
	label_0 = label_0/len(labels)
	label_1 = label_1/len(labels)


	return(label_0, label_1)

label_0, label_1 = labels_pred(trainlabel_list)

#Convert label to int
label_int = [int (x) for x in trainlabel_list]


train_y_1 = list()
train_y_0 = list()
train_raw = open("traindata.txt", "r")
counter = 0

#Split training data according to the trainlabels:

for line in train_raw:
		train_y_1.append(line)
		counter += 1
		if counter == 152:
			break

for line in train_raw:
	if line not in train_y_1:
		train_y_0.append(line)


present_1 = 0
present_0 = 0
absent_1 = 0
absent_0 = 0

present_1 = df[1:152].sum(axis=0)
present_1 = present_1.tolist()

present_0  = df[153:322].sum(axis=0)
present_0 = present_0.tolist()

absent_1 = (df[1:152] == 0).sum(axis=0)
absent_0 = (df[152:322]==0).sum(axis=0)
absent_1 = absent_1.tolist()
absent_0 = absent_0.tolist()


def Probablity_class_0(class0,class1):
	p_0 = 0
	p_1 = 0
	prob_0 = []
	prob_1 = []
	for i in range(len(class0)):
		p_0 = (class0[i] +1)/((class0[i]+class1[i])+2)
		p_1 = (class1[i] +1)/((class0[i]+class1[i])+2)
		prob_0.append(p_0)
		prob_1.append(p_1)

	return (prob_0, prob_1)
probablity_0, probablity_1= Probablity_class_0(present_0,present_1)

prod_list_0 = []
prod_list_1 = []
prince = []
product = 0
product = 1
error = []
count = 0
train_raw = open("traindata.txt", "r")
for sentence in train_raw:
	krunch = sentence.split()
	final_0=1
	final_1 = 1
	for word in krunch:
		if word in vocabulary:
			temp_0 = 1
			temp_1 = 1
			ind = vocabulary.index(word)
			temp_0 = probablity_0[ind]
			temp_1 = probablity_1[ind]
			final_0 = final_0 * temp_0
			final_1 = final_1 * temp_1
	if final_0>final_1:
		error.append(0)
	else:
		error.append(1)
for i in range(len(error)):
	if error[i]==label_int[i]:
		count += 1

accuracy = (count/len(trainlabel_list))*100
error_1 = []
test = open("testdata.txt", "r")
for sentence in test:
	jon = sentence.split()
	final_00= 1
	final_11= 1
	for word in jon:
		if word in vocabulary:
			temp_00 = 1
			temp_11 = 1
			ind = vocabulary.index(word)
			temp_00 = probablity_0[ind]
			temp_11 = probablity_1[ind]
			final_00 = final_00 * temp_00
			final_11 = final_11* temp_11
	if final_00>final_11:
		error_1.append(0)
	#print("{} is a Wise Saying".format(sentence))
	else:
		error_1.append(1)

test_label_int = []
test_label_final = []

test_labels = open("testlabels.txt", "r")
for jostle in test_labels:
	test_label_int.append(jostle)
for bug in test_label_int:
	blind = bug.split()
	for jonty in blind:
		test_label_final.append(jonty)

test_label_final = [int (x) for x in test_label_final]

counter = 0

for j in range(len(error_1)):
	if error_1[j] == test_label_final[j]:
		counter +=1
test_accuracy = (counter/len(test_label_final))*100

feature_set_1 = []
tester= open("testdata.txt", "r")
for pink in tester:
	wording = pink.split()
	featuresq = [0] * len(vocabulary)
	for w2 in wording:
		if w2 in vocabulary:
			inder = vocabulary.index(w2)
			featuresq[inder] = 1
	feature_set_1.append(featuresq)
df_test = pnd.DataFrame(feature_set_1, columns=vocabulary)

# print(test_label_final)
# print(error_1)

#Calculate accuracy using SKLearn Naive Bayes:
nb = GaussianNB()
nb.fit(df, label_int)
king = nb.predict(df)
blipper = 1
for i in range(len(label_int)):
	if label_int[i] == king[i]:
		blipper += 1
accuracy_sktrain = (blipper/len(label_int))

#Calculate testing accuracy using SKLearn :
countesh = 0
y_predict = nb.predict(df_test)
for i in range(len(test_label_final)):
	if test_label_final[i] == y_predict[i]:
		countesh += 1
accuracy_testsk = (countesh/len(test_label_final))*100

#Calculate Training Accuracy using Logistic Regression:
lr = LogisticRegression()
lr.fit(df, label_int)
accuracy_23 = lr.predict(df)
#print(accuracy)
kliber = 0
for i in range(len(label_int)):
	if label_int[i] == accuracy_23[i]:
		kliber += 1
accuracy_01 = kliber/ len(label_int)

#Calculate Testing Accuracy using Logistic Regression SKLearn :
smith = 0
korn = lr.predict(df_test)
for i in range(len(test_label_final)):
	if test_label_final[i] == korn[i]:
		smith += 1
accuracy_lrtest = smith/len(test_label_final)
#Create output file to store outputs
jack = open('output.txt', 'w')

# Print Outputs
print("Training Accuracy Naive Bayes = {} %".format(accuracy), file = jack)
print("Training Accuracy Naive Bayes SK LEARN = {} %".format(accuracy_sktrain*100), file=jack)
print("Training Accuracy Logistic Regression  SK Learn= {} %".format(accuracy_01*100), file = jack)
print("Testing Accuracy Naive Bayes= {} %".format(test_accuracy), file=jack)
print("Testing Accuracy Naive Bayes SK LEARN = {} %".format(accuracy_testsk), file = jack)
print("Testing Accuracy using Logistic Regression SK Learn = {} %".format(accuracy_lrtest*100), file=jack)

jack.close()






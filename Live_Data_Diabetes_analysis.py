import pandas as pd
import numpy as np

# Loading of dataset from file manager 
#data = pd.read_csv("diabetes.csv")
data =pd.read_csv('DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research.csv')
print(data.head())
#Dataset  preprocessing
data["gender"]=data["gender"].replace("Other","Female")
# To know all the variables in gender : print(data['gender'].unique())
gender_encode={
    'Male':0,
    'Female':1
}
diabetic_encode={
    'No' :0,
    'Yes':1
}
smoking_encode={
    'never':0,
    'not current':1,
    'former':2,
    'current':3,
    'ever':4,
    'No Info':5
}
data['gender']=data['gender'].map(gender_encode)
data['diabetic']=data["diabetic"].map(diabetic_encode)


 #Filling missing value or empty cell in that dataset
x = data["gender"].mean()
data.fillna({"gender": x}, inplace=True)


#Confirm the datatype of the col in the dataset
print(data.info())

#Confirm the Rows and cols of the dataset
print(data.shape)

print(data.describe())

#To confirm if there is an empty cell
print("Numbers of empty cell : ", data.isnull().any().sum());
print(data.isna().sum())
# Importation of models and selection model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import accuracy_score, classification_report, precision_score,precision_recall_fscore_support, roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#appending metric scores of all algorithms in array data structure
#algorithms = ['Random Forest','Logistic Regression',"SGDC",'ANN', 'Ensemble'];
algorithms = ['VotingClassifier','StackingClassifier', 'My_Model']
accuracies = []

precisions = []

recalls = []

f1_scores = []

#appending those metrics inside a function
def append_metrics(accuracy, precision, recall, f1):
	accuracies.append(accuracy)

	precisions.append(precision)

	recalls.append(recall)

	f1_scores.append(f1)
	

# I have to seprate the outcome as in our target from the rest of the vairables	
y= data["diabetic"] # I named my outcome as in target as y

#The exact line of code to separate the two and i also named it as X
x = data.drop("diabetic", axis=1)

#Confirm the roll and col of x and y(the target or outcome)
print(x.shape, y.shape)

#scale train_data
scaled_train = StandardScaler().fit_transform(x)

# Splitting the data into training and validation (testimh )
X_train, X_test, y_train, y_test = train_test_split(
    scaled_train, y, test_size=0.2, random_state=25)
    
print(y_train.shape, y_test.shape)

# building the all models
model_1 = RandomForestClassifier(random_state=25)
model_2 = LogisticRegression(random_state=25, max_iter=1000)
model_3 = SGDClassifier(random_state=25)
model_4 = MLPClassifier(
    hidden_layer_sizes=(100,),  # One hidden layer with 100 neurons
    activation='relu',                   # 'relu' activation function
    solver='adam',                      # 'adam' solver for weight optimization
    max_iter=500,                     # Set the number of iterations (epochs)
    random_state=25              # Set the seed for reproducibility
)
model_5 = LGBMClassifier()
model_6 = CatBoostClassifier()


# training
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)
model_4.fit(X_train, y_train)
model_5.fit(X_train, y_train)
model_6.fit(X_train, y_train)


# predicting
target_randomForest = model_1.predict(X_test)
target_LR= model_2.predict(X_test)
target_SGDC = model_3.predict(X_test)
target_NN = model_4.predict(X_test)
target_LGBM = model_5.predict(X_test)
target_CATBOOST = model_6.predict(X_test)

# random forest evaluation metrics
random_accuracy = accuracy_score(y_test, target_randomForest)
random_precision, random_recall,random_f1, _ = precision_recall_fscore_support(y_test, target_randomForest, average='weighted')
#append_metrics(random_accuracy, random_precision, random_recall, random_f1)
print("Random Forest Accuracy : ",random_accuracy)

#Logistics Regression metrics
LR_accuracy = accuracy_score(y_test, target_LR)
LR_precision, LR_recall,LR_f1, _ = precision_recall_fscore_support(y_test, target_LR, average='weighted')
#append_metrics(LR_accuracy, LR_precision, LR_recall, LR_f1)
print("LR accuracy : ", LR_accuracy)

#SGDC  evaluation metrics
SGDC_accuracy = accuracy_score(y_test, target_SGDC)
SGDC_precision, SGDC_recall,SGDC_f1, _ = precision_recall_fscore_support(y_test, target_SGDC, average='weighted')
#append_metrics(SGDC_accuracy, SGDC_precision, SGDC_recall, SGDC_f1)
print("SGDC accuracy : ", SGDC_accuracy)

#NEURAL NETWORK EVALUATION METRICS
NN_accuracy = accuracy_score(y_test, target_NN)
NN_precision, NN_recall,NN_f1, _ = precision_recall_fscore_support(y_test, target_NN, average='weighted')
#append_metrics(NN_accuracy, NN_precision, NN_recall, NN_f1)
print("Artificial Neural Network Accuracy : ",NN_accuracy)

# LIGHTGBMClassifier evaluation metrics
lgbm_accuracy = accuracy_score(y_test, target_LGBM)
lgbm_precision, lgbm_recall,lgbm_f1, _ = precision_recall_fscore_support(y_test, target_LGBM, average='weighted')
#append_metrics(random_accuracy, random_precision, random_recall, random_f1)
print("LightGBMClassifier : ", lgbm_accuracy)

# CATBOOSTClassifier evaluation metrics
catBoost_accuracy = accuracy_score(y_test, target_CATBOOST)
catBoost_precision, catBoost_recall,catBoost_f1, _ = precision_recall_fscore_support(y_test, target_CATBOOST, average='weighted')
#append_metrics(random_accuracy, random_precision, random_recall, random_f1)
print("CatBoostClassifier : ", catBoost_accuracy)

#predicting for training dataset (Not actually neccessary)
target_ran = model_1.predict(X_train)
target_L= model_2.predict(X_train)
target_Sg= model_3.predict(X_train)

#evaluation of training dataset (wont print the accuracy score out)
accuracy = round(accuracy_score(y_test, target_randomForest),3)*100
accuracy_test = round(accuracy_score(y_train, target_ran),3)*100;
#print(accuracy, accuracy_test)


#==========================
#Ensemble Method starts here 
#==========================
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. votingClassifier 
#2. StackingClassifier
#3.  My Ensemble Model

# Create VotingClassifier
# 'hard' voting: majority class label
EModel_1 = VotingClassifier(estimators=[('rf', model_1), ('lgbm', model_5), ('catboost', model_6)], voting='hard')
EModel_1 = EModel_1.fit(X_train, y_train)
target_VC = EModel_1.predict(X_test)
print(f"Hard Voting Accuracy: {accuracy_score(y_test, target_VC):.4f}")

# evaluation of voting Classifier model
VotingClassifier_accuracy = accuracy_score(y_test, target_VC)
#auc = round(roc_auc_score(y_test, pred_final), 3)
VoteClass_precision, VoteClass_recall, VoteClass_f1, _ = precision_recall_fscore_support(y_test, target_VC, average='weighted')
append_metrics(VotingClassifier_accuracy, VoteClass_precision, VoteClass_recall, VoteClass_f1)

# Stacking Classifier starts here
estimators = [
    ('lr', model_2),
    ('rf', model_1),
    ('nn', model_4),
]

# Define meta-classifier for Stacking ensemble Model
final_estimator = model_6

# Create StackingClassifier
EModel_2 = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5 # Cross-validation folds for training meta-model
)
EModel_2.fit(X_train, y_train)
target_SC = EModel_2.predict(X_test)
print(f"Stacking Classifier Accuracy: {accuracy_score(y_test, target_SC):.4f}")

# evaluation of stackClassifier model
StackingClassifier_accuracy = accuracy_score(y_test, target_SC)
#auc = round(roc_auc_score(y_test, pred_final), 3)
StackClass_precision, StackClass_recall, StackClass_f1, _ = precision_recall_fscore_support(y_test, target_SC, average='weighted')
append_metrics(StackingClassifier_accuracy, StackClass_precision, StackClass_recall, StackClass_f1)


#my model analysis Ensemble method
LIGHTGBM_weight = lgbm_accuracy/(lgbm_accuracy+catBoost_accuracy+random_accuracy)
CATBOOST_weight = catBoost_accuracy/(lgbm_accuracy+catBoost_accuracy+random_accuracy)
random_weight = random_accuracy/(lgbm_accuracy+catBoost_accuracy+random_accuracy)
print("==================================")
print(LIGHTGBM_weight,CATBOOST_weight, random_weight)
print("==================================")
my_model = np.round(LIGHTGBM_weight*target_LGBM + random_weight*target_randomForest + CATBOOST_weight*target_CATBOOST)

# evaluation of the proposed model 
my_model_accuracy = accuracy_score(y_test, my_model)
#auc = round(roc_auc_score(y_test, pred_final), 3)
my_model_precision, my_model_recall, my_model_f1, _ = precision_recall_fscore_support(y_test, my_model, average='weighted')
append_metrics(my_model_accuracy, my_model_precision, my_model_recall, my_model_f1)
print("My model accuracy : ", my_model_accuracy)

#Showing Higher metrics comparison
results_df = pd.DataFrame({

'Algorithm': algorithms,

'Accuracy': accuracies,

'Precision': precisions,

'Recall': recalls,

'F1': f1_scores

})


results_df = results_df.sort_values(by='Accuracy', ascending=False)

print(results_df)
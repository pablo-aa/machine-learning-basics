# Algorithm to classify benign and malignant lesions

# Reading Dataset
import pandas as pd

df = pd.read_csv("features_benign_malignant.csv",sep=";")
# print(df.head())
print(df.shape)

# Pre processing

## Normalization with MinMax 
def normalize(method, df):
    if(method == "none"):
        return df
    if(method == "MinMaxScaler"):
        from sklearn.preprocessing import MinMaxScaler

        image_col = df.Images # Saving Image columns to don't normalize them
        diagnosis_col = df.diagnosis # Saving Image columns to don't normalize them

        scaler  = MinMaxScaler(feature_range=(0,1)) 
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df['Images'] = image_col
        df['diagnosis'] = diagnosis_col
        return df

df = normalize("none", df)
# print(df.head())


## Undersampling and Oversampling
import matplotlib.pyplot as plt

# data = {'Benign':0, 'Malignant':0}

# data['Benign'], data['Malignant'] = df.diagnosis.value_counts()

# fig = plt.figure(figsize=(8,6))
# plt.bar(list(data.keys()),list(data.values()), width=0.4, color="#99D5CA")
# plt.xlabel("Classes")
# plt.ylabel("No. of each Class")
# fig.suptitle("Why Undersampling and Oversampling?")
# plt.show()

### Aplying methods

class_0, class_1 = df.diagnosis.value_counts()

df_class_0 = df[df['diagnosis'] == 0]
df_class_1 = df[df['diagnosis'] == 1]

df_under_0 = df_class_0.sample(552, random_state = 2)
df_over_1 = df_class_1.sample(552, replace=True, random_state = 2)

df_prep = pd.concat([df_under_0, df_over_1])

df_prep = df_prep.reset_index()
df_prep = df_prep.drop(columns=['index'])
# print(df_prep)

# Visualizing diagnosis after Methods
# plt.figure(figsize=(8,6))
# df_prep.diagnosis.value_counts().plot(kind="bar", title = 'Count diagnosis', color="#99D5CA")
# plt.show()

# X and y axis
feature_cols = list(df.columns[:-1])
X = df_prep[feature_cols[1:]]
y = df_prep.diagnosis

# print(X)

# Filtering feature cols

def selectionMethod(method, X):
    if(method == "none"):
        return X
    if(method == "kbest"):
        # "Select K best" selection method
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2

        selector = SelectKBest(chi2, k=75)
        selector.fit(X,y)
        cols = list(selector.get_support(indices=True))
        new_features = pd.Series(feature_cols)[cols]

        X = pd.DataFrame(selector.transform(X), columns=list(new_features))
        return X

        # "PCA" selection Method

X = selectionMethod("none", X)


def generateModel(model, X, y):
    # Splitting Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=33, stratify=y)

    if(model == "SVM"):
        # SVM method
        from sklearn import svm
        import time
        start = time.time();
        clf_svm = svm.SVC(kernel='rbf', C=10,gamma=0.1, random_state=23, probability=True)
        clf_svm.fit(X_train,y_train)
        end = time.time();
        print(f"time: {end-start} s")
        y_pred_svm = clf_svm.predict(X_test)
        return (clf_svm, X_test,y_test, y_pred_svm)
    if(model == "BNB"):
        from sklearn.naive_bayes import BernoulliNB
        clf_bnb = BernoulliNB(alpha=0.1)
        clf_bnb.fit(X_train,y_train)
        y_pred_bnb = clf_bnb.predict(X_test)
        return (clf_bnb, X_test,y_test, y_pred_bnb)
    if(model == "DTC"):
        from sklearn.tree import DecisionTreeClassifier
        clf_dtc = DecisionTreeClassifier(criterion="entropy", max_depth=None, min_samples_leaf=3)
        clf_dtc.fit(X_train, y_train)
        y_pred_dtc = clf_dtc.predict(X_test)
        return (clf_dtc, X_test,y_test, y_pred_dtc)


clf_svm, X_test_svm, y_test, y_pred_svm = generateModel("SVM", X,y)
clf_bnb, X_test_bnb, y_test, y_pred_bnb = generateModel("BNB", X,y)
clf_dtc, X_test_dtc, y_test, y_pred_dtc = generateModel("DTC", X,y)


# Métricas dos modelo
from sklearn import metrics
def generate_metrics(title, y_pred_model, cnf_matrix_bool):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred_model)
    if(cnf_matrix_bool):
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        class_names = ["Benign", "Malignant"]
        cnf = ConfusionMatrixDisplay(cnf_matrix,display_labels=class_names)
        cnf.plot(cmap=plt.cm.Blues)
        plt.title(title+" Confusion Matrix")
        plt.show()
    
    tn, fp = cnf_matrix[0] 
    fn, tp = cnf_matrix[1]
    print(title, "Metrics:")

    print("Accuracy:",(tp+tn)/(tn+tp+fn+fp)) # (TP + TN)/(TOTAL)
    print("Precision:",tp/(tp+fp)) # TP/(TP+FP)
    print("Recall/Sensitivity:",tp/(tp+fn)) # TP/(TP+FN) 
    print("Specificity:",tn/(tn+fp), "\n"); # TN/(TN + FP)

generate_metrics("Support Vector Machine", y_pred_svm, False)
generate_metrics("Bernoulli Naive Bayes", y_pred_bnb, False)
generate_metrics("Decision Tree Classifier CART", y_pred_dtc, False)

def generate_ROC(title ,clf, X_test, y_test, final):
    y_pred_proba = clf.predict_proba(X_test)[::,1] 
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label=title+" "+str(auc))
    plt.legend(loc=4)
    if(final):
        plt.xlabel("False Positives")
        plt.ylabel("True Positives")
        plt.show()
    

generate_ROC("Support Vector Machine", clf_svm, X_test_svm, y_test, False);
generate_ROC("Bernoulli Naive Bayes",clf_bnb, X_test_bnb, y_test, False);
generate_ROC("Decision Tree Classifier CART",clf_dtc, X_test_dtc, y_test, True);



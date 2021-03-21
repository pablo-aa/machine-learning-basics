'''
Regressão logística para classificação de pessoas com diabetes

Diabetes - Positive (1)
No Diabetes - Negative (0)

Confusion Matrix |tn, fp|
                 |fn, tp|

'''
# Pegando dados da tabela
import pandas as pd

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

df = pd.read_csv("diabetes.csv", header=None, names=col_names, skiprows=1)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

X = df[feature_cols]
y = df.label

# Dividindo treino e teste
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=7, stratify=y)
'''
Parâmetros da função train_test_split:
    - test_size: Proporção do teste (e treinamento consequentemente).
    - random_state: Seed para pegar dados aleatórios. Por default esse valor
    sofre alterações a cada execução, para ter sempre o mesmo output deve-se 
    escolher um int arbitrário para o mesmo.
    - stratify: Estratifica treino e teste, garantindo mesma proporção das labels
    em ambos.
'''

# Aplicando Regressão logistica
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(max_iter=200)
'''
    max_iter: Máximo de iterações da função de otimização (default=100).
'''
lg.fit(X_train, y_train)

y_pred = lg.predict(X_test)

# Avaliando modelo pela confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Metrics
tn, fp = cnf_matrix[0] 
fn, tp = cnf_matrix[1]
print("Métricas do modelo:")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # (TP + TN)/(TOTAL)
print("Precision:",metrics.precision_score(y_test, y_pred)) # TP/(TP+FP)
print("Recall/Sensitivity:",metrics.recall_score(y_test, y_pred)) # TP/(TP+FN) 
print("Specificity:",tn/(tn+fp)); # TN/(TN + FP)

# Visualizando a confusion matrix

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
class_names = ["No Diabetes", "Diabetes"]
plot_confusion_matrix(lg,X_test, y_test, cmap=plt.cm.Purples, display_labels=class_names)
plt.show()
# Curva ROC
'''
predict_proba: Calcula probabilidades da i-ésima pessoa ser 0 ou 1, retorna matriz
com duas colunas(probabilidade de 0 e 1) e "quantidade de amostras" linhas.
obs: a notação [::,1] é equivalente a seleciona a coluna número 1 da matriz. Essa
notação é um recurso da biblioteca numpy.
'''
y_pred_proba = lg.predict_proba(X_test)[::,1] # probabilidade do i-ésimo paciente ser 1 (no diabetes)
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="diabetes classification, auc="+str(auc))
plt.legend(loc=4)

plt.show()




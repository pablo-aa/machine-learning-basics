'''
Regressão linear Reconhecer digitos 
Escritos a mão
'''
from sklearn.datasets import load_digits
digits = load_digits()

# Visualizando dados do Dataset

# 1797 images of 64 pixels(8x8)
print(digits.data.shape) 
print(digits.data)
print(digits.target)

import numpy as np
import matplotlib.pyplot as plt

# Visualizando o Dataset
plt.figure(figsize=(10,6))
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.05, wspace=0.2)
for index, (image, label) in enumerate(zip(digits.data[0:10],digits.target[0:10])):
    plt.subplot(2,5,index+1)
    plt.imshow(np.reshape(image,(8,8)), cmap=plt.cm.gray)
    plt.title('Training: %d\n' %label, fontsize=10)

plt.show()

# Dividindo treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# Aplicando refressão logística
from sklearn.linear_model import LogisticRegression

lgreg = LogisticRegression(solver='lbfgs')

lgreg.fit(X_train, y_train)

y_pred = lgreg.predict(X_test)

# Acurácia do modelo

from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix

# Implement confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
class_names = ["0","1","2","3","4","5","6","7","8","9"]
plot_confusion_matrix(lgreg,X_test, y_test, cmap=plt.cm.Purples, display_labels=class_names)
plt.show()



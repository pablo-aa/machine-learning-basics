# -*- coding: utf-8 -*-
"""decision_tree.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bcQdRnwfPagcJSZtTb6rhb39dud4c4tc
"""

''' 
Algoritmo de Decision Tree
 Situação problema: 
 Dado alguns fatores climáticos, decida se pode ou não acontecer um
 jogo de tênis.
 Jogar Tênis:
 2 classes: ("Não": 0, "Sim": 1)
 4 atributos:
   - Tempo ("Ensolarado": 0, "Nublado": 1, "Chuvoso": 2)
   - Temperatura ("Agradável": 0, "Moderada": 1, "Quente": 2)
   - Humidade ("Normal": 0, "Alta": 1)
   - Vento ("Fraco": 0, "Forte": 1)
'''

# Importando pandas para analisar os dados da tabela
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

col_names = ['Tempo', 'Temperatura', 'Humidade', 'Vento', 'Tenis']

table = pd.read_excel('tenis.xlsx', names=col_names)

# Retirando primeira linha da tabela
table = table.drop([0])
table.head()

# Tratando de forma numérica os parâmetros do problema
Tempo = {"Ensolarado": 0, "Nublado": 1, "Chuvoso": 2}
table['Tempo'] = table['Tempo'].map(Tempo)

Temperatura = {"Agradável": 0, "Moderada": 1, "Quente": 2}
table['Temperatura'] = table['Temperatura'].map(Temperatura)

Humidade = {"Normal": 0, "Alta": 1}
table['Humidade'] = table['Humidade'].map(Humidade)

Vento = {"Fraco": 0, "Forte": 1}
table['Vento'] = table['Vento'].map(Vento)

Tenis = {"Não": 0, "Sim": 1}
table['Tenis'] = table['Tenis'].map(Tenis)

table

# Definindo atributos e rótulos da Árvore de decisão
feature_cols = ['Tempo', 'Temperatura', 'Humidade', 'Vento']
X = table[feature_cols]
Y = table.Tenis

# Criando objeto da Árvore de decisão
clf = DecisionTreeClassifier(criterion="entropy")

# Contruindo a Árvore de decisão
clf = clf.fit(X,Y)

# Guardando resultados dos rótulos previstos pelo modelo
Y_pred = clf.predict(X)

# Primeira forma de visualização da Árvore
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
Image(graph.create_png())

# Segunda forma de visualização da Árvore
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

data = tree.export_graphviz(clf, out_file=None, feature_names=feature_cols)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()

# Avaliando "Accuracy" da Árvore
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(Y, Y_pred))
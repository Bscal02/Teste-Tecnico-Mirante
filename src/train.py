import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, classification_report, roc_auc_score, auc, roc_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Carregar banco de dados
full_data = pd.read_excel('telco_customer_churn 1.xlsx', index_col='customerID')
# fazendo um data split aleatório de 70/30
X = full_data.drop('Churn', axis=1)
y = full_data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7, random_state=45641564)

# Criando pipeline de preprocessamento de dados

## Selecionando features numéricas e categóricas
numerical_ix = X_train.select_dtypes(include=np.number).columns
categorical_ix = X_train.select_dtypes(exclude=np.number).columns

## Pipelines para cada tipo de dado
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder()),
    ('scaler', StandardScaler())])

## Juntando pipelines
preprocessor = ColumnTransformer([
        ('numerical', numerical_transformer, numerical_ix),
        ('categorical', categorical_transformer, categorical_ix)],
         remainder='passthrough')

# selecionando algoritmos para comparação

## Lista de algoritmos a serem comparados
classifiers = [
    KNeighborsClassifier(),
    LogisticRegression(random_state=123),
    DecisionTreeClassifier(random_state=123),
    RandomForestClassifier(random_state=123),
    GradientBoostingClassifier(random_state=123)
    ]

classifier_names = [
    'KNeighborsClassifier()',
    'LogisticRegression()',
    'DecisionTreeClassifier()',
    'RandomForestClassifier()',
    'GradientBoostingClassifier()'
]

model_scores = []

## loop para calcular o AUC de teste de cada um
for classifier, name in zip(classifiers, classifier_names):
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(k=len(X_train.columns))),
        ('classifier', classifier)])
    pipe.fit(X_train,y_train)
    y_probs = pipe.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_probs)
    model_scores.append(score)

# tabela com os valores de AUC por modelo
model_performance = pd.DataFrame({
    'Classifier': 
      classifier_names, 
    'Test AUC':
      model_scores
})

sorted_models = model_performance.sort_values('Test AUC', ascending = False, ignore_index=True)
## separando modelo com o melhor resultado
best_model = eval(sorted_models['Classifier'][0])

# pipeline para a seleção de hiper parametros

param_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(k=len(X.columns))),
        ('classifier', best_model)
])

# valores a serem testados
grid = {
    "selector__k": [13,14,15,16,17,18],
    "classifier__max_depth":[1,3,5],
    "classifier__learning_rate":[0.01,0.1,1],
    "classifier__n_estimators":[100,200,300,400]
}

# função pora testar cada combinação usando AUC de validação cruzada nos dados treinamento
gridsearch = GridSearchCV(estimator=param_pipe, param_grid=grid, n_jobs= 1, scoring='roc_auc')
gridsearch.fit(X_train,y_train) 

# pipeline final do modelo
end_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(k=13)), 
    ('classifier', best_model)])\
    .fit(X_train,y_train)

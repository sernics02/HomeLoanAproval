import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

def naive_bayes(df):
  # Eliminar la columna 'Loan_ID' y la columna objetivo 'LoanStatus_Y'
  X = df.drop(['Loan_ID', 'LoanStatus'], axis=1)
  y = df['LoanStatus']

  # Dividir el conjunto de datos en entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Inicializar y entrenar el clasificador Naive Bayes
  nb_classifier = MultinomialNB()
  nb_classifier.fit(X_train, y_train)

  # Realizar predicciones en el conjunto de prueba
  y_pred = nb_classifier.predict(X_test)

  # Evaluar el rendimiento del clasificador
  accuracy = accuracy_score(y_test, y_pred)
  print("\n\n\nNAIVE BAYES CLASSIFIER\n\n\n")
  print(f'Accuracy: {accuracy:.2f}')

  # Mostrar el informe de clasificación
  print('\nClassification Report:\n', classification_report(y_test, y_pred))

def main():
  df = pd.read_csv("homeLoan.csv")

  # Codificar variables categóricas
  label_encoder = LabelEncoder()
  df['Gender'] = label_encoder.fit_transform(df['Gender'])
  df['Married'] = label_encoder.fit_transform(df['Married'])
  df['Education'] = label_encoder.fit_transform(df['Education'])
  df['SelfEmployed'] = label_encoder.fit_transform(df['SelfEmployed'])
  df['PropertyArea'] = label_encoder.fit_transform(df['PropertyArea'])
  df['LoanStatus'] = label_encoder.fit_transform(df['LoanStatus'])

  # Manejo de valores faltantes
  imputer = SimpleImputer(strategy='mean')
  df['LoanAmount'] = imputer.fit_transform(df['LoanAmount'].values.reshape(-1, 1))
  df['LoanAmountTerm'] = imputer.fit_transform(df['LoanAmountTerm'].values.reshape(-1, 1))
  df['Dependents'] = imputer.fit_transform(df['Dependents'].values.reshape(-1, 1))

  # ver que datos son nulos
  print(df.isnull().sum())

  # Entrenar y evaluar el clasificador Naive Bayes
  naive_bayes(df)

if __name__ == "__main__":
  main()

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

VARIABLES = ('Gender','Married','Dependents','Education','SelfEmployed','ApplicantIncome','CoapplicantIncome','LoanAmount','LoanAmountTerm',   'PropertyArea')

def read_info(dataframe):
  print(dataframe.head())

    # Información sobre el data frame
  print(dataframe.info())

    # Estadísticas descriptivas de las variables numéricas
  print(dataframe.describe())

def distributions(dataframe):
  # Visualización de distribución de variables numéricas
  for variable in VARIABLES:
    sns.histplot(dataframe[variable], bins=20)
    plt.show()

def drop_outliers(dataframe, variable, max_value, min_value=None):
  if min_value is not None:
    new_dataframe = dataframe.loc[(dataframe[variable] <= max_value) & (dataframe[variable] >= min_value)]
  else:
    new_dataframe = dataframe.loc[dataframe[variable] <= max_value]
  return new_dataframe
  
def show_histplots(df):
  # Visualización de distribución de variables numéricas
  fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
  sns.histplot(df['ApplicantIncome'], bins=20, ax=axes[0])
  sns.histplot(df['CoapplicantIncome'], bins=20, ax=axes[1])
  sns.histplot(df['LoanAmount'], bins=20, ax=axes[2])
  plt.show()
def preprocess(dataframe):
  pass 

def one_hot_encoder(dataframe):
    columns_to_encode = ('Gender', 'Married', 'Education', 'SelfEmployed', 'PropertyArea', 'LoanStatus')

    df_encoded = dataframe.copy()

    for column in columns_to_encode:
        # Aplicar One-Hot Encoding
        one_hot_encoded = pd.get_dummies(dataframe[column], prefix=column, drop_first=True)
        
        # Convertir los valores a 0 o 1
        one_hot_encoded = one_hot_encoded.astype(int)
        
        # Agregar las nuevas columnas al DataFrame original
        df_encoded = pd.concat([df_encoded, one_hot_encoded], axis=1)
        
        # Eliminar la columna original si lo deseas
        df_encoded.drop([column], axis=1, inplace=True)

    return df_encoded

def k_means(X, n_clusters, max_iter=10):
    # Initialize missing values to their column means
    missing = np.isnan(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    for i in range(max_iter):
        if i > 0:
            # initialize KMeans with the previous set of centroids. this is much
            # faster and makes it easier to check convergence (since labels
            # won't be permuted on every iteration), but might be more prone to
            # getting stuck in local minima.
            cls = KMeans(n_clusters, init=prev_centroids)
        else:
            # do multiple random initializations in parallel
            cls = KMeans(n_clusters, n_init=8)

        # perform clustering on the filled-in data
        labels = cls.fit_predict(X_hat)
        centroids = cls.cluster_centers_

        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels
        prev_centroids = cls.cluster_centers_

    return labels, centroids, X_hat


def naive_bayes(df):
  # Eliminar la columna 'Loan_ID' y la columna objetivo 'LoanStatus_Y'
  X = df.drop(['Loan_ID', 'LoanStatus_Y'], axis=1)
  y = df['LoanStatus_Y']

  # Dividir el conjunto de datos en entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Inicializar y entrenar el clasificador Naive Bayes
  nb_classifier = MultinomialNB()
  nb_classifier.fit(X_train, y_train)

  # Realizar predicciones en el conjunto de prueba
  y_pred = nb_classifier.predict(X_test)

  # Evaluar el rendimiento del clasificador
  accuracy = accuracy_score(y_test, y_pred)
  print("NAIVE BAYES CLASSIFIER\n")
  print(f'Accuracy: {accuracy:.2f}')

  # Mostrar el informe de clasificación
  # print('\nClassification Report:\n', classification_report(y_test, y_pred))

def buildClassificatonTree(csv_path, target_column):
  # Cargar datos desde el archivo CSV
  data = pd.read_csv(csv_path)
  
  # Separar características (X) y variable objetivo (y)
  X = data.drop(columns=[target_column])
  y = data[target_column]
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Construir el modelo de árbol de clasificación
  tree_classifier = DecisionTreeClassifier()
  
  # Entrenar el modelo
  tree_classifier.fit(X_train, y_train)
  
  # Predecir sobre el conjunto de prueba
  y_pred = tree_classifier.predict(X_test)
  
  # Calcular la precisión del modelo
  accuracy = accuracy_score(y_test, y_pred)
  print("\nARBOL DE CLASIFICACION\n")
  print("Accuracy: ", accuracy)
  
  return tree_classifier

def buildKNNeighbors(csv_path, target_column, k = 5):
  # Cargar datos desde el archivo CSV
    data = pd.read_csv(csv_path)
    
    # Separar características (X) y variable objetivo (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Construir el modelo de clasificador kNN
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    
    # Entrenar el modelo
    knn_classifier.fit(X_train, y_train)
    
    # Predecir sobre el conjunto de prueba
    y_pred = knn_classifier.predict(X_test)
    
    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("\nCLASIFICADOR KNN", "\n\nAccuracy: ", accuracy)
    
    return knn_classifier

## main
def main():

  archivo_csv = 'homeLoanAproval.csv'

  dataframe = pd.read_csv(archivo_csv)
  dataframe = one_hot_encoder(dataframe) # Convertir todos los valores categóricos en valores enteros
#
  dataframe = drop_outliers(dataframe, 'ApplicantIncome', max_value = 3000) # Eliminar los valores atípicos de la columna 'ApplicantIncome'
  dataframe = drop_outliers(dataframe, 'LoanAmount', max_value = 350) # Eliminar los valores atípicos de la columna 'LoanAmount'
  dataframe = drop_outliers(dataframe, 'CoapplicantIncome', max_value = 3000, min_value = 10) # Eliminar los valores atípicos de la columna 'CoapplicantIncome'
  
  X = dataframe.values
  dataframe.to_csv('homeLoanAproval1.csv', index=False)
  # print(X)
  labels, centroids, X_hat = k_means(X, n_clusters=4, max_iter=10) # Imputar los valores faltantes utilizando K-Means
  # read_info(dataframe)
  dataframe.to_csv('homeLoanAproval1.csv', index=False) 
  dataframe = pd.DataFrame(X_hat, columns=dataframe.columns)
  dataframe.to_csv('homeLoanAproval1.csv', index=False)
  naive_bayes(dataframe)
  buildClassificatonTree('homeLoanAproval1.csv', 'LoanStatus_Y')
  buildKNNeighbors('homeLoanAproval1.csv', 'LoanStatus_Y')

if __name__ == "__main__":
  main()




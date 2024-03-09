import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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


def ignore_null_values(dataframe):
  df_sin_nulos = dataframe.dropna(axis=1)
  return df_sin_nulos

def remove_null_values(dataframe):
  df_sin_nulos = dataframe.dropna(axis=0)
  return df_sin_nulos


def infiere_null_values_mean(dataframe, variable):
  mean = dataframe[variable].mean()
  dataframe[variable] = dataframe[variable].fillna(int(mean))
  return dataframe


def infiere_null_values_median(dataframe, variable):
  median = dataframe[variable].median()
  dataframe[variable] = dataframe[variable].fillna(int(median))
  return dataframe


def infiere_null_values_mode(dataframe, variable):
  dataframe[variable] = dataframe[variable].fillna(dataframe[variable].mode()[0])
  return dataframe


def convert_to_int(dataframe, variable):
  dataframe[variable] = dataframe[variable].astype(int)
  return dataframe


def drop_outilers(dataframe, variable, max): 
 # Por ejemplo, el max es 1000

  # Seleccionar las filas que cumplen con la condición
  new_dataframe = dataframe.loc[dataframe[variable] <= max]
  return new_dataframe
  

def kMeans(df, n_clusters=2):
  # Select the features for clustering
  features = ['ApplicantIncome', 'LoanAmount', "Married_Yes", "Education_Not_Graduate", "SelfEmployed_Yes", "PropertyArea_Semiurban", "PropertyArea_Urban", "CoapplicantIncome"]
  
  # Perform K-means clustering
  kmeans = KMeans(n_clusters)
  kmeans.fit(df[features])
  
  # Add the cluster labels to the DataFrame
  df['Cluster'] = kmeans.labels_
  
  # Print the cluster centers
  print(kmeans.cluster_centers_)
  
  # Plot the clusters
  sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Cluster')
  plt.show()
  return df 
  

from sklearn.cluster import KMeans
import pandas as pd

def kMeans_impute_missing(df, n_clusters=4, method='mean'):
    # Select the features for clustering
    features = ['ApplicantIncome', 'LoanAmount', "Married_Yes", "Education_Not_Graduate", "SelfEmployed_Yes", "PropertyArea_Semiurban", "PropertyArea_Urban", "CoapplicantIncome"]
  
    # Copy the DataFrame to avoid modifying the original one
    df_copy = df.copy()
    
    # Remove rows with missing values in selected features
    df_filtered = df_copy.dropna(subset=features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters)
    kmeans.fit(df_filtered[features])
  
    # Add the cluster labels to the DataFrame
    df_filtered['Cluster'] = kmeans.labels_
    
    # Find the cluster label for rows with missing values
    missing_values_indices = df_copy.index[df_copy[features].isnull().any(axis=1)]
    df_missing_values = df_copy.loc[missing_values_indices]
    cluster_labels_for_missing_values = kmeans.predict(df_missing_values[features])
    
    # Impute missing values based on the mean or mode of the cluster
    for cluster_label in range(n_clusters):
        cluster_data = df_filtered[df_filtered['Cluster'] == cluster_label]
        if method == 'mean':
            cluster_means = cluster_data.mean()
            impute_values = cluster_means[features]
        elif method == 'mode':
            cluster_modes = cluster_data.mode().iloc[0]
            impute_values = cluster_modes[features]
        df_copy.loc[(df_copy.index.isin(missing_values_indices)) & (cluster_labels_for_missing_values == cluster_label), features] = impute_values

    return df_copy


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
        
        # Agregar las nuevas columnas al DataFrame original
        df_encoded = pd.concat([df_encoded, one_hot_encoded], axis=1)
        
        # Eliminar la columna original si lo deseas
        df_encoded.drop([column], axis=1, inplace=True)

    return df_encoded

def impute_by_cluster(dataframe, method='mode'):
    cluster = 'Cluster'
    # Agrupar los datos por las etiquetas de cluster
    grouped_data = dataframe.groupby(cluster)

    # Calcular la media o la moda para cada cluster
    if method == 'mean':
        cluster_stats = grouped_data.mean()
    elif method == 'mode':
        cluster_stats = grouped_data.apply(lambda x: mode(x)[0][0])

    # Iterar sobre las características que tienen valores faltantes
    for column in dataframe.columns:
        missing_values = dataframe[column].isnull()
        if missing_values.any():
            # Imputar la media o la moda del cluster al que pertenece cada valor faltante
            for cluster_label, stats in cluster_stats.iterrows():
                fill_value = stats[column]
                dataframe.loc[(dataframe[cluster] == cluster_label) & missing_values, column] = fill_value

    return dataframe


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
  print("\n\n\nNAIVE BAYES CLASSIFIER\n\n\n")
  print(f'Accuracy: {accuracy:.2f}')

  # Mostrar el informe de clasificación
  print('\nClassification Report:\n', classification_report(y_test, y_pred))

## main
def main():

  archivo_csv = 'homeLoan.csv'

  dataframe = pd.read_csv(archivo_csv)
#   dataframe2 = dataframe.copy()
#  # show_histplots(dataframe)

#   ### meter en funcion 
#   by_mode = ['Gender', 'Married', 'SelfEmployed', 'Education', 'LoanAmountTerm']
#   by_mean = ['Dependents', 'LoanAmount']
#   for column in by_mode:
#     dataframe = infiere_null_values_mode(dataframe, column)

#   for column in by_mean:
#     dataframe = infiere_null_values_median(dataframe, column)

#   dataframe.to_csv('homeLoanAproval2.csv', index=False)

#   dataframe2 = pd.read_csv('homeLoanAproval2.csv')
#   ### meter en funcion 
  
#   # Elimina los outliers
#   dataframe2 = drop_outilers(dataframe2, 'ApplicantIncome', 30000)
#   dataframe2 = drop_outilers(dataframe2, 'LoanAmount', 500)
 # distributions(dataframe2)
#  read_info(dataframe2)
  
  #dataframe2 = preprocess(dataframe2)
 # dataframe2 = one_hot_encode_dataframe(dataframe2)
  # dataframe2 = dataframe.copy()
  # dataframe2 = impute_missing_values(dataframe2)
  dataframe = one_hot_encoder(dataframe)
 # dataframe = ignore_null_values(dataframe)
  dataframe = drop_outilers(dataframe, 'ApplicantIncome', 30000)
  dataframe = drop_outilers(dataframe, 'LoanAmount', 350)
  dataframe = drop_outilers(dataframe, 'CoapplicantIncome', '1_000_000')
  # dataframe = kMeans(dataframe, 4)
  # dataframe = impute_by_cluster(dataframe, 'mode')

  dataframe = kMeans_impute_missing(dataframe)

  read_info(dataframe)
  dataframe.to_csv('homeLoanAproval3.csv', index=False)
  # dataframe2 = one_hot_encoder(dataframe2)
  # dataframe4 = remove_null_values(dataframe)
  # read_info(dataframe2)
  #dataframe2.to_csv('homeLoanAproval3.csv', index=False)
  #naive_bayes(dataframe2)
 # naive_bayes(dataframe2)
  #kMeans(dataframe2, 5)

if __name__ == "__main__":
  main()




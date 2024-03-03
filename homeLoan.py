import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

VARIABLES = ['Loan_ID','Gender','Married','Dependents','Education','SelfEmployed', 'ApplicantIncome','CoapplicantIncome','LoanAmount','LoanAmountTerm','PropertyArea','LoanStatus']

def printAllRows(df):
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', 12)
  print(df)

def csvData(df):
  df.info()
  print(df.describe())

def removeNullValues(df, variable):
  df = df.dropna(subset=[variable])
  df.to_csv('homeLoanAproval2.csv', index=False)
  df = pd.read_csv('homeLoanAproval2.csv')
  return df

# Funcion para imprimir el histograma
def printHistPlot(df):
  for variable in VARIABLES:
    sns.histplot(df[variable])
    plt.show()

# Funcion para imprimir el blotpot
def printBoxPlot(df):
  for variable in VARIABLES:
    sns.boxplot(df[variable])
    plt.show()

# Inferir valores nulos a traves de la media
def infiere_null_values_mean(dataframe, variable):
  media = dataframe[variable].mean()
  # media a entero
  dataframe[variable] = dataframe[variable].fillna(media).mean()
  return dataframe

# Inferir valores nulos a traves de la mediana
def infiere_null_values_median(dataframe, variable):
  dataframe[variable] = dataframe[variable].fillna(dataframe[variable].median())
  return dataframe

# Inferir valores nulos a traves de la moda
def infiere_null_values_mode(dataframe, variable):
  dataframe[variable] = dataframe[variable].fillna(dataframe[variable].mode()[0])
  return dataframe

def deleteValues(df, variable, minValue):
  dfFiltered = df.loc[df[variable] <= minValue]
  # Guardar el archivo
  dfFiltered.to_csv('homeLoanAproval3.csv', index=False)
  dfFiltered = pd.read_csv('homeLoanAproval3.csv')
  return dfFiltered

def kMeans(df, clusters = 2):
  # Select the features for clustering
  features = ['ApplicantIncome', 'LoanAmount']
  
  # Perform K-means clustering
  kmeans = KMeans(n_clusters=clusters)
  kmeans.fit(df[features])
  
  # Add the cluster labels to the DataFrame
  df['Cluster'] = kmeans.labels_
  
  # Print the cluster centers
  # print(kmeans.cluster_centers_)
  
  # Plot the clusters
  sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Cluster')
  plt.show()

# main
def main():
  # Specify the file path
  file_path = './homeLoan.csv'

  # Load the CSV file into a pandas DataFrame
  df = pd.read_csv(file_path)

  inferMode = ['Gender', 'Married', 'SelfEmployed', 'Education', 'LoanAmountTerm']
  inferMedian = ['Dependents', 'LoanAmount']
  for variable in inferMode:
    df = infiere_null_values_mode(df, variable)

  for variable in inferMedian:
    df = infiere_null_values_median(df, variable)

  df.to_csv('homeLoanAproval2.csv', index=False)
  df2 = pd.read_csv('homeLoanAproval2.csv')

  df2 = deleteValues(df2, 'ApplicantIncome', 30000)
  df2 = deleteValues(df2, 'LoanAmount', 500)  
  
  kMeans(df2, 5);

if __name__ == "__main__":
  main()
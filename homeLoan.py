import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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

def executeKMeans(df, clusters = 2):
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

  # Guardar cada cluster en un csv
  for i in range(clusters):
    dfCluster = df.loc[df['Cluster'] == i]
    dfCluster.to_csv(f'cluster{i}.csv', index=False)

  plt.show()

def preprocessing(df):
  pass

def oneHotEncodingDF(df):
  columns = ('Gender', 'Married', 'Education', 'SelfEmployed', 'PropertyArea', 'LoanStatus')

  dfEncoded = df.copy()

  for column in columns:
    oneHotEncoded = pd.get_dummies(dfEncoded[column], prefix=column, drop_first=True)

    dfEncoded = pd.concat([dfEncoded, oneHotEncoded], axis=1)

    dfEncoded.drop([column], axis=1, inplace=True)
  
  return dfEncoded

def inferValuesByGroup(df):
  values = ['Gender', 'Married', 'SelfEmployed', 'Education', 'LoanAmountTerm', 'Dependents', 'LoanAmount']

  # Inferir todos los valores anteriores dividiendolos por grupos y usando la moda
  for variable in values:
    df[variable] = df.groupby('PropertyArea')[variable].apply(lambda x: x.fillna(x.mode()[0]))
  
  return df

def naiveBayes(df):
  if 'Loan_ID' in df.columns:
    dataframe = df.drop('Loan_ID', axis=1, inplace=True)

  x = df.drop('LoanStatus_Y', axis=1)
  y = df['LoanStatus_Y']
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
  model = GaussianNB()
  model.fit(x_train, y_train)

  y_pred = model.predict(x_test)
  print(accuracy_score(y_test, y_pred))

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

  # df2 = preprocessing(df2)

  df2 = oneHotEncodingDF(df2)
  # Guardar en un csv
  df2.to_csv('homeLoanAproval4.csv', index=False)

  executeKMeans(df2, 4)
  naiveBayes(df2)

if __name__ == "__main__":
  main()

# Esse arquivo irá extrair os dados do arquivo
# 'unoeste_historico_de_chuva.csv' e colocalos
# no MySQL no formato adequado para uma análise ARIMA

import mysql.connector
from mysql.connector import MySQLConnection

hostName = "localhost";
userName = "admin";
thePassword = "admin";
databaseName = "analytics";

print("\nIniciando conexão com o banco de dados:");
print(" Host: " + hostName);
print(" User: " + userName);
print(" Database: " + databaseName);

try:
  connection : MySQLConnection = mysql.connector.connect(
    host=hostName,
    user=userName,
    passwd=thePassword,
    database=databaseName
  );
except:
  print("Erro ao acessar o banco de dados! Confira seus dados!")
  exit()
print("Conexão estabelecida com sucesso!!\n\n");

sql_drop_command = "DROP TABLE IF EXISTS rain_history";
sql_create_command = "CREATE TABLE rain_history ( data DATE PRIMARY KEY, precipitacao FLOAT );"

print("Executando SQL Commands :");
print(sql_drop_command);

connection.cursor().execute(sql_drop_command);
connection.commit()

print(sql_create_command);
connection.cursor().execute(sql_create_command);
connection.commit()

#LOCAL indica que o arquivo está no cliente!
import os
theFile = os.getcwd() + os.path.sep + "unoeste_historico_de_chuva.csv";

print("\n\nIniciando leitura do arquivo: \n   " + theFile)

from myownapi.AnalyticsARIMA import AnalyticsARIMA;
analytics  = AnalyticsARIMA();
analytics.read_csv(theFile);

print("\nTratando variáveis!")
for columName in list(analytics.getColumnsNames()):
    analytics.tratarVariaveisNulasComMediaDasOutras(columName);

print("\nTransformando dataset para formato correto!")

#Tratando os dados (Vide Treino7)
import pandas as pd
def toDate(year, month):
  dateString = "{}-1-{}".format(month, int(year));
  return pd.to_datetime(dateString);
df = pd.DataFrame({"Data": [], "Precipitacao": []})
for index, row in analytics.df.iterrows():
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 1)], "Precipitacao": [row['Janeiro']]}), ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 2)], "Precipitacao": [row['Fevereiro']]}),
                 ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 3)], "Precipitacao": [row['Março']]}), ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 4)], "Precipitacao": [row['Abril']]}), ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 5)], "Precipitacao": [row['Maio']]}), ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 6)], "Precipitacao": [row['Junho']]}), ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 7)], "Precipitacao": [row['Julho']]}), ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 8)], "Precipitacao": [row['Agosto']]}), ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 9)], "Precipitacao": [row['Setembro']]}), ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 10)], "Precipitacao": [row['Outubro']]}), ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 11)], "Precipitacao": [row['Novembro']]}),
                 ignore_index=True);
  df = df.append(pd.DataFrame({"Data": [toDate(row['Ano'], 12)], "Precipitacao": [row['Dezembro']]}),
                 ignore_index=True);
analytics.setDataframe(df);

print("\nComitando mudanças!")


for row in df.itertuples():
  sql_date = row[1];
  sql_value = row[2];
  sql_insert_command = "INSERT INTO rain_history VALUES (\"{0}\", \"{1}\")".format(sql_date, sql_value);
  print(sql_insert_command);
  connection.cursor().execute(sql_insert_command);

connection.commit();

print("\nDataset exportado para " + hostName + " -> " + databaseName + ".db");

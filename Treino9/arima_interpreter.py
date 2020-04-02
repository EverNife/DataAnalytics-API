
# Esse arquivo irá extrair os dados
# de uma aplicação MYSQL e ira processar
# as análises e retornar o resultado.

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


sql_select_command = "SELECT * FROM rain_history;";

print("Executando SQL Commands :");
print(sql_select_command);

cursor = connection.cursor();
cursor.execute(sql_select_command);
result = cursor.fetchall();

from myownapi.AnalyticsARIMA import AnalyticsARIMA;
analytics = AnalyticsARIMA();

import pandas as pd

dateArray = [];
valueArray = [];

for entry in result:
  dateArray.append(entry[0]);
  valueArray.append(entry[1]);

dataFrameData = { 'Date':  dateArray,
        'Value': valueArray}


original_df = pd.DataFrame(data=dataFrameData, columns=['Date','Value']);
analytics.setDataframe(original_df);

for row in analytics.df.itertuples():
  date = pd.to_datetime(row[1]);
  value = row[2];

analytics.arimaDefinirColunaObjetivo("Value","Date");

analytics.aplicarARIMA(verbose=True)

json_forecast = analytics.ARIMAForecastToJson(setps=24);

import json

json_forecast_decoded : dict = json.loads(json_forecast);
new_json = list();
for aTuple in json_forecast_decoded.items():
  new_entry = {
    "Date" : aTuple[0],
    "Value" : aTuple[1]
  }
  new_json.append(new_entry);


json_origin = analytics.df.to_json();#http://pandas-docs.github.io/pandas-docs-travis/reference/api/pandas.DataFrame.to_json.html?highlight=to_json#pandas.DataFrame.to_json
#Usar orient='record' facilita o output para um json sem index

dump_out          = json.dumps(json.loads(original_df.to_json(orient='records'))); #Ensure formatting by double doing the same things!
dump_out_predict  = json.dumps(new_json);

print(dump_out);
print(dump_out_predict);




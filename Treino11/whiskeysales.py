#Dataset rela de uma empresa "The IWSR"
from myownapi.AnalyticsARIMA import AnalyticsARIMA;

analytics = AnalyticsARIMA();
dataset_file = "C:/Users/Petrus/Desktop/UNESP/Docs 2016-2020/2019/Segundo Semestre/TCC2/TCC BigData Analytics/Treino11/whiskeysales.csv";

analytics.read_csv(dataset_file);
analytics.tratarVariaveisNulasComMediaDasOutras('Cases');

import datetime
def yearToDate(year):
    return datetime.datetime(year,1,1);
analytics.arimaDefinirColunaObjetivo(nomeDaColunaObjetivo='Cases',nomeDaColunaDeDatas='Year', funcaoDeConversaDeDatas=yearToDate)
analytics.aplicarARIMA(verbose=True,ARIMA_SASONALIDADE=1);

#Export part
import json
json_original_all = json.loads(analytics.df.to_json())['Cases'];

pred = analytics.ARIMAPredictionToPred(forecastStartingDate='1-1-2010')
json_pred_2010 = pred.predicted_mean.to_json();
json_pred_confidence_2010 = pred.conf_int().to_json();

pred = analytics.ARIMAForecastToPred(steps=10);
json_forecast_10Years = pred.predicted_mean.to_json();
json_forecast_confidence_10Years = pred.conf_int().to_json();


#Repetindo mesma coisa
analytics.read_csv(dataset_file);
analytics.tratarVariaveisNulasComValorEspecifico('Cases',0);

import datetime
def yearToDate(year):
    return datetime.datetime(year,1,1);
analytics.arimaDefinirColunaObjetivo(nomeDaColunaObjetivo='Cases',nomeDaColunaDeDatas='Year', funcaoDeConversaDeDatas=yearToDate)
analytics.aplicarARIMA(verbose=True,ARIMA_SASONALIDADE=1);


pred = analytics.ARIMAPredictionToPred(forecastStartingDate='1-1-2010')
json_pred_2010_zero = pred.predicted_mean.to_json();
json_pred_confidence_2010_zero = pred.conf_int().to_json();

pred = analytics.ARIMAForecastToPred(steps=10);
json_forecast_10Years_zero = pred.predicted_mean.to_json();
json_forecast_confidence_10Years_zero = pred.conf_int().to_json();


the_output = {
    "original":json_original_all,
    "config":{
        "maxTicks": 20000000,
        "stepSize": 1000000
    },
    "data":[
        {
            "name":"json_pred_2010",
            "type":"normal",
            "value":json.loads(json_pred_2010)
        },
        {
            "name": "json_pred_confidence_2010",
            "type": "confidence",
            "value": json.loads(json_pred_confidence_2010)
        },
        {
            "name": "json_forecast_10Years",
            "type": "normal",
            "value": json.loads(json_forecast_10Years)
        },
        {
            "name": "json_forecast_confidence_10Years",
            "type": "confidence",
            "value": json.loads(json_forecast_confidence_10Years)
        },
        {
            "name": "json_pred_2010_zero",
            "type": "normal",
            "value": json.loads(json_pred_2010_zero)
        },
        {
            "name": "json_forecast_10Years_zero",
            "type": "normal",
            "value": json.loads(json_forecast_10Years_zero)
        }
    ]
}

import time
milliseconds = int(round(time.time() * 1000))

fileName = "arima_export_" + str(milliseconds);

print("Dumping data in " + fileName + "_data.json");
f = open(fileName + "_data.json", "a")
f.write(json.dumps(the_output));
f.close()

#Venda de shampoo durante 3 anos
#Fonte: LIVRO Time Series Data Library (citing: Makridakis, Wheelwright and Hyndman (1998))

#Descrição: This dataset describes the monthly number of sales of shampoo over a 3-year period.
from myownapi.AnalyticsARIMA import AnalyticsARIMA;

analytics = AnalyticsARIMA();

dataset_file = "C:/Users/Petrus/Desktop/UNESP/Docs 2016-2020/2019/Segundo Semestre/TCC2/TCC BigData Analytics/Treino11/shampoo.csv";

analytics.read_csv(dataset_file);
analytics.arimaDefinirColunaObjetivo(nomeDaColunaObjetivo='Sales',nomeDaColunaDeDatas='Date')
analytics.aplicarARIMA();

import json
json_original_all = json.loads(analytics.df.to_json())['Sales'];

pred = analytics.ARIMAPredictionToPred(forecastStartingDate="2013")
json_pred_2013 = pred.predicted_mean.to_json();
json_pred_confidence_2013 = pred.conf_int().to_json();

pred = analytics.ARIMAForecastToPred(steps=12);
json_forecast_2014 = pred.predicted_mean.to_json();
json_forecast_confidence_2014 = pred.conf_int().to_json();

the_output = {
    "original":json_original_all,
    "data":[
        {
            "name":"json_pred_2013",
            "type":"normal",
            "value":json.loads(json_pred_2013)
        },
        {
            "name": "json_pred_confidence_2013",
            "type": "confidence",
            "value": json.loads(json_pred_confidence_2013)
        },
        {
            "name": "json_forecast_2014",
            "type": "normal",
            "value": json.loads(json_forecast_2014)
        },
        {
            "name": "json_forecast_confidence_2014",
            "type": "confidence",
            "value": json.loads(json_forecast_confidence_2014)
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

#Venda de shampoo durante 3 anos
#Fonte: LIVRO Time Series Data Library (citing: Makridakis, Wheelwright and Hyndman (1998))

#Descrição: This dataset describes the monthly number of sales of shampoo over a 3-year period.
from statsmodels.tsa.statespace.mlemodel import PredictionResultsWrapper

from myownapi.AnalyticsARIMA import AnalyticsARIMA;

analytics = AnalyticsARIMA();
analytics.getVersion()

dataset_file = "C:/Users/Petrus/Desktop/UNESP/Docs 2016-2020/2019/Segundo Semestre/TCC2/TCC BigData Analytics/Treino11/shampoo.csv";

analytics.read_csv(dataset_file);
analytics.arimaDefinirColunaObjetivo(nomeDaColunaObjetivo='Sales',nomeDaColunaDeDatas='Date')
analytics.head(3)

analytics.aplicarARIMA();


import json

json_original_all = json.dumps(json.loads(analytics.df.to_json())['Sales']);


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
            "value":json.loads(json_pred_2013)
        },
        {
            "name": "json_pred_confidence_2013",
            "value": json.loads(json_pred_confidence_2013)
        },
        {
            "name": "json_forecast_2014",
            "value": json.loads(json_forecast_2014)
        },
        {
            "name": "json_forecast_confidence_2014",
            "value": json.loads(json_forecast_confidence_2014)
        }
    ]
}

#Dataset rela de uma empresa "The IWSR"
from myownapi.AnalyticsARIMA import AnalyticsARIMA;

analytics = AnalyticsARIMA();
dataset_file = "C:/Users/Petrus/Desktop/UNESP/Docs 2016-2020/2019/Segundo Semestre/TCC2/TCC BigData Analytics/Treino11/cases-brazil-cities-time.csv";

analytics.read_csv(dataset_file);

df = analytics.df.filter(['date', 'city', 'deaths']);
ndf = df.loc[df['city'] == "São Paulo/SP"];
analytics.setDataframe(ndf);

analytics.arimaDefinirColunaObjetivo(nomeDaColunaObjetivo='deaths',nomeDaColunaDeDatas='date')
analytics.aplicarARIMA(verbose=True,ARIMA_SASONALIDADE=1);




#Export part
import json
json_original_all = json.loads(analytics.df.to_json())['deaths'];

pred = analytics.ARIMAPredictionToPred(forecastStartingDate='2020-04-20')
json_pred_2020_04_20 = pred.predicted_mean.to_json();
json_pred_confidence_2020_04_20 = pred.conf_int().to_json();

pred = analytics.ARIMAForecastToPred(steps=20);
json_forecast_20Dias = pred.predicted_mean.to_json();
json_forecast_confidence_20Dias = pred.conf_int().to_json();


the_output = {
    "original":json_original_all,
    "config":{
        "maxTicks": 10000,
        "stepSize": 25,
        "period": 1
    },
    "data":[
        {
            "name":"json_pred_2020_04_20",
            "type":"normal",
            "value":json.loads(json_pred_2020_04_20)
        },
        {
            "name": "json_pred_confidence_2020_04_20",
            "type": "confidence",
            "value": json.loads(json_pred_confidence_2020_04_20)
        },
        {
            "name": "json_forecast_20Dias",
            "type": "normal",
            "value": json.loads(json_forecast_20Dias)
        },
        {
            "name": "json_forecast_confidence_20Dias",
            "type": "confidence",
            "value": json.loads(json_forecast_confidence_20Dias)
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

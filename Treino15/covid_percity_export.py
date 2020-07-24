#Dataset rela de uma empresa "The IWSR"
from myownapi.AnalyticsARIMA import AnalyticsARIMA;

analytics = AnalyticsARIMA();
analytics.read_csv('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv');
analytics.df.head(10)
analytics.info

df = analytics.df.filter(['city','state', 'ibgeID', 'deaths', 'totalCases', 'date']);
df = df.loc[df['state'] == "SP"];

ibg_ids = df.filter(['ibgeID']);
ibg_ids = ibg_ids.drop_duplicates()

max = len(ibg_ids['ibgeID']);
contador = 0;

import os;
for ibge_id in ibg_ids['ibgeID']:
    contador = contador + 1;
    try:
        city_df = df.loc[df['ibgeID'] == ibge_id];
        city_name = city_df['city'].iloc[0];

        fileName = "cities/" + str(ibge_id) + ".json";

        if os.path.isfile(fileName):
            print("[" + str(contador) + "/" + str(max) + "] Being Skipped");
            continue


        print("[" + str(contador) + "/" + str(max) + "] Aplicando ARIMA na cidade - " + city_name);

        analytics = AnalyticsARIMA();
        analytics.setDataframe(city_df.copy())
        analytics.arimaDefinirColunaObjetivo(nomeDaColunaObjetivo='deaths', nomeDaColunaDeDatas='date');
        analytics.aplicarARIMA();

        #Export part
        import json
        json_original_all = json.loads(analytics.df.to_json())['deaths'];

        middle_row = int(analytics.df.size / 2)
        middle_date = analytics.df.iloc[middle_row].name;

        pred = analytics.ARIMAPredictionToPred(forecastStartingDate=middle_date)
        json_pred_middle = pred.predicted_mean.to_json();
        json_pred_middle_confidence = pred.conf_int().to_json();

        pred = analytics.ARIMAForecastToPred(steps=30);
        json_forecast = pred.predicted_mean.to_json();
        json_forecast_confidence = pred.conf_int().to_json();

        the_output = {
            "original":json_original_all,
            "config":{
                "city_id": ibge_id,
                "city_name": city_name,
                'aic': analytics.modelFit.aic,
                'bic': analytics.modelFit.bic
            },
            "data":[
                {
                    "name":"Prediction " + city_name,
                    "type":"normal",
                    "value":json.loads(json_pred_middle)
                },
                {
                    "name":"Pred-Confidence " + city_name,
                    "type": "confidence",
                    "value": json.loads(json_pred_middle_confidence)
                },
                {
                    "name":"Forecast " + city_name,
                    "type": "normal",
                    "value": json.loads(json_forecast)
                },
                {
                    "name":"Forecast-Confidence " + city_name,
                    "type": "confidence",
                    "value": json.loads(json_forecast_confidence)
                }
            ]
        }

        import time
        milliseconds = int(round(time.time() * 1000))

        print("Dumping data in " + fileName);
        f = open(fileName, "w+")
        f.write(json.dumps(the_output));
        f.close()
    except:
        print("An exception occurred at [" + str(contador) + "] - " + str(ibge_id))

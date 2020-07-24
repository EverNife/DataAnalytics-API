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

        print("[" + str(contador) + "/" + str(max) + "] Aplicando ARIMA na cidade - " + city_name);

        analytics = AnalyticsARIMA();
        analytics.setDataframe(city_df.copy())
        break;
    except:
        print("An exception occurred at [" + str(contador) + "] - " + str(ibge_id))

from fbprophet import Prophet

m = Prophet()
m.fit(analytics.df)
future = m.make_future_dataframe(periods=30)
future.tail(30);
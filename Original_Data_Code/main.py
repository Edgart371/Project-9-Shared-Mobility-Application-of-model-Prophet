from utils import *
from dbinteractions import write_to_db, write_day_agg_to_db
import logging
import numpy as np
import pandas as pd
from fbprophet import Prophet
from datetime import datetime, time, timedelta
from multiprocessing import cpu_count, Pool
import pycountry



def fit_model(CITY, TYPE, PROVIDER, COUNTRY, TZ, SOURCE, SHARING_TYPE):
    conn_pg = init_rds_connection('PROD', 'psycopg2')

    query_data  = open("sql/data.sql").read().format(CITY, TYPE, PROVIDER, TZ, MODEL_DAYS, SOURCE)
    data = doQuery(conn_pg, query_data)

    # get already modeled data days not to use for fit
    query_modeled_days  = open("sql/excluded_model_days.sql").read().format(CITY, TYPE, PROVIDER, SHARING_TYPE)

    modeled_days = doQuery(conn_pg, query_modeled_days)
    conn_pg.close()

    df = pd.DataFrame(data)
    df.columns = ['ds','y','rain','snow','main','time_lost','before_lag','nb_vehic_avail']

    # FORWARD FILL IF MISSING WEATHER VALUES
    for col in ['main','rain','snow','nb_vehic_avail']:
        df[col] = df[col].ffill()
        df[col] = df[col].bfill()

    df = pd.get_dummies(df,columns=['main'],drop_first=True) # one-hot encode icon and drop 1 col for corr

    df['floor'] = 0.0 # set minimum trend

    m = Prophet(seasonality_mode='multiplicative', changepoint_range=0.95)
    # GET COUNTRY CODE AND ADD PROPHET HOLIDAYS
    country_info = pycountry.countries.get(name=COUNTRY.replace("_"," ")) or pycountry.countries.get(official_name=COUNTRY.replace("_"," ")) or pycountry.countries.get(alpha_3=COUNTRY)
    m.add_country_holidays(country_name=country_info.alpha_2)

    # ADD REGRESSORS
    for reg in [e for e in list(df.columns) if e not in ('ds','y','floor')]:
        m.add_regressor(reg)

    # KEEP ONLY PREVIOUS DAY AND - TO FIT
    dt = datetime.combine(datetime.today() - timedelta(days = LAG_DAY_MODELED), time.min)
    # dont fit with modeled days
    to_fit = df[(~df.ds.dt.date.isin([i[0] for i in modeled_days])) & (df.ds < dt)]

    m.fit(to_fit)

    return m, df

def calculate_confidence_interval_at_day_lvl(m, to_predict):
    samples = m.predictive_samples(to_predict)
    # 1000 SAMPLES FROM PREDICTION
    samples_df = pd.DataFrame.from_records(samples["yhat"])

    samples_df['date'] = to_predict.ds.reset_index().ds
    samples_df['day'] = samples_df['date'].dt.date

    # GET SUM PER DAY FOR YHAT (should be same as sum of hour data)
    upper_lower = samples_df.groupby("day").sum()

    daily_predict = upper_lower.mean(axis=1).reset_index()
    daily_predict.rename(columns={0: "yhat"}, inplace=True)
    daily_predict['day'] = pd.to_datetime(daily_predict['day'])

    # calculate number of hours for filter non-full days
    day_hours = samples_df.day.value_counts().reset_index()
    day_hours.columns = ['day', 'count']
    day_hours['day'] = pd.to_datetime(day_hours['day'])

    daily_predict = daily_predict.merge(day_hours, on = 'day')

    # Upper and lower values of yhat are computed following fbprophet's approach
    daily_predict['yhat_lower'] = upper_lower.apply(lambda x: np.percentile(x, 10), axis=1).tolist()
    daily_predict['yhat_upper'] = upper_lower.apply(lambda x: np.percentile(x, 90), axis=1).tolist()
    # Keep only full days
    daily_predict = daily_predict[daily_predict['count'] == 24]

    # no values below 0
    for c in ['yhat_lower','yhat','yhat_upper']:
        daily_predict.loc[daily_predict[c] < 0, c] = 0

    return daily_predict

def prophet_algorithm(serie_info:ProviderSerieInfo):

    try:
        logging.info(str(serie_info) + str(datetime.now()))

        CITY = serie_info.city
        TYPE = serie_info.type
        PROVIDER = serie_info.provider
        COUNTRY = serie_info.country
        TZ = serie_info.tz
        SOURCE = serie_info.source
        SHARING_TYPE = serie_info.sharing_type

        m, df = fit_model(CITY, TYPE, PROVIDER, COUNTRY, TZ, SOURCE, SHARING_TYPE)

        future = m.make_future_dataframe(periods=PREDICT_HOURS, freq='H', include_history=False)

        with_past = future.merge(df, on='ds',how='outer').sort_values(by=['ds'])

        # FORWARD FEED REG DATA TO KEEP LAST KNOWN VALUES IN CASE OF MISSING DATA
        for reg in [e for e in list(with_past.columns) if e not in ('ds','y','time_lost','before_lag')]:
            with_past[reg] = with_past[reg].ffill()

        # KEEP ONLY CURRENT DAY AND + TO PREDICT
        dt = datetime.combine(datetime.today(), time.min) - timedelta(days=PREV_PREDICT_DAYS) # prev days to test/compare
        to_predict = with_past[with_past.ds >= dt].copy()

        ### PREDICTION : PUT TIME_LOST AT 0 FOR 'IDEAL' RESULT
        to_predict.loc[:,['time_lost']] = 0.0
        to_predict.loc[:,['before_lag']] = 0

        predictions = m.predict(to_predict)
        # WRITE RESULTS TO POSTGRES
        # true for time_lost_0 (we dont write false anymore, that corresponded to non-ideal result))
        write_to_db(CITY, TYPE, PROVIDER, SOURCE, predictions, True, SCHEMA, SHARING_TYPE) 

        # GET CONFIDENCE INTERVAL AT DAY LEVEL
        daily_predict = calculate_confidence_interval_at_day_lvl(m, to_predict)
        
        write_day_agg_to_db(CITY, TYPE, PROVIDER, SOURCE, daily_predict, True, SCHEMA, SHARING_TYPE)

    
    except Exception as e:
        logging.warning('problem with :' + str(serie_info))
        logging.exception(str(e))
        log_error(str(serie_info), str(e))

def run_process(id_process):
    logging.info("-- start queue "+str(id_process))
    run = True
    while run:
        run, serie_info  = get_serie_to_process()
        if run:
            prophet_algorithm(serie_info)


       
if __name__ == "__main__":
    logging.info("-- start")
    logging.basicConfig(level = logging.INFO)

    conn_pg_init = init_rds_connection('PROD', 'psycopg2')
    PREDICT_HOURS = int(os.environ.get("PREDICT_HOURS")) # NB FUTURE HOURS TO PREDICT
    PREV_PREDICT_DAYS = int(os.environ.get("PREV_PREDICT_DAYS")) # NB PAST DAYS TO RE-PREDICT     tmp to change
    MODEL_DAYS = int(os.environ.get("MODEL_DAYS")) # NB DAYS OF DATA USED TO CREATE MODEL
    LAG_DAY_MODELED = 1
    SCHEMA_WRITE = os.environ.get("SCHEMA_WRITE") # schema to write to. PROD : work4 / DEV : staging
    PROCESSES = int(os.environ.get("PROCESSES") or cpu_count())

    if SCHEMA_WRITE == 'PROD':
        SCHEMA = 'work4' 
    elif SCHEMA_WRITE == 'DEV':
        SCHEMA = 'staging'

    logging.info("processes : " + str(PROCESSES))

    pool = Pool(PROCESSES)

    print(pool.map(run_process,range(PROCESSES)))

    logging.info("-- end")

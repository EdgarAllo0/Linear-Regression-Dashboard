# Libraries
import streamlit as st
import pandas as pd
from fredapi import Fred
from pytrends.request import TrendReq
import logging


@st.cache_data
def get_google_trends_mentions(
        symbol: str
) -> pd.DataFrame:

    pytrend = None

    try:
        pytrend = TrendReq(hl='en-US', tz=360)
        pytrend.build_payload(kw_list=[symbol],
                              timeframe='all',
                              geo='US',
                              )
        return pytrend.interest_over_time()

    except Exception as e:
        if pytrend.google_rl:
            logging.error(f'{e}; returning empty DataFrame')
        else:
            logging.error(f'{e}')

        return pd.DataFrame()


@st.cache_data
def get_fred_data(
        symbol: str
) -> pd.DataFrame:

    fred_key = 'b3d5ecf25a74071fbeffd7afb766a3fc '

    fred = Fred(api_key=fred_key)

    df = fred.get_series(symbol)

    return df

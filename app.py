import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import yfinance as yf
from datetime import datetime
import time

current_date = datetime.now().strftime('%Y-%m-%d')

# print execution time for this function

def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {execution_time} seconds")
        return result
    return wrapper

@st.cache_data
@print_execution_time
def download_data(stock_symbol, start_date, end_date=current_date):
    amzn = yf.Ticker(stock_symbol)
    amzn_hist = amzn.history(start=start_date, end=end_date)
    df = amzn_hist.reset_index()
    return df


def predict_stock_price(forecast_period, forecast_feature,df):
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str[:-6])

    df = df[['Date', forecast_feature]]
    df = df.rename(columns={'Date': 'ds', forecast_feature: 'y'})

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=forecast_period)
    forecast = m.predict(future)

    return forecast , m

def take_input():
    stock_symbol = st.selectbox('Select Stock Symbol:', ('AMZN', 'GOOGL', 'AAPL', 'MSFT', 'TSLA'))
    start_date = st.date_input('Select Start Date:', datetime(2021, 1, 1))
    end_date = st.date_input('Select End Date:', datetime.now())

    forecast_feature = st.selectbox('Select Forecasting Feature:', ('Close', 'Open', 'High', 'Low'))
    forecast_period = st.number_input('Forecast Period (days):', min_value=1, max_value=365, value=30)

    return stock_symbol, start_date, end_date, forecast_feature, forecast_period

def show_forecast_plots(forecast, m, forecast_feature, end_date):
        st.subheader('Interactive Forecast Plot')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Price'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(dash='dash'), name='Lower Bound'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(dash='dash'), name='Upper Bound'))
        fig.add_shape(type="line",
                      x0=end_date,
                      y0=min(forecast['yhat_lower']),
                      x1=end_date,
                      y1=max(forecast['yhat_upper']),
                      line=dict(color="red", width=2, dash="dashdot"))
        fig.update_layout(title=f'Forecasted {forecast_feature} Prices',
                          xaxis_title='Date',
                          yaxis_title='Price')
        st.plotly_chart(fig)

        # Plot individual components
        fig = m.plot_components(forecast)
        st.pyplot(fig)

def main():
    st.title('Stock Price Forecasting App')
    
    stock_symbol, start_date, end_date, forecast_feature, forecast_period = take_input()

    # Make predictions and plot
    if st.button('Generate Forecast'):
        with st.spinner('Generating Forecast...'):
            df = download_data(stock_symbol, start_date, end_date)
            forecast, m = predict_stock_price(forecast_period, forecast_feature, df)
        
        # print forecast data
        st.subheader('Forecast Data')
        st.write(forecast)

        show_forecast_plots(forecast, m, forecast_feature, end_date)


if __name__ == '__main__':
    main()

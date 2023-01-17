
# importing libraries

import tensorflow as tf
from pandas.plotting import autocorrelation_plot
import datetime as dt
import math
import os
import pandas_datareader as dr
import cufflinks as cf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yahooFinance
import streamlit as st

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from datetime import datetime

from streamlit_option_menu import option_menu

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from alpha_vantage.fundamentaldata import FundamentalData

from sklearn.metrics import max_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score, mean_poisson_deviance, mean_gamma_deviance
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed


from prophet import Prophet
from prophet.plot import plot_plotly
from gnews import GNews

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use("ggplot")
sns.set_style("darkgrid")
yahooFinance.pdr_override()

#Setting page configuration
st.set_page_config(
    page_title="Final Year Project",
    page_icon="🧊",
    initial_sidebar_state="expanded"
)




# ------------------------------------------------------------------ Tab 1 starts from here ---------------------------------------------------------------

selected = option_menu(
    menu_title=None,
    options=["Predictions", "Stocks Info",
             "Time Series", "Stock News"],
    icons=["graph-up-arrow", "search", "info-circle", "newspaper"],
    default_index=0,
    orientation="horizontal",
)

if selected == "Predictions":
    #tab1, tab2, tab3 = st.tabs(["Stocks Trend Prediction", "Stock Search", "Popular Stocks"])
    # with tab1:
    # Taking input from yahoo finance
    st.title("STOCKS TREND PREDICTION WITH CNN-LSTM")
    st.markdown(
        '<style>h1{color: Aqua; font-size: 35px;}</style>', unsafe_allow_html=True)
    with st.expander("Expand me!"):
        st.write('This tab is about analysis of "Stock Market Predictions". For this, I used CNN-LSTM approach to create a model, then use it to train on stock market data. Further implementation is discussed below...')
    # Taking input for stocks like AAPl, GOOGLE, TESLA etc...

    st.sidebar.image('Logo.png', width=120)
    subheader4 = '''<div><h2 style = "color:Aqua; font-size:23px; font-family: Garamond;">For Stock Prediction</h2></div>'''
    st.sidebar.markdown(subheader4, unsafe_allow_html=True)
    # Taking input for stocks like AAPl, GOOGLE, TESLA etc...
    input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
    start = st.sidebar.date_input("Start Date", dt.date(2010, 1, 1))
    end = st.sidebar.date_input("End Date", dt.date(2018, 12, 30))
    data = dr.data.get_data_yahoo(input, start, end)

    subheader = '''<div><h2 style = "color:#D0ECE7; font-size:23px; font-family: Garamond;">Data Preprocessing & Analysis</h2></div>'''
    st.markdown(subheader, unsafe_allow_html=True)

    # Describing the data
    st.write('Top 5 rows from dataset')
    st.table(data.head())
    st.write('Last 5 rows from dataset')
    st.table(data.tail())
    st.write('Statistical description of dataframe')
    st.table(data.describe())

    st.write("Analyzing The Dataset")
    st.line_chart(data)

    st.line_chart(data['Close'])

    st.line_chart(data['Open'])

    st.line_chart(data['High'])

    st.line_chart(data['Low'])

    data.isnull().sum()
    data.reset_index(drop=True, inplace=True)
    data.fillna(data.mean(), inplace=True)

    # Monthly Average Data
    ma_day = [10, 50, 100]

    for ma in ma_day:
        column_name = "MA for %s days" % (str(ma))
        data[column_name] = pd.DataFrame.rolling(data['Close'], ma).mean()

    st.write("Daily Return Percentage")
    data['Daily Return'] = data['Close'].pct_change()
    # plot the daily return percentage
    fig2 = plt.figure(figsize=(14, 5))
    st.line_chart(data['Daily Return'])
    # st.pyplot(fig2)

    sns.displot(data['Daily Return'].dropna(), bins=100,
                color='green', height=4, aspect=3)
    st.pyplot()

    st.write('After filling the null values with mean...')
    data.reset_index(drop=True, inplace=True)
    data.fillna(data.mean(), inplace=True)
    st.table(data.head())

    data.nunique()
    data.sort_index(axis=1, ascending=True)

    cols_plot = ['Open', 'High', 'Low', 'Close', 'Volume',
                 'MA for 10 days', 'MA for 50 days', 'MA for 100 days', 'Daily Return']
    axes = data[cols_plot].plot(
        marker='.', alpha=0.7, linestyle='None', figsize=(12, 15), subplots=True)
    for ax in axes:
        ax.set_ylabel('Daily trade')

    plt.plot(data['Close'], label="Close price")
    plt.xlabel("Timestamp")
    plt.ylabel("Closing price")
    st.pyplot()

    df = data
    df.describe().transpose()

    # The data has been analysed but it must be converted into data of shape [100,1] to make it easier for CNN to train...
    # Else it won't select necessary features and the model will fail

    from sklearn.model_selection import train_test_split

    X = []
    Y = []
    window_size = 100
    for i in range(1, len(df) - window_size - 1, 1):
        first = df.iloc[i, 2]
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append((df.iloc[i + j, 2] - first) / first)
        temp2.append((df.iloc[i + window_size, 2] - first) / first)
        X.append(np.array(temp).reshape(100, 1))
        Y.append(np.array(temp2).reshape(1, 1))

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=True)

    train_X = np.array(x_train)
    test_X = np.array(x_test)
    train_Y = np.array(y_train)
    test_Y = np.array(y_test)

    train_X = train_X.reshape(train_X.shape[0], 1, 100, 1)
    test_X = test_X.reshape(test_X.shape[0], 1, 100, 1)

    #CNN and LSTM
    # For CNN, the layers are created with sizes 64,128,64 with kernel size = 3. In every layer, TimeDistributed function is added to track the features for every temporal slice of data with respect to time. In between, MaxPooling layers are added.
    # After that, it's passed to Bi-LSTM layers

    # For creating model and training

    model = tf.keras.Sequential()

    # Neural Network model

    # CNN layers
    model.add(TimeDistributed(Conv1D(64, kernel_size=3,
                                     activation='relu', input_shape=(None, 100, 1))))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Flatten()))

    # LSTM layers
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(100, return_sequences=False)))
    model.add(Dropout(0.5))

    # Final layers
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    history = model.fit(train_X, train_Y, validation_data=(
        test_X, test_Y), epochs=40, batch_size=40, verbose=1, shuffle=True)

    subheader1 = '''<div><h2 style = "color:#D0ECE7; font-size:23px; font-family: Garamond;">Trainig the Model</h2></div>'''
    st.markdown(subheader1, unsafe_allow_html=True)
    st.write('Analyzing the TRAIN & VAL Loss')
    figure1 = plt.figure(figsize=(9, 2))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    st.pyplot(figure1)

    st.write('Analyzing the TRAIN-MSE & VAL-MSE')
    figure2 = plt.figure(figsize=(9, 2))
    plt.plot(history.history['mse'], label='train mean squared error')
    plt.plot(history.history['val_mse'], label='val mean squared error')
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    st.pyplot(figure2)

    st.write('Analyzing the TRAIN-MAE & VAL-MAE')
    figure3 = plt.figure(figsize=(9, 2))
    plt.plot(history.history['mae'], label='train mean absolute error')
    plt.plot(history.history['val_mae'], label='val mean absolute error')
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    st.pyplot(figure3)

    model.evaluate(test_X, test_Y)
    # predict probabilities for test set
    yhat_probs = model.predict(test_X, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]

    var = explained_variance_score(test_Y.reshape(-1, 1), yhat_probs)
    #print('Variance: %f' % var)

    r2 = r2_score(test_Y.reshape(-1, 1), yhat_probs)
    #print('R2 Score: %f' % var)

    var2 = max_error(test_Y.reshape(-1, 1), yhat_probs)
    #print('Max Error: %f' % var2)

    subheader2 = '''<div><h2 style = "color:#D0ECE7; font-size:23px; font-family: Garamond;">Predicted Result</h2></div>'''
    st.markdown(subheader2, unsafe_allow_html=True)
    predicted = model.predict(test_X)
    test_label = test_Y.reshape(-1, 1)
    predicted = np.array(predicted[:, 0]).reshape(-1, 1)
    len_t = len(train_X)
    for j in range(len_t, len_t + len(test_X)):
        temp = data.iloc[j, 3]
        test_label[j - len_t] = test_label[j - len_t] * temp + temp
        predicted[j - len_t] = predicted[j - len_t] * temp + temp

    figure4 = plt.figure(figsize=(10, 3))
    plt.plot(predicted, color='green', label='Predicted  Stock Price')
    plt.plot(test_label, color='red', label='Real Stock Price')
    plt.title(' Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(' Stock Price')
    plt.legend()
    st.pyplot(figure4)

    # First we need to save a model
    # In this part, the model is saved and loaded back again. Then, it's made to train again but with different data to check it's loss and prediction

    model.save("model.h5")
    new_model = tf.keras.models.load_model("./model.h5")

    subheader3 = '''<div><h2 style = "color:#D0ECE7; font-size:23px; font-family: Garamond;">Testing The Model</h2></div>'''
    st.markdown(subheader3, unsafe_allow_html=True)
    st.write(
        'Testing the model but with with different dataset to check its loss and prediction')
    data2 = dr.data.get_data_yahoo(input, start='2019-01-01', end='2022-12-31')
    data2.reset_index(drop=True, inplace=True)
    data2.fillna(data.mean(), inplace=True)
    st.table(data2.head())

    df2 = data2

    X = []
    Y = []
    window_size = 100
    for i in range(1, len(df2) - window_size - 1, 1):
        first = df2.iloc[i, 4]
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append((df2.iloc[i + j, 4] - first) / first)
        # for j in range(week):
        temp2.append((df2.iloc[i + window_size, 4] - first) / first)
        # X.append(np.array(stock.iloc[i:i+window_size,4]).reshape(50,1))
        # Y.append(np.array(stock.iloc[i+window_size,4]).reshape(1,1))
        # print(stock2.iloc[i:i+window_size,4])
        X.append(np.array(temp).reshape(100, 1))
        Y.append(np.array(temp2).reshape(1, 1))

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False)

    train_X = np.array(x_train)
    test_X = np.array(x_test)
    train_Y = np.array(y_train)
    test_Y = np.array(y_test)

    train_X = train_X.reshape(train_X.shape[0], 1, 100, 1)
    test_X = test_X.reshape(test_X.shape[0], 1, 100, 1)

    model.evaluate(test_X, test_Y)

    predicted = model.predict(test_X)
    test_label = test_Y.reshape(-1, 1)
    predicted = np.array(predicted[:, 0]).reshape(-1, 1)
    len_t = len(train_X)
    for j in range(len_t, len_t + len(test_X)):
        temp = data2.iloc[j, 3]
        test_label[j - len_t] = test_label[j - len_t] * temp + temp
        predicted[j - len_t] = predicted[j - len_t] * temp + temp

    figure5 = plt.figure(figsize=(10, 3))
    plt.plot(predicted, color='green', label='Predicted  Stock Price')
    plt.plot(test_label, color='red', label='Real Stock Price')
    plt.title(' Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(' Stock Price')
    plt.legend()
    st.pyplot(figure5)

    # EDA Analysis for COVID-19 breakdown
    dataX = dr.data.get_data_yahoo(
        'AAPL', start='2020-01-01', end='2021-11-29')
    dataY = dr.data.get_data_yahoo(
        'AAPL', start='2018-01-01', end='2020-01-01')
    subheader4 = '''<div><h2 style = "color:#D0ECE7; font-size:23px; font-family: Garamond;">EDA For Visualizing The Collecting Data</h2></div>'''
    st.markdown(subheader4, unsafe_allow_html=True)
    st.write('Two Datasets Before & After :red[COVID-19]')
    st.write('First Dataset After :red[COVID-19]')
    st.table(dataX.describe())
    st.write('Second Dataset Before :red[COVID-19]')
    st.table(dataY.describe())

    # sns.distplot(dataX['Close'])
    # sns.distplot(dataY['Close'])
    # st.pyplot()

    st.write('PLots For Data_X')
    fig, ax = plt.subplots(4, 2, figsize=(15, 13))
    sns.boxplot(x=dataX["Close"], ax=ax[0, 0])
    sns.distplot(dataX['Close'], ax=ax[0, 1])
    sns.boxplot(x=dataX["Open"], ax=ax[1, 0])
    sns.distplot(dataX['Open'], ax=ax[1, 1])
    sns.boxplot(x=dataX["High"], ax=ax[2, 0])
    sns.distplot(dataX['High'], ax=ax[2, 1])
    sns.boxplot(x=dataX["Low"], ax=ax[3, 0])
    sns.distplot(dataX['Low'], ax=ax[3, 1])
    plt.tight_layout()
    st.pyplot()

    st.write('PLots For Data_Y')
    fig, ax = plt.subplots(4, 2, figsize=(15, 13))
    sns.boxplot(x=dataY["Close"], ax=ax[0, 0])
    sns.distplot(dataY['Close'], ax=ax[0, 1])
    sns.boxplot(x=dataY["Open"], ax=ax[1, 0])
    sns.distplot(dataY['Open'], ax=ax[1, 1])
    sns.boxplot(x=dataY["High"], ax=ax[2, 0])
    sns.distplot(dataY['High'], ax=ax[2, 1])
    sns.boxplot(x=dataY["Low"], ax=ax[3, 0])
    sns.distplot(dataY['Low'], ax=ax[3, 1])
    plt.tight_layout()
    st.pyplot()

    st.write("Heatmaps Showing The Relationships")
    plt.figure(figsize=(15, 6))
    sns.heatmap(dataX.corr(), cmap=plt.cm.Reds, annot=True)
    plt.title('Displaying the relationship between the features of the data (During COVID)',
              fontsize=13)
    st.pyplot()

    plt.figure(figsize=(15, 6))
    sns.heatmap(dataY.corr(), cmap=plt.cm.Blues, annot=True)
    plt.title('Displaying the relationship between the features of the data (Before COVID)',
              fontsize=13)
    st.pyplot()

    with st.expander("Conclusion!"):
        st.write('Stock movements are highly unpredictable in nature but we can try and build machine learning models to get a rough idea about the trend and future price movements. I used two methods for predicting future prices CNN-LSTM. Other models can also be used and the accuracy can also be improved further by trying out different parameters or changing the layers in the model.')

# ------------------------------------------------------------------ Tab 2 start form here ----------------------------------------------------------------

# with tab2:
if selected == "Stocks Info":
    subheader3 = '''<div><h1 style = "color:Aqua; font-size:35px;">GET INFO. FROM YAHOO FINANCE</h1></div>'''
    st.markdown(subheader3, unsafe_allow_html=True)
    with st.expander("Expand me!"):
        st.write('This tab allows you to search and retrieve information on any stock from Yahoo! Finance using its ticker and display a line charts, the last closing price, and the daily volume. The tab also allows you to view the following additional information for every stock searched')

    def local_css(file_name):
        with open(file_name) as f:
            st.sidebar.markdown(
                f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css("style.css")

    # ticker search feature in sidebar
    subheader5 = '''<div><h2 style = "color:Aqua; font-size:23px; font-family: Garamond;">For Stocks Info</h2></div>'''
    st.sidebar.markdown(subheader5, unsafe_allow_html=True)
    selected_stock = st.sidebar.text_input("Enter stock ticker", "GOOG")
    button_clicked = st.sidebar.button("GO")
    if button_clicked == "GO":
        main()

    # main function
    def main():
        st.subheader("""Daily **Closing Price** For """ + selected_stock)
        # get data on searched ticker
        stock_data = yahooFinance.Ticker(selected_stock)
        # get historical data for searched ticker
        stock_df = stock_data.history(
            period='1d', start='2020-01-01', end=None)
        # print line chart with daily closing prices for searched ticker
        st.line_chart(stock_df.Close)

        st.subheader("""Last **closing Price** For """ + selected_stock)
        # define variable today
        today = datetime.today().strftime('%Y-%m-%d')
        # get current date data for searched ticker
        stock_lastprice = stock_data.history(
            period='1d', start=today, end=today)
        # get current date closing price for searched ticker
        last_price = (stock_lastprice.Close)
        # if market is closed on current date print that there is no data available
        if last_price.empty == True:
            st.write("No data available at the moment")
        else:
            st.table(last_price)

        # get daily volume for searched ticker
        st.subheader("""Daily **Volume** For """ + selected_stock)
        st.line_chart(stock_df.Volume)

        # additional information feature in sidebar
        st.sidebar.subheader("""Display Additional Information""")
        # checkbox to display stock actions for the searched ticker
        actions = st.sidebar.checkbox("Stock Actions")
        if actions:
            st.subheader("""Stock **Actions** For """ + selected_stock)
            display_action = (stock_data.actions)
            if display_action.empty == True:
                st.write("No data available at the moment")
            else:
                st.table(display_action)

        # checkbox to display quarterly financials for the searched ticker
        #financials = st.sidebar.checkbox("Quarterly Financials")
        # if financials:
        #    st.subheader("""**Quarterly financials** for """ + selected_stock)
        #    display_financials = (stock_data.quarterly_financials)
        #    if display_financials.empty == True:
        #        st.write("No data available at the moment")
        #    else:
        #       st.write(display_financials)

        # checkbox to display list of institutional shareholders for searched ticker
        major_shareholders = st.sidebar.checkbox("Institutional Shareholders")
        if major_shareholders:
            st.subheader(
                """**Institutional Investors** For """ + selected_stock)
            display_shareholders = (stock_data.institutional_holders)
            if display_shareholders.empty == True:
                st.write("No data available at the moment")
            else:
                st.table(display_shareholders)

        # checkbox to display quarterly balance sheet for searched ticker
        balance_sheet = st.sidebar.checkbox("Balance Sheet")
        key = '6MK9ZOT10UNQ8MY0'
        fd = FundamentalData(key, output_format='pandas')
        if balance_sheet:
            st.subheader('Balance Sheet For ' + selected_stock)
            balance_sheet = fd.get_balance_sheet_annual(balance_sheet)[0]
            bs = balance_sheet.T[2:]
            bs.columns = list(balance_sheet.T.iloc[0])
            st.write(bs)

        # checkbox to display quarterly cashflow for searched ticker
        cashflow = st.sidebar.checkbox("Cash Flow")
        if cashflow:
            st.subheader('Cash Flow For ' + selected_stock)
            cashflow = fd.get_cash_flow_annual(cashflow)[0]
            cf = cashflow.T[2:]
            cf.columns = list(cashflow.T.iloc[0])
            st.write(cf)


        # checkbox to display list of analysts recommendation for searched ticker
        #analyst_recommendation = st.sidebar.checkbox("Analysts Recommendation")
        # if analyst_recommendation:
        #    st.subheader("""**Analysts recommendation** for """ + selected_stock)
        #    analyst_recommendation = fd.get_recommendations(
        #        analyst_recommendation)[0]
        #    ac = statement.T[2:]
        #   ac.columns = list(statement.T.iloc[0])
        #   st.write(ac)

    if __name__ == "__main__":
        main()


# ----------------------------------------------------------------Tab3 starts from here ------------------------------------------------------------------

# with tab3:
if selected == "Time Series":
    subheader6 = '''<div><h1 style = "color:Aqua; font-size:35px;">TIME SERIES PREDICTION</h1></div>'''
    st.markdown(subheader6, unsafe_allow_html=True)
    with st.expander("Expand me!"):
        st.write('This tab is about to predict time series data using PROPHET & the predictions of n days will start after the end date')
    company = st.text_input(
        "Enter Stock/Index Ticker in Capitals", value='AAPL')
    start = st.date_input("Start Date")
    end = st.date_input("End Date")
    period = st.number_input(
        "Enter number of days you want to predict", step=1, value=365)

    submit = st.button("Submit")
    st.markdown("""---""")
    if submit:
        # get data from yahoo
        df = dr.data.get_data_yahoo(company, start, end)

        # data preprocessing
        df = df.reset_index()
        new_df = df[['Date', 'Close']]
        new_df = new_df.rename(columns={'Date': 'ds', 'Close': 'y'})

        # initialize prophet model
        fp = Prophet(daily_seasonality=True)
        fp.fit(new_df)

        # make future predictions
        future = fp.make_future_dataframe(periods=period)
        forecast = fp.predict(future)

        st.subheader('Predicted Result')
        # Plot the predictions
        fig = plot_plotly(fp, forecast)
        fig.update_xaxes(title_text='Time')
        y_text = '{company_name} Stock price'.format(company_name=company)
        fig.update_yaxes(title_text=y_text)
        fig.update_layout(
            autosize=False,
            width=720,
            height=400,)

        st.plotly_chart(fig)


# ----------------------------------------------------------------------- Tab4 strats from here ---------------------------------------------------------

if selected == "Stock News":
    subheader7 = '''<div><h1 style = "color:Aqua; font-size:35px;">STOCK NEWS</h1></div>'''
    st.markdown(subheader7, unsafe_allow_html=True)
    with st.expander("Expand me!"):
        st.write('This tab will allow you to search for stocks news as the stock market is heavily based on the news, results of the company and its sales. So having a news system integrated in it would help a lot. For this I used the Gnews library for fetching the news of the stocks we want')
    user_input = st.text_input("Enter Stock name")
    state = st.button("Get News!")
    st.markdown("""---""")
    if state:
        news = GNews().get_news(user_input)
        if news:
            for i in news:
                st.markdown(f"**{i['title']}**")
                st.write(f"Published Date - {i['published date']}")
                st.write(i["description"])
                st.markdown(f"[Article Link]({i['url']})")
                st.markdown("""---""")
        else:
            st.write("No news for this stock")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Set title
#st.title("MoMo_bot - Crypto and Stock Price Predictor")
st.set_page_config(page_title="MoMo", layout="wide")
st.markdown("<h1 style='text-align: center; color: orange;'>MoMo_bot - Crypto and Stock Price Predictor</h1>", unsafe_allow_html=True)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """


st.markdown(hide_st_style, unsafe_allow_html=True)
local_css("style/style.css")

st.markdown("""
<style>
.big-font {
    font-size:19px !important;
}
</style>
""", unsafe_allow_html=True)


col1,col2 = st.columns(2)

with col1:
    st.markdown('<h3 style="color: green;">🤖 Generator mode</h3>', unsafe_allow_html=True)

    st.markdown(
        '<p class="big-font">🔧 Machine learning model to generate synthetic data based on historical data</p>',
        unsafe_allow_html=True)
    st.markdown('<p class="big-font">📈 Current historical data are downloaded from yfinance <p class="big-font">',
                unsafe_allow_html=True)
    st.markdown('<p class="big-font">📉 Select different time intervals</p>', unsafe_allow_html=True)

with col2:
    st.markdown('<h3 style="color: green;">🎲 Prediction mode</h3>', unsafe_allow_html=True)

    # st.markdown("")
    st.markdown(
        '<p class="big-font">🔧 Machine learning model to predict crypto & stock prices for time interval given by the user</p>',
        unsafe_allow_html=True)
    # st.markdown("")
    st.markdown('<p class="big-font">📈 This model will be trained using historical & synthetic data<p class="big-font">',
                unsafe_allow_html=True)
    # st.markdown("")
    st.markdown('<p class="big-font">📉 After training the model will produce buy/sell signals and display them here or via Telegram/Viber to the user</p>', unsafe_allow_html=True)

st.markdown('<h3 style="color: orange;">❗ Currently this app is used to display crypto & stock prices, ML models needs to be implemented</h3>', unsafe_allow_html=True)
#ccol1,ccol2,ccol3 = st.columns(3)

#with ccol2:
#    st.markdown('<h3 style="color: purple;">📝 Features</h3>', unsafe_allow_html=True)
#
#    # To-do list with checkboxes
#    task_1 = st.checkbox("Plot crypto & stock prices", value=True)
#    task_2 = st.checkbox("Different time intervals", value=True)
#    task_3 = st.checkbox("Select between open,close,.. pricess")
#    task_4 = st.checkbox("Select between open,close,.. prices")

# Modes selection
modes = st.sidebar.radio("Select Mode", ("Generator", "Prediction"))
done = False

if modes == "Generator":
    # Generator mode
    st.sidebar.header("Generator Settings")




    symbol = st.sidebar.selectbox("Select a symbol", ("BTC-USD", "ETH-USD", "AAPL", "GOOGL"))
    interval = st.sidebar.selectbox("Select interval", ("1d", "1h", "15m", "5m"))
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))
    n_samples = st.sidebar.slider("Number of Synthetic Samples", 1, 100, 10)
    iterations = st.sidebar.slider("Number of Training Iterations", 1, 10, 5)
    random_state = st.sidebar.number_input("Random State", value=42)

    # Load historical data
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval,progress=False)

    # Calculate moving averages
    #data["SMA"] = data["Close"].rolling(short_ma).mean()
    #data["LMA"] = data["Close"].rolling(long_ma).mean()

    # Prepare data for regression
    data = data.dropna()  # Remove rows with missing values

    #data = data.reset_index()
    #data = data[["Date", "Close"]]

    # Create plot using Plotly Express
    fig = px.line(data, x=data.index, y=["Close"], labels={"value": "Price", "variable": "Metric"},
                  title=f"{symbol} Price")
    #fig.update_layout(legend=dict(x=1, y=1), margin=dict(l=4, r=4, t=60, b=4), height=500)
    #fig.update_layout(legend=dict(x=1, y=1), margin=dict(l=4, r=4, t=60, b=4), height=800)
    fig.update_layout(legend=dict(x=1, y=1), margin=dict(l=4, r=4, t=60, b=4), width=1200, height=500)

    # Show plot in Streamlit app
    st.plotly_chart(fig)
    #print(data)





    if st.sidebar.button("Train Generator Model") and done:
        """
        st.header("Generator Model Training")
        # Prepare data for generator training
        with st.spinner("Training in progress..."):
            # Prepare data for generator training
            #features = data.drop(columns=["Date"]).values
            features = data.values
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)

            # Train VAE model
            latent_dim = 2  # Number of latent dimensions
            original_dim = features.shape[1]  # Number of original dimensions

            # Encoder
            encoder_input = tf.keras.Input(shape=(original_dim,))
            encoder_hidden = tf.keras.layers.Dense(10, activation="relu")(encoder_input)
            z_mean = tf.keras.layers.Dense(latent_dim)(encoder_hidden)
            z_log_var = tf.keras.layers.Dense(latent_dim)(encoder_hidden)


            # Sampling function
            def sampling(args):
                z_mean, z_log_var = args
                epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
                return z_mean + tf.math.exp(z_log_var / 2) * epsilon


            z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

            # Decoder
            decoder_hidden = tf.keras.layers.Dense(10, activation="relu")(z)
            decoder_output = tf.keras.layers.Dense(original_dim)(decoder_hidden)

            # VAE model
            vae = tf.keras.Model(encoder_input, decoder_output)

            # VAE loss
            reconstruction_loss = tf.keras.losses.mean_squared_error(encoder_input, decoder_output)
            kl_loss = -0.5 * tf.add_n(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis=-1)
            vae_loss = np.mean(reconstruction_loss + kl_loss)
            vae.add_loss(vae_loss)
            vae.compile(optimizer="adam")

            # Train VAE
            vae.fit(scaled_features, scaled_features, epochs=iterations, batch_size=32)

            # Generate synthetic data
            synthetic_latent = np.random.normal(size=(n_samples, latent_dim))
            synthetic_features = vae.decoder.predict(synthetic_latent)
            synthetic_data = pd.DataFrame(scaler.inverse_transform(synthetic_features), columns=data.columns[1:])

            # Plot MSE values
            mse_values = np.mean(np.square(features - synthetic_features), axis=1)
            plt.plot(range(1, iterations + 1), mse_values)
            plt.xlabel("Iterations")
            plt.ylabel("MSE")
            plt.title("Generator Model Training - MSE")
            st.pyplot()

            # Show synthetic data
            st.header("Synthetic Data")
            st.dataframe(synthetic_data)
            """
elif modes == "Prediction":
    # Prediction mode
    st.sidebar.header("Prediction Settings")

    # Emoji bullet points
    st.markdown("🔮 **Prediction Mode**")
    st.markdown("💡 Train a prediction model using historical data and synthetic data.")
    st.markdown("📊 Make predictions for the selected prediction date using the trained model.")
    st.markdown("📈 Display the historical data, moving averages, and predicted prices.")

    symbol = st.sidebar.selectbox("Select a symbol", ("BTC-USD", "ETH-USD", "AAPL", "GOOGL"))
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))
    prediction_date = st.sidebar.date_input("Prediction date", value=pd.to_datetime("today"))
    interval = st.sidebar.selectbox("Select interval", ("1d", "1h", "15m", "5m"))
    short_ma = st.sidebar.number_input("Enter the time window for the short moving average", min_value=1, max_value=365, value=50)
    long_ma = st.sidebar.number_input("Enter the time window for the long moving average", min_value=1, max_value=365, value=200)

    # Load historical data
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    data["SMA"] = data["Close"].rolling(short_ma).mean()
    data["LMA"] = data["Close"].rolling(long_ma).mean()

    # Create plot using Plotly Express
    fig = px.line(data, x=data.index, y=["Close", "SMA", "LMA"], labels={"value": "Price", "variable": "Metric"},
                  title=f"{symbol} Price and Moving Averages")
    fig.update_layout(legend=dict(x=0, y=1), margin=dict(l=20, r=20, t=60, b=20), height=500)

    # Show plot in Streamlit app
    st.plotly_chart(fig, use_container_width=True)



    if st.sidebar.button("Train Prediction Model") and done:
        """

        # Prepare data for regression
        X = data[["Open", "High", "Low", "Volume", "SMA", "LMA"]]
        y = data["Close"]

        # Train prediction model
        prediction_model = LinearRegression()
        prediction_model.fit(X, y)

        # Get prediction data from Yahoo Finance API
        prediction_data = yf.download(symbol, start=prediction_date, end=prediction_date, interval=interval)
        prediction_data["SMA"] = prediction_data["Close"].rolling(short_ma).mean()
        prediction_data["LMA"] = prediction_data["Close"].rolling(long_ma).mean()

        if prediction_data.empty:
            st.warning("No data available for the selected prediction date.")
        else:
            prediction_X = prediction_data[["Open", "High", "Low", "Volume", "SMA", "LMA"]]
            predictions = prediction_model.predict(prediction_X)

            # Create DataFrame with predicted prices
            prediction_df = pd.DataFrame({"Date": prediction_data.index, "Predicted Close": predictions})
            prediction_df.set_index("Date", inplace=True)

            # Create plot using Plotly Express
            fig = px.line(data, x=data.index, y=["Close", "SMA", "LMA"], labels={"value": "Price", "variable": "Metric"},
                          title=f"{symbol} Price and Moving Averages")
            fig.add_scatter(x=prediction_df.index, y=prediction_df["Predicted Close"], mode="lines", name="Predicted Close")
            fig.update_layout(legend=dict(x=0, y=1), margin=dict(l=20, r=20, t=60, b=20), height=500)

            # Show plot and predictions in Streamlit app
            st.plotly_chart(fig)
            st.write("## Predicted Prices")
            st.dataframe(prediction_df)
        """

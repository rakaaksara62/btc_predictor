import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# ========================
# HEADER PROJECT
# ========================
st.markdown("## Project Responsi Komputasi Numerik by")
st.markdown("### KN-45")
st.markdown("Raka Khairan Taqi Aksara | Vino Ageliansyah | Muhamad Aprizal")
st.markdown("3332250139 | 3332250140 | 3332250151")

st.title("📊 Analisis Data Realtime Bitcoin")
st.write("Metode: Interpolasi dan Regresi Linear")

# ========================
# TOMBOL UPDATE
# ========================
if st.button("🔄 Ambil / Perbarui Data"):

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if 'prices' not in data:
            st.error("API tidak mengembalikan data harga.")
            st.stop()

        prices = data['prices']

    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mengambil data dari API: {e}")
        st.stop()

    # ========================
    # DATAFRAME
    # ========================
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Ambil 50 data terakhir
    df = df.tail(50)

    st.subheader("📋 Data Harga Bitcoin")
    st.dataframe(df)

    # ========================
    # PREPARASI DATA
    # ========================
    x = np.arange(len(df))
    y = df['price'].values

    # ========================
    # INTERPOLASI
    # ========================
    f_interp = interp1d(x, y, kind='linear')

    x_interp = np.linspace(
        x.min(),
        x.max(),
        len(x) * 5
    )

    y_interp = f_interp(x_interp)

    # ========================
    # REGRESI LINEAR
    # ========================
    model = LinearRegression()

    model.fit(
        x.reshape(-1, 1),
        y
    )

    y_pred = model.predict(
        x.reshape(-1, 1)
    )

    # ========================
    # NILAI REGRESI
    # ========================
    coef = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(
        x.reshape(-1, 1),
        y
    )

    # ========================
    # VISUALISASI
    # ========================
    st.subheader("📈 Visualisasi Analisis")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x,
        y,
        'o-',
        label="Data Asli"
    )

    ax.plot(
        x_interp,
        y_interp,
        '--',
        label="Interpolasi"
    )

    ax.plot(
        x,
        y_pred,
        linewidth=2,
        label="Regresi Linear"
    )

    ax.set_xlabel("Waktu (index)")
    ax.set_ylabel("Harga BTC (USD)")
    ax.set_title("Grafik Analisis Harga Bitcoin")
    ax.legend()

    st.pyplot(fig)

    # ========================
    # INFORMASI REGRESI
    # ========================
    st.subheader("📐 Informasi Regresi Linear")

    st.write(f"Koefisien Regresi (slope): {coef:.4f}")
    st.write(f"Intercept: {intercept:.4f}")
    st.write(f"R² Score: {r2:.4f}")

    st.write(
        f"Persamaan Regresi: y = {coef:.4f}x + {intercept:.4f}"
    )

    # ========================
    # PREDIKSI
    # ========================
    st.subheader("🔮 Prediksi Harga Bitcoin")

    future_step = st.slider(
        "Prediksi beberapa langkah ke depan",
        1,
        10,
        3
    )

    future_x = np.array([
        [len(x) + future_step]
    ])

    future_price = model.predict(future_x)

    st.success(
        f"Prediksi harga Bitcoin "
        f"{future_step} langkah ke depan "
        f"adalah: ${future_price[0]:,.2f}"
    )

else:
    st.info(
        "Klik tombol di atas untuk mengambil "
        "data terbaru Bitcoin."
    )
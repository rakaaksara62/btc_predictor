import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# ========================
# KONFIGURASI HALAMAN
# ========================
st.set_page_config(
    page_title="Analisis Bitcoin",
    page_icon="📊",
    layout="wide"
)

# ========================
# HEADER PROJECT
# ========================
st.markdown("## Project Responsi Komputasi Numerik by")
st.markdown("### KN-45")
st.markdown("Raka Khairan Taqi Aksara | Vino Ageliansyah | Muhamad Aprizal")
st.markdown("3332250139 | 3332250140 | 3332250151")
st.markdown(
    "[🔗 Source Code Project](https://github.com/rakaaksara62/btc_predictor)"
)

st.title("📊 Analisis Data Realtime Bitcoin")
st.write("Metode: Interpolasi dan Regresi Linear")

# ========================
# SESSION STATE
# ========================
if 'df' not in st.session_state:
    st.session_state.df = None

# ========================
# TOMBOL AMBIL DATA
# ========================
if st.button("🔄 Ambil / Perbarui Data Bitcoin"):

    url = (
        "https://api.coingecko.com/api/v3/"
        "coins/bitcoin/market_chart"
        "?vs_currency=usd&days=1"
    )

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        data = response.json()

        if 'prices' not in data:
            st.error("API tidak mengembalikan data harga.")
            st.stop()

        prices = data['prices']

        # ========================
        # DATAFRAME
        # ========================
        df = pd.DataFrame(
            prices,
            columns=['timestamp', 'price']
        )

        df['timestamp'] = pd.to_datetime(
            df['timestamp'],
            unit='ms'
        )

        # Ambil 50 data terakhir
        df = df.tail(50)

        # Simpan ke session state
        st.session_state.df = df

        st.success("Data Bitcoin berhasil diperbarui!")

    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mengambil data: {e}")

# ========================
# JIKA DATA SUDAH ADA
# ========================
if st.session_state.df is not None:

    df = st.session_state.df

    # ========================
    # TAMPILKAN DATA
    # ========================
    st.subheader("📋 Data Harga Bitcoin")

    st.dataframe(
        df,
        use_container_width=True
    )

    # ========================
    # HITUNG INTERVAL WAKTU
    # ========================
    time_diff = (
        df['timestamp'].iloc[1]
        - df['timestamp'].iloc[0]
    )

    minutes_per_step = (
        time_diff.total_seconds() / 60
    )

    # ========================
    # PREPARASI DATA
    # ========================
    x = np.arange(len(df))
    y = df['price'].values

    # ========================
    # INTERPOLASI
    # ========================
    f_interp = interp1d(
        x,
        y,
        kind='linear'
    )

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
    # VISUALISASI ANALISIS
    # ========================
    st.subheader("📈 Visualisasi Analisis")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Data asli
    ax.plot(
        x,
        y,
        'o-',
        label="Data Asli"
    )

    # Interpolasi
    ax.plot(
        x_interp,
        y_interp,
        '--',
        label="Interpolasi"
    )

    # Regresi
    ax.plot(
        x,
        y_pred,
        linewidth=3,
        label="Regresi Linear"
    )

    ax.set_xlabel("Index Waktu")
    ax.set_ylabel("Harga BTC (USD)")
    ax.set_title("Grafik Analisis Harga Bitcoin")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # ========================
    # INFORMASI REGRESI
    # ========================
    st.subheader("📐 Informasi Regresi Linear")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Koefisien Regresi",
            f"{coef:.4f}"
        )

    with col2:
        st.metric(
            "Intercept",
            f"{intercept:.2f}"
        )

    with col3:
        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

    st.write(
        f"""
        ### Persamaan Regresi

        y = {coef:.4f}x + {intercept:.4f}
        """
    )

    # ========================
    # PREDIKSI
    # ========================
    st.subheader("🔮 Prediksi Harga Bitcoin")

    future_step = st.slider(
        "Prediksi beberapa langkah ke depan",
        min_value=1,
        max_value=10,
        value=3
    )

    # Konversi ke menit
    future_minutes = (
        future_step * minutes_per_step
    )

    # Data masa depan
    future_x = np.array([
        [len(x) + future_step]
    ])

    # Prediksi harga
    future_price = model.predict(
        future_x
    )

    st.success(
        f"""
        Prediksi harga Bitcoin
        sekitar {future_minutes:.0f} menit ke depan adalah:

        ${future_price[0]:,.2f}
        """
    )

    # ========================
    # VISUALISASI PREDIKSI
    # ========================
    st.subheader("📉 Visualisasi Prediksi")

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # Data asli
    ax2.plot(
        x,
        y,
        'o-',
        label="Data Asli"
    )

    # Garis regresi
    ax2.plot(
        x,
        y_pred,
        linewidth=2,
        label="Regresi Linear"
    )

    # Titik prediksi
    ax2.scatter(
        len(x) + future_step,
        future_price[0],
        s=150,
        label="Prediksi Masa Depan"
    )

    # Garis bantu
    ax2.plot(
        [x[-1], len(x) + future_step],
        [y[-1], future_price[0]],
        '--'
    )

    ax2.set_xlabel("Index Waktu")
    ax2.set_ylabel("Harga BTC (USD)")
    ax2.set_title("Grafik Prediksi Harga Bitcoin")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)

    # ========================
    # INFORMASI TAMBAHAN
    # ========================
    st.subheader("⏱ Informasi Interval Data")

    st.info(
        f"""
        Interval antar data ≈
        {minutes_per_step:.0f} menit per data point.

        Jadi:
        1 langkah ≈ {minutes_per_step:.0f} menit
        """
    )

else:
    st.info(
        "Klik tombol di atas untuk mengambil "
        "data Bitcoin terbaru."
    )
st.markdown(
    "[🔗 Source Code Project](https://github.com/rakaaksara62/btc_predictor)"
)
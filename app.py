import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

# ========================
#   ESTILO PERSONALIZADO
# ========================
# Colores: Verde + Negro
st.markdown(
    """
    <style>
    body {
        background-color: #0A0A0A;
        color: #00FF7F;
    }
    .stApp {
        background-color: #0A0A0A;
        color: #00FF7F;
    }
    .css-18e3th9, .css-1d391kg {
        background-color: #0A0A0A;
        color: #00FF7F;
    }
    .stSelectbox, .stNumberInput, .stTextInput, .stDateInput {
        background-color: #0A0A0A !important;
        color: #00FF7F !important;
    }
    .stButton>button {
        background-color: #00FF7F;
        color: black;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Simulador de Portafolios – Qori Clean (Verde & Negro)")

# ========================
#     CONFIGURACIÓN BASE
# ========================
# === PORTAFOLIOS ORIGINALES ===
portafolios = {
    'Bajo': {
        'tickers': ['AGG','GLD','LQD','VIG'],
        'pesos': np.array([0.4, 0.1, 0.3, 0.2])
    },
    'Medio': {
        'tickers': ['AGG','GLD','LQD','QQQ','SPY'],
        'pesos': np.array([0.25, 0.1, 0.15, 0.25, 0.25])
    },
    'Alto': {
        'tickers': ['ARKK','NVDA','QQQ','TSLA','XBI'],
        'pesos': np.array([0.2, 0.2, 0.3, 0.1, 0.2])
    }
}

# Selección de portafolio
selected_portfolio = st.selectbox('Selecciona un portafolio', list(portafolios.keys()))

tickers = portafolios[selected_portfolio]['tickers']
preset_weights = portafolios[selected_portfolio]['pesos']
weights = {}
st.header("Pesos del Portafolio (predefinidos, editables)")

cols = st.columns(4)
for i, ticker in enumerate(tickers):
    col = cols[i % 4]
    weights[ticker] = col.number_input(
        f"Peso {ticker}",
        min_value=0.0,
        max_value=1.0,
        value=float(preset_weights[i]),
        step=0.01
    )

total_weight = sum(weights.values())(weights.values())
st.write(f"**Peso total:** {total_weight:.2f}")

if abs(total_weight - 1) > 0.001:
    st.error("Los pesos deben sumar 1.0 para continuar.")
    st.stop()

# ========================
#      APORTES OPCIONALES
# ========================
st.header("¿Se harán contribuciones?")
contrib_bool = st.radio("Selecciona una opción:", ["No", "Sí"])

contrib_amount = 0
contrib_freq = None

if contrib_bool == "Sí":
    contrib_amount = st.number_input("Monto de contribución:", min_value=0.0, step=10.0)
    contrib_freq = st.selectbox("Frecuencia de contribución:", ["Mensual", "Anual"])

# ========================
#  PERÍODO DE SIMULACIÓN
# ========================
start_date = st.date_input("Fecha de inicio", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("Fecha final", value=pd.to_datetime("2025-01-01"))

# ========================
#  DESCARGA DE DATOS
# ========================
@st.cache_data
def load_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end)["Adj Close"]

data = load_data(tickers, start_date, end_date)

# ========================
#      CÁLCULOS
# ========================
returns = data.pct_change().dropna()
weighted_returns = (returns * np.array(list(weights.values()))).sum(axis=1)

portfolio_value = [1]

for r in weighted_returns:
    new_value = portfolio_value[-1] * (1 + r)

    # Aporte si corresponde
    if contrib_bool == "Sí":
        if contrib_freq == "Mensual":
            new_value += contrib_amount / 12
        elif contrib_freq == "Anual":
            new_value += contrib_amount

    portfolio_value.append(new_value)

portfolio_value = pd.Series(portfolio_value, index=[returns.index.min()] + list(returns.index))

# ========================
#      GRÁFICOS
# ========================
st.header("Crecimiento del Portafolio")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(portfolio_value)
ax.set_title("Valor del Portafolio en el Tiempo", color="#00FF7F")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.tick_params(colors="#00FF7F")
for spine in ax.spines.values():
    spine.set_color("#00FF7F")
st.pyplot(fig)

# ========================
#  MÉTRICAS BÁSICAS (sin anual equivalente)
# ========================
st.header("Métricas del Portafolio")
total_return = portfolio_value.iloc[-1] - 1
volatility = weighted_returns.std()
sharpe = (weighted_returns.mean()) / volatility if volatility > 0 else 0

st.write(f"**Retorno total:** {total_return:.2%}")
st.write(f"**Volatilidad:** {volatility:.2%}")
st.write(f"**Sharpe Ratio:** {sharpe:.3f}")

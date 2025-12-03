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

# FIX: quitar el paréntesis extra
total_weight = sum(weights.values())
st.write(f"**Peso total:** {total_weight:.2f}")

if abs(total_weight - 1) > 0.001:
    st.error("Los pesos deben sumar 1.0 para continuar.")
    st.stop()

# ========================
#      APORTES OPCIONALES
# ========================
st.header("¿Se harán contribuciones?")
contrib_bool = st.radio("Selecciona una opción:", ["No", "Sí"])

contrib_amount = 0.0
contrib_freq = None

if contrib_bool == "Sí":
    contrib_amount = st.number_input("Monto de contribución (por periodo):", min_value=0.0, step=10.0, value=0.0)
    contrib_freq = st.selectbox("Frecuencia de contribución (define el 'periodo'):", ["Mensual", "Anual"])

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
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    # Normalizar la salida a DataFrame con tickers como columnas de precios ajustados
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.get_level_values(0):
            adj = df['Adj Close']
        elif 'Close' in df.columns.get_level_values(0):
            adj = df['Close']
        else:
            adj = df.dropna(axis=1, how='all')
    else:
        if 'Adj Close' in df.columns:
            adj = df['Adj Close']
        elif 'Close' in df.columns:
            adj = df['Close']
        else:
            adj = df
    # Si es Series (un ticker), convertir a DataFrame
    if isinstance(adj, pd.Series):
        adj = adj.to_frame(name=tickers[0])
    return adj.dropna(how='all')

data = load_data(tickers, start_date, end_date)
if data.empty:
    st.error("No se encontraron datos para los tickers y rango seleccionados.")
    st.stop()

# ========================
#      CÁLCULOS
# ========================
# Alinear pesos por nombres de ticker (evita errores por distinto orden)
weights_series = pd.Series(weights)
available = [c for c in data.columns if c in weights_series.index]
if len(available) != len(weights_series):
    st.warning(f"Algunos tickers no tienen datos y serán excluidos: {set(weights_series.index) - set(available)}")
    weights_series = weights_series.loc[available]
    weights_series = weights_series / weights_series.sum()

# Calcular retornos
returns = data[available].pct_change().dropna()
if returns.empty:
    st.error("No hay suficientes datos para calcular rendimientos.")
    st.stop()

# Weighted returns alineando por columna
weighted_returns = returns.mul(weights_series, axis=1).sum(axis=1)

# Construir valor del portafolio (comienza en 1)
portfolio_values = [1.0]
dates = []
for date, r in weighted_returns.items():
    prev = portfolio_values[-1]
    new_value = prev * (1 + r)
    # Contribuciones: interpretar contrib_amount como monto por periodo seleccionado
    if contrib_bool == "Sí" and contrib_amount > 0:
        if contrib_freq == "Mensual":
            new_value += contrib_amount  # se añade por cada periodo (asumir que el periodo coincide con la frecuencia de returns)
        elif contrib_freq == "Anual":
            # Añadir la contribución anual si la fecha es en enero (heurística simple)
            try:
                if hasattr(date, 'month') and date.month == 1:
                    new_value += contrib_amount
            except Exception:
                pass
    portfolio_values.append(new_value)
    dates.append(date)

# Índice con punto inicial ligeramente anterior para mostrar capital inicial
initial_date = returns.index[0] - MonthEnd(1)
all_dates = [initial_date] + dates
portfolio_series = pd.Series(portfolio_values, index=all_dates)

# ========================
#      GRÁFICOS
# ========================
st.header("Crecimiento del Portafolio")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(portfolio_series.index, portfolio_series.values, color="#00FF7F", marker='o', linewidth=1)
ax.set_title("Valor del Portafolio en el Tiempo", color="#00FF7F")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.tick_params(colors="#00FF7F")
for spine in ax.spines.values():
    spine.set_color("#00FF7F")
ax.set_ylabel("Valor")
st.pyplot(fig)

# ========================
#  MÉTRICAS BÁSICAS (anualizadas donde corresponde)
# ========================
st.header("Métricas del Portafolio")
total_return = portfolio_series.iloc[-1] - portfolio_series.iloc[0]

# Inferir frecuencia para anualizar
inferred = pd.infer_freq(returns.index)
if inferred is None:
    periods_per_year = 12
else:
    if inferred.startswith('B') or inferred.startswith('D'):
        periods_per_year = 252
    elif inferred.startswith('M'):
        periods_per_year = 12
    elif inferred.startswith('A') or inferred.startswith('Y'):
        periods_per_year = 1
    else:
        periods_per_year = 12

mean_period = weighted_returns.mean()
vol_period = weighted_returns.std()
annual_return = mean_period * periods_per_year
annual_vol = vol_period * np.sqrt(periods_per_year)
sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

st.write(f"**Retorno total:** {total_return:.2%}")
st.write(f"**Retorno anual (aprox.):** {annual_return:.2%}")
st.write(f"**Volatilidad (anualizada):** {annual_vol:.2%}")
st.write(f"**Sharpe Ratio (aprox.):** {sharpe:.3f}")

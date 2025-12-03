import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

# ========================
#   CONFIGURACIÓN DE LA PÁGINA
# ========================
st.set_page_config(
    page_title="Simulador de Portafolios — Qori",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
#   ESTILO PROFESIONAL (VERDE & NEGRO)
# ========================
st.markdown(
    """
    <style>
    :root{
        --qori-green: #00FF7F;
        --qori-dark: #060606;
        --qori-grey: #2b2b2b;
    }
    /* Fuente y reset ligero */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, #root, .streamlit-container {
        background-color: var(--qori-dark) !important;
        color: var(--qori-green) !important;
        font-family: 'Inter', sans-serif;
    }

    /* App container */
    .stApp {
        background: linear-gradient(180deg, rgba(6,6,6,1) 0%, rgba(10,10,10,1) 100%) !important;
        padding-top: 1rem !important;
    }

    /* Card style for main content */
    .main-card {
        background-color: #080808;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
        border: 1px solid rgba(0,255,127,0.08);
    }

    /* Sidebar */
    .css-1d391kg .css-1d391kg, .sidebar .stSidebar {
        background-color: #070707 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #070707 !important;
        border-right: 1px solid rgba(0,255,127,0.06);
    }

    /* Headings and titles */
    .stMarkdown h1, .stTitle {
        color: var(--qori-green) !important;
    }
    .stMarkdown h2, .stHeader {
        color: var(--qori-green) !important;
    }

    /* Inputs */
    input, .stTextInput, .stNumberInput, .stSelectbox, .stDateInput {
        background-color: #0b0b0b !important;
        color: var(--qori-green) !important;
        border: 1px solid rgba(0,255,127,0.08) !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, rgba(0,255,127,0.95), rgba(0,205,100,0.95)) !important;
        color: #000 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 12px !important;
    }

    /* Metrics and numbers */
    .stMetric {
        background: linear-gradient(180deg, rgba(10,10,10,0.6), rgba(6,6,6,0.6));
        border: 1px solid rgba(0,255,127,0.06);
        border-radius: 8px;
        padding: 10px;
    }

    /* Table */
    .stDataFrame table {
        background-color: #0b0b0b;
        color: var(--qori-green);
    }

    /* Smaller tweaks */
    .css-1adrfps {color: var(--qori-green) !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# App title, with a small subtitle
st.markdown("<div class='main-card'><h1>Simulador de Portafolios — Qori Clean</h1><p style='color: #9ef7c7'>Interfaz profesional en tonos verde & negro</p></div>", unsafe_allow_html=True)

# ========================
#     CONFIGURACIÓN BASE
# ========================
# === PORTAFOLIOS ORIGINALES ===
portafolios = {
    'Bajo': {
        'tickers': ['AGG', 'GLD', 'LQD', 'VIG'],
        'pesos': np.array([0.4, 0.1, 0.3, 0.2])
    },
    'Medio': {
        'tickers': ['AGG', 'GLD', 'LQD', 'QQQ', 'SPY'],
        'pesos': np.array([0.25, 0.1, 0.15, 0.25, 0.25])
    },
    'Alto': {
        'tickers': ['ARKK', 'NVDA', 'QQQ', 'TSLA', 'XBI'],
        'pesos': np.array([0.2, 0.2, 0.3, 0.1, 0.2])
    }
}

# ------------------------
# CONTROLES EN LA BARRA LATERAL
# ------------------------
st.sidebar.header("Configuración del Portafolio")

selected_portfolio = st.sidebar.selectbox('Selecciona un portafolio', list(portafolios.keys()))
tickers = portafolios[selected_portfolio]['tickers']
preset_weights = portafolios[selected_portfolio]['pesos']

st.sidebar.markdown("### Pesos (ajustables)")
weights = {}
for i, ticker in enumerate(tickers):
    weights[ticker] = st.sidebar.number_input(
        f"{ticker}",
        min_value=0.0,
        max_value=1.0,
        value=float(preset_weights[i]),
        step=0.01,
        format="%.2f"
    )

# Contribuciones en sidebar
st.sidebar.markdown("---")
st.sidebar.header("Contribuciones")
contrib_bool = st.sidebar.radio("¿Se harán contribuciones?", ["No", "Sí"])
contrib_amount = 0.0
contrib_freq = None
if contrib_bool == "Sí":
    contrib_amount = st.sidebar.number_input("Monto por periodo:", min_value=0.0, step=10.0, value=0.0)
    contrib_freq = st.sidebar.selectbox("Frecuencia:", ["Mensual", "Anual"])

# Período en sidebar
st.sidebar.markdown("---")
st.sidebar.header("Periodo de Simulación")
start_date = st.sidebar.date_input("Fecha de inicio", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("Fecha final", value=pd.to_datetime("2025-01-01"))

# Mostrar peso total y validación (en main)
total_weight = sum(weights.values())
st.markdown(f"<div class='main-card'><strong>Peso total:</strong> {total_weight:.2f}</div>", unsafe_allow_html=True)
if abs(total_weight - 1) > 0.001:
    st.error("Los pesos deben sumar 1.0 para continuar.")
    st.stop()

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
st.markdown("<div class='main-card'><h2>Crecimiento del Portafolio</h2></div>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(portfolio_series.index, portfolio_series.values, color="#00FF7F", marker=None, linewidth=2)
ax.fill_between(portfolio_series.index, portfolio_series.values, color="#003d19", alpha=0.15)
ax.set_title("Valor del Portafolio en el Tiempo", color="#00FF7F", fontsize=14)
ax.set_facecolor("#060606")
fig.patch.set_facecolor("#060606")
ax.grid(color="#03300f", linestyle='--', linewidth=0.4, alpha=0.6)
ax.tick_params(colors="#00FF7F")
for spine in ax.spines.values():
    spine.set_color("#00FF7F")
ax.set_ylabel("Valor", color="#00FF7F")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# ========================
#  MÉTRICAS BÁSICAS (anualizadas donde corresponde)
# ========================
st.markdown("<div class='main-card'><h2>Métricas del Portafolio</h2></div>", unsafe_allow_html=True)
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

# Mostrar métricas en columnas para mejor apariencia
c1, c2, c3, c4 = st.columns(4)
c1.metric("Retorno total", f"{total_return:.2%}")
c2.metric("Retorno anual (aprox.)", f"{annual_return:.2%}")
c3.metric("Volatilidad (anualizada)", f"{annual_vol:.2%}")
c4.metric("Sharpe Ratio (aprox.)", f"{sharpe:.3f}")

# Mostrar tabla de pesos finales y returns recientes
st.markdown("<div class='main-card'><h3>Detalle</h3></div>", unsafe_allow_html=True)
st.write(pd.DataFrame({
    "Ticker": weights_series.index,
    "Peso": weights_series.values
}).set_index("Ticker"))

# Pequeña nota al final
st.markdown("<div style='margin-top:12px;color:#9ef7c7;'>Interfaz actualizada: diseño más profesional, esquema de color verde/negro y disposición en sidebar para controles. Ajusta pesos, fechas y contribuciones desde la barra lateral.</div>", unsafe_allow_html=True)

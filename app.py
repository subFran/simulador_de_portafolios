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
#   ESTILO (SOLO COLORES DE LA PÁGINA)
# ========================
st.markdown(
    """
    <style>
    :root{
        --qori-green: #00FF7F;
        --qori-dark: #060606;
    }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, #root, .streamlit-container {
        background-color: var(--qori-dark) !important;
        color: var(--qori-green) !important;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, rgba(6,6,6,1) 0%, rgba(10,10,10,1) 100%) !important;
    }
    [data-testid="stSidebar"] {
        background-color: #070707 !important;
        border-right: 1px solid rgba(0,255,127,0.06);
    }
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Simulador de Portafolios – Qori (verde & negro)")

# ========================
#     CONFIGURACIÓN BASE
# ========================
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
# BARRA LATERAL: CONTROLES
# ------------------------
st.sidebar.header("Configuración del Portafolio")

selected_portfolio = st.sidebar.selectbox('Selecciona un portafolio', list(portafolios.keys()))
tickers = portafolios[selected_portfolio]['tickers']
preset_weights = portafolios[selected_portfolio]['pesos']

# Opción: bloquear o permitir edición de pesos (resuelve tu comentario)
edit_weights = st.sidebar.checkbox("Editar pesos manualmente", value=False, help="Desactiva para usar los pesos predefinidos (no editables).")

weights = {}
if edit_weights:
    st.sidebar.markdown("### Pesos (ajustables)")
    for i, ticker in enumerate(tickers):
        weights[ticker] = st.sidebar.number_input(
            f"{ticker}",
            min_value=0.0,
            max_value=1.0,
            value=float(preset_weights[i]),
            step=0.01,
            format="%.2f"
        )
else:
    # Mostrar pesos predefinidos en la sidebar (no editables)
    st.sidebar.markdown("### Pesos (predefinidos)")
    for i, ticker in enumerate(tickers):
        st.sidebar.write(f"{ticker}: {preset_weights[i]:.2f}")
    weights = {tickers[i]: float(preset_weights[i]) for i in range(len(tickers))}

# ------------------------
# Moneda y tipo de cambio (se restauró, como pediste)
# ------------------------
st.sidebar.markdown("---")
currency = st.sidebar.selectbox("Moneda de visualización", ["USD", "PEN"])
exchange_rate = None
if currency == "PEN":
    exchange_rate = st.sidebar.number_input("Tipo de cambio (PEN por USD)", min_value=0.0, value=3.8, step=0.01)

# Monto inicial (separado de contribuciones)
st.sidebar.markdown("---")
initial_capital = st.sidebar.number_input("Monto inicial (en la moneda seleccionada)", min_value=0.0, value=1000.0, step=10.0)

# Contribuciones: se muestran sólo si se elige "Sí" y serán mensuales (como pediste)
st.sidebar.markdown("---")
st.sidebar.header("Contribuciones")
contrib_bool = st.sidebar.radio("¿Se harán contribuciones periódicas?", ["No", "Sí"])
contrib_amount = 0.0
contrib_freq = None
if contrib_bool == "Sí":
    contrib_amount = st.sidebar.number_input("Monto de contribución (por periodo - mensual)", min_value=0.0, value=0.0, step=10.0)
    contrib_freq = "Mensual"  # for clarity: we force monthly as you requested

# Fechas
st.sidebar.markdown("---")
start_date = st.sidebar.date_input("Fecha de inicio", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("Fecha final", value=pd.to_datetime("2025-01-01"))

# Mostrar peso total y validación (en main)
total_weight = sum(weights.values())
st.write(f"**Peso total:** {total_weight:.2f}")
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
weights_series = pd.Series(weights)
available = [c for c in data.columns if c in weights_series.index]
if len(available) != len(weights_series):
    st.warning(f"Algunos tickers no tienen datos y serán excluidos: {set(weights_series.index) - set(available)}")
    weights_series = weights_series.loc[available]
    # Re-normalizar si se excluyeron tickers
    weights_series = weights_series / weights_series.sum()

# Calcular retornos
returns = data[available].pct_change().dropna()
if returns.empty:
    st.error("No hay suficientes datos para calcular rendimientos.")
    st.stop()

# Weighted returns alineando por columna
weighted_returns = returns.mul(weights_series, axis=1).sum(axis=1)

# Construir valor del portafolio partiendo de initial_capital (en la moneda seleccionada)
portfolio_values = [float(initial_capital)]
dates = []
for date, r in weighted_returns.items():
    prev = portfolio_values[-1]
    new_value = prev * (1 + r)
    # Contribuciones: si el usuario dijo Sí, se añade la contribución por periodo (mensual)
    if contrib_bool == "Sí" and contrib_amount > 0:
        # Contrib_amount ya está en la moneda seleccionada y es independiente del monto inicial
        new_value += float(contrib_amount)
    portfolio_values.append(new_value)
    dates.append(date)

# Índice con punto inicial ligeramente anterior para mostrar capital inicial
initial_date = returns.index[0] - MonthEnd(1)
all_dates = [initial_date] + dates
portfolio_series = pd.Series(portfolio_values, index=all_dates)

# ========================
#      GRÁFICOS (MANTENER GRÁFICO COMO ESTABA: solo cambiamos colores de la página)
# ========================
st.header("Crecimiento del Portafolio")
fig, ax = plt.subplots(figsize=(10,5))
# Usamos estilo de gráfico original (verde sobre negro) tal como pediste que no cambiara
ax.plot(portfolio_series.index, portfolio_series.values, color="#00FF7F", marker='o', linewidth=1)
ax.set_title("Valor del Portafolio en el Tiempo", color="#00FF7F")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.tick_params(colors="#00FF7F")
for spine in ax.spines.values():
    spine.set_color("#00FF7F")
ax.set_ylabel(f"Valor ({currency})", color="#00FF7F")
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

# Mostrar métricas (incluimos etiqueta de moneda)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Retorno total", f"{total_return:.2%}")
c2.metric("Retorno anual (aprox.)", f"{annual_return:.2%}")
c3.metric("Volatilidad (anualizada)", f"{annual_vol:.2%}")
c4.metric("Sharpe Ratio (aprox.)", f"{sharpe:.3f}")

# Mostrar tabla de pesos finales y returns recientes
st.subheader("Detalle de Pesos")
st.write(pd.DataFrame({
    "Ticker": weights_series.index,
    "Peso": weights_series.values
}).set_index("Ticker"))

# Si la moneda es PEN y se ingreso tipo de cambio, mostrar equivalentes en USD como referencia
if currency == "PEN" and exchange_rate and exchange_rate > 0:
    st.markdown(f"**Referencia:** 1 USD = {exchange_rate:.2f} PEN")
    st.markdown("Últimos valores del portafolio (moneda seleccionada y equivalente USD):")
    df_values = pd.DataFrame({
        "Fecha": portfolio_series.index,
        f"Valor ({currency})": portfolio_series.values,
        "Valor (USD)": portfolio_series.values / exchange_rate
    })
    st.write(df_values.tail(5).set_index("Fecha"))
else:
    st.markdown("Últimos valores del portafolio:")
    df_values = pd.DataFrame({
        "Fecha": portfolio_series.index,
        f"Valor ({currency})": portfolio_series.values
    })
    st.write(df_values.tail(5).set_index("Fecha"))

st.markdown("""
Nota:
- He restaurado la opción de moneda (USD/PEN) y la entrada de tipo de cambio para referencia.
- El gráfico mantuvo su apariencia original (línea verde sobre fondo negro) —solo cambié los colores de la página y la disposición de controles.
- Las contribuciones son independientes del monto inicial: si seleccionas "Sí", el monto que pongas se añadirá cada periodo (mensual) aparte del capital inicial.
""")

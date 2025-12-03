# ============================================ 
# CONFIGURACIÓN Y LIBRERÍAS 
# ============================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from datetime import datetime

st.set_page_config(page_title="Simulador de Portafolio", layout="wide")

# ======== ESTILO ==========
st.markdown("""<style>
:root{
    --card-radius: 12px;
    --card-padding: 18px;
    --container-max-w: 1100px;
    --sidebar-min-w: 300px;
    --shadow-1: 0 6px 18px rgba(0,0,0,0.08);
    --shadow-2: 0 10px 30px rgba(0,0,0,0.06);
}
</style>""", unsafe_allow_html=True)

st.title("Simulador de Portafolio de Inversión")

# ============================================
# 1. PORTAFOLIOS
# ============================================

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

# ============================================
# 2. DESCARGA DE DATOS
# ============================================

@st.cache_data
def get_adj_close(tickers):
    data = yf.download(
        tickers,
        start='2022-12-01',
        end='2025-11-30',
        interval='1mo',
        auto_adjust=True,
        progress=False
    )
    if isinstance(data.columns, pd.MultiIndex):
        return data['Close'].dropna()
    return data.dropna()

# ============================================
# 3. CÁLCULO DE ESTADÍSTICAS
# ============================================

def portafolio_stats(tickers, pesos):
    adj = get_adj_close(tickers)
    returns = adj.pct_change().dropna()
    mean_m = returns.mean().dot(pesos)
    cov = returns.cov()
    port_var = float(pesos.T @ cov.values @ pesos)
    return mean_m, port_var, np.sqrt(port_var)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("Parámetros de Simulación")

tipo_portafolio = st.sidebar.selectbox("Perfil de Riesgo", ['Bajo', 'Medio', 'Alto'])
monto_inicial = st.sidebar.number_input("Monto Inicial", min_value=1000, value=10000)
anos = st.sidebar.slider("Horizonte (Años)", 1, 10, 5)
moneda = st.sidebar.selectbox("Moneda del monto inicial", ["PEN", "USD"])
tipo_cambio = st.sidebar.number_input("Tipo de Cambio (PEN/USD)", value=3.80)
simulaciones = st.sidebar.slider("Simulaciones Monte Carlo", 100, 5000, 1000)

# ============================================
# OBTENER PORTAFOLIO
# ============================================

tickers = portafolios[tipo_portafolio]['tickers']
pesos = portafolios[tipo_portafolio]['pesos']

st.sidebar.markdown("### Composición del Portafolio")
st.sidebar.table(pd.DataFrame({"Ticker": tickers, "Peso": pesos}))

# ============================================
# LÓGICA PRINCIPAL
# ============================================

mean_m, var_m, std_m = portafolio_stats(tickers, pesos)

# Conversión de moneda
if moneda == "USD":
    monto = monto_inicial
else:
    monto = monto_inicial / tipo_cambio  # PEN → USD

# PROYECCIÓN
meses = anos * 12
tiempo = np.arange(meses + 1)
valores = monto * (1 + mean_m) ** tiempo

# MONTE CARLO
simulaciones_array = np.zeros((simulaciones, meses + 1))
simulaciones_array[:, 0] = monto

for t in range(1, meses + 1):
    r = np.random.normal(mean_m, std_m, simulaciones)
    simulaciones_array[:, t] = simulaciones_array[:, t-1] * (1 + r)

p5 = np.percentile(simulaciones_array, 5, axis=0)
p95 = np.percentile(simulaciones_array, 95, axis=0)

# ============================================
# 🔥 CONTRIBUCIONES
# ============================================

st.subheader("Contribuciones adicionales")

if anos == 1:
    contrib_mensual = st.number_input(
        "Aportación mensual (USD)",
        min_value=0.0, value=0.0, step=10.0
    )
    contrib_anual = 0
else:
    contrib_mensual = st.number_input(
        "Aportación mensual (USD)",
        min_value=0.0, value=0.0, step=10.0
    )
    contrib_anual = st.number_input(
        "Aportación anual (USD)",
        min_value=0.0, value=0.0, step=50.0
    )

total_meses = anos * 12

contrib_meses = np.zeros(total_meses)
if contrib_mensual > 0:
    contrib_meses[:] = contrib_mensual

contrib_anios = np.zeros(total_meses)
if anos > 1 and contrib_anual > 0:
    for year in range(anos):
        contrib_anios[year * 12] = contrib_anual

aporte_total = contrib_meses + contrib_anios

valores_con_aportes = []
valor = monto

for i in range(total_meses):
    valor = valor * (1 + mean_m) + aporte_total[i]
    valores_con_aportes.append(valor)

st.metric("Valor final con aportes", f"${valores_con_aportes[-1]:,.2f}")

# ============================================
# MÉTRICAS
# ============================================

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rendimiento Esperado", f"{mean_m*100:.2f}%")
col2.metric("Volatilidad", f"{std_m*100:.2f}%")
col3.metric("Valor final esperado (USD)", f"{valores[-1]:,.2f}")
col4.metric("Varianza", f"{var_m:.6f}")

# ============================================
# GRÁFICO
# ============================================

fig, ax = plt.subplots(figsize=(10,6))
ax.fill_between(tiempo/12, p5, p95, alpha=0.3, label="Banda 90% MC")
ax.plot(tiempo/12, valores, color='blue')

if moneda == "PEN":
    valores_min = valores * 0.95
    valores_max = valores * 1.05
    ax.fill_between(tiempo/12, valores_min, valores_max, color='orange', alpha=0.2, label="TC ±5%")

ax.set_title(f"Proyección de Capital - Portafolio {tipo_portafolio}")
ax.set_xlabel("Años")
ax.set_ylabel("Monto (USD)")
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)

# ============================================
# TABLA TC (PEN)
# ============================================

if moneda == "PEN":
    st.subheader("Sensibilidad al Tipo de Cambio (solo PEN)")

    df_tc = pd.DataFrame({
        "Escenario": ["TC -5%", "TC Actual", "TC +5%"],
        "Tipo de Cambio": [tipo_cambio*0.95, tipo_cambio, tipo_cambio*1.05],
        "Valor Final (USD)": [
            valores[-1] * 0.95,
            valores[-1],
            valores[-1] * 1.05
        ],
        "Valor Final (PEN)": [
            (valores[-1] * 0.95) * (tipo_cambio * 0.95),
            valores[-1] * tipo_cambio,
            (valores[-1] * 1.05) * (tipo_cambio * 1.05)
        ]
    })

    st.table(df_tc.style.format({
        "Tipo de Cambio": "{:.2f}",
        "Valor Final (USD)": "{:,.2f}",
        "Valor Final (PEN)": "{:,.2f}"
    }))

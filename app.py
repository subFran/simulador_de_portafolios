import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from datetime import datetime

# ===== CONFIGURACIÓN DE PÁGINA =====
st.set_page_config(page_title="Simulador de Portafolio", layout="wide")

# ===== ESTILO ELEGANTE (SIN CAMBIAR COLORES) =====
st.markdown(
    """
    <style>
    /* Variables simples para consistencia */
    :root{
      --card-radius: 12px;
      --card-padding: 18px;
      --container-max-w: 1100px;
      --sidebar-min-w: 300px;
      --shadow-1: 0 6px 18px rgba(0,0,0,0.08);
      --shadow-2: 0 10px 30px rgba(0,0,0,0.06);
    }

    /* Centrar el contenido principal y limitar ancho para lectura */
    [data-testid="stAppViewContainer"] {
      display: flex;
      justify-content: center;
    }
    .reportview-container .main .block-container {
      max-width: var(--container-max-w);
      padding-top: 2rem;
      padding-bottom: 2rem;
      padding-left: 1.5rem;
      padding-right: 1.5rem;
    }

    /* Barra lateral más espaciosa */
    [data-testid="stSidebar"] {
      min-width: var(--sidebar-min-w);
      padding-top: 2rem;
      padding-left: 1.1rem;
      padding-right: 1.1rem;
      padding-bottom: 2rem;
    }

    /* Tarjetas / cajas internas: fondo transparente por defecto, pero con padding, borde redondeado y sombra sutil */
    .css-1d391kg, .css-18e3th9, .css-1v0mbdj, .stAlert, .stExpander {
      border-radius: var(--card-radius) !important;
      padding: var(--card-padding) !important;
      box-shadow: var(--shadow-1) !important;
      background-clip: padding-box;
      transition: box-shadow 0.18s ease, transform 0.12s ease;
    }
    .css-1d391kg:hover, .css-18e3th9:hover, .css-1v0mbdj:hover {
      box-shadow: var(--shadow-2) !important;
    }

    /* Botones con radio y micro-animación (sin cambiar color) */
    .stButton>button, button[kind="primary"] {
      border-radius: 10px !important;
      padding: 8px 14px !important;
      border: 1px solid rgba(0,0,0,0.06) !important;
      box-shadow: none !important;
      transition: transform 0.09s ease, box-shadow 0.09s ease;
      font-weight: 600 !important;
    }
    .stButton>button:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }

    /* Inputs, selects y sliders: mayor separación y padding (sin alterar colores) */
    input, textarea, select {
      padding: 8px 10px !important;
      border-radius: 8px !important;
      border: 1px solid rgba(0,0,0,0.06) !important;
    }
    .stSlider>div div[role="slider"], .stSlider>div input {
      margin-top: 6px;
    }

    /* Métricas: mayor peso y espaciado */
    .stMetric > div {
      gap: 6px;
    }
    .stMetricValue {
      font-size: 1.25rem !important;
      font-weight: 700 !important;
    }
    .stMetricLabel {
      opacity: 0.95 !important;
      font-weight: 600 !important;
    }

    /* Tablas: mayor separación entre filas y bordes sutiles (sin colores nuevos) */
    table {
      border-collapse: separate !important;
      border-spacing: 0 6px !important;
    }
    thead th {
      font-weight: 700 !important;
    }
    tbody tr {
      border-radius: 8px;
    }

    /* Encabezado principal: un poco más compacto y elegante */
    h1 {
      letter-spacing: -0.6px;
      margin-bottom: 0.35rem;
      font-weight: 800;
    }

    /* Evitar alterar imágenes y gráficos */
    img, svg {
      filter: none !important;
    }

    /* Ajustes responsivos menores */
    @media (max-width: 900px) {
      .reportview-container .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
      }
      [data-testid="stSidebar"] { min-width: 220px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Simulador de Portafolio de Inversión")

# ===== 1. DEFINIR PORTAFOLIOS Y PESOS =====
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

# ===== 2. FUNCIÓN PARA DESCARGAR PRECIOS AJUSTADOS (CACHED) =====
@st.cache_data
def get_adj_close(tickers):
    start_date = '2022-12-01'
    end_date = '2025-12-01'

    # Descargar datos
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval='1mo',
        auto_adjust=True,
        progress=False
    )

    # Tomar la tabla de 'Close' y eliminar filas con NA (comportamiento tipo Excel)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            adj_close = data['Close'].dropna()
        else:
            adj_close = data.dropna()
    else:
        if 'Close' in data.columns:
            adj_close = data['Close'].dropna()
        else:
            adj_close = data.dropna()

    return adj_close

# ===== 3. FUNCIÓN PARA CALCULAR ESTADÍSTICAS DEL PORTAFOLIO =====
def portafolio_stats(tickers, pesos):

    adj_close = get_adj_close(tickers)
    if adj_close.empty:
        st.error("No se obtuvieron precios ajustados para los tickers solicitados.")
        raise ValueError("No hay datos históricos para los tickers.")

    # Usar solo filas completas (dropna any) para reproducir el comportamiento del notebook/Excel
    returns = adj_close.pct_change().dropna()
    if returns.empty:
        st.error("No se pudieron calcular rendimientos (pct_change) con los datos descargados.")
        raise ValueError("Rendimientos vacíos.")

    # Asegurarse de que las columnas estén en el orden de 'tickers'
    available_tickers = [t for t in tickers if t in returns.columns]
    if len(available_tickers) != len(tickers):
        st.warning(f"Algunos tickers faltan tras filtrar filas: usando {available_tickers}")

    # Re-alinear pesos según el orden de available_tickers
    weights_map = {t: float(pesos[i]) for i, t in enumerate(tickers)}
    pesos_disp = np.array([weights_map[t] for t in available_tickers], dtype=float)
    pesos_disp = pesos_disp / pesos_disp.sum()

    # Cálculos
    mean_mensual = returns[available_tickers].mean().dot(pesos_disp)
    cov_matrix = returns[available_tickers].cov()
    cov_vals = cov_matrix.loc[available_tickers, available_tickers].values

    port_var_mensual = float(pesos_disp.T @ cov_vals @ pesos_disp)
    port_std_mensual = np.sqrt(port_var_mensual)

    return mean_mensual, port_var_mensual, port_std_mensual

# ===== SIDEBAR - CONTROLES =====
st.sidebar.header("Parámetros de Simulación")

tipo_portafolio = st.sidebar.selectbox("Perfil de Riesgo", ['Bajo', 'Medio', 'Alto'])
monto_inicial = st.sidebar.number_input("Monto Inicial", min_value=1000, max_value=1000000, value=10000, step=1000)
anos = st.sidebar.slider("Horizonte de Inversión (Años)", 1, 10, 5)
moneda = st.sidebar.selectbox("Moneda", ['PEN', 'USD'])
tipo_cambio = st.sidebar.number_input("Tipo de Cambio (PEN/USD)", value=3.80, step=0.01)
tipo_tasa = st.sidebar.selectbox("Tipo de Tasa", ['Mensual histórica', 'Anual equivalente'])
simulaciones = st.sidebar.slider("Número de Simulaciones (Monte Carlo)", 100, 5000, 1000)

# ===== LÓGICA PRINCIPAL =====
tickers = portafolios[tipo_portafolio]['tickers']
pesos = portafolios[tipo_portafolio]['pesos']

# Mostrar composición del portafolio
st.sidebar.markdown("### Composición")
df_comp = pd.DataFrame({
    'Ticker': tickers,
    'Peso': [f"{p*100:.0f}%" for p in pesos]
})
st.sidebar.table(df_comp)

with st.spinner('Descargando datos y calculando...'):
    try:
        mean_mensual, var_mensual, std_mensual = portafolio_stats(tickers, pesos)

        # ===== Ajuste según tipo de tasa =====
        if tipo_tasa == 'Anual equivalente':
            mean = (1 + mean_mensual)**12 - 1
            std = std_mensual * np.sqrt(12)
            var = std**2
        else:
            mean = mean_mensual
            std = std_mensual
            var = var_mensual

        # ===== Ajuste monto inicial según moneda =====
        if moneda.upper() == 'USD':
            monto = monto_inicial / tipo_cambio
        else:
            monto = monto_inicial

        # ===== Valor acumulado determinístico =====
        meses = anos * 12
        tiempo = np.arange(0, meses + 1)
        valores = monto * (1 + mean)**tiempo

        # ===== Simulación Monte Carlo =====
        np.random.seed(42)
        simulaciones_array = np.zeros((simulaciones, meses+1))
        simulaciones_array[:,0] = monto

        for t in range(1, meses+1):
            r = np.random.normal(mean, std, simulaciones)
            simulaciones_array[:,t] = simulaciones_array[:,t-1] * (1 + r)

        # Percentiles
        valores_min_mc = np.percentile(simulaciones_array, 5, axis=0)
        valores_max_mc = np.percentile(simulaciones_array, 95, axis=0)

        # ===== Banda ±5% solo para USD =====
        if moneda.upper() == 'USD':
            valores_min = valores * 0.95
            valores_max = valores * 1.05
        else:
            valores_min = valores_min_mc
            valores_max = valores_max_mc

        # ===== VISUALIZACIÓN DE RESULTADOS =====
        # 1. Métricas Principales
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rendimiento Esperado", f"{mean*100:.2f}%")
        col2.metric("Volatilidad", f"{std*100:.2f}%")
        col3.metric("Valor Final Esperado", f"{valores[-1]:,.2f}")
        col4.metric("Varianza", f"{var:.6f}")

        # 2. Gráfico
        fig, ax = plt.subplots(figsize=(10, 6))

        # Banda MC
        ax.fill_between(tiempo/12, valores_min_mc, valores_max_mc, color='gray', alpha=0.3, label='Banda ±90% MC')

        # Proyección
        ax.plot(tiempo/12, valores, marker='o', linestyle='-', color='blue', label='Valor proyectado')

        # Banda TC (USD)
        if moneda.upper() == 'USD':
            ax.fill_between(tiempo/12, valores_min, valores_max, color='orange', alpha=0.2, label='Rango TC ±5%')

        ax.axhline(
            y=monto_inicial if moneda.upper()=='PEN' else monto_inicial/tipo_cambio,
            color='r', linestyle='--', label='Monto Inicial'
        )

        ax.set_title(f"Proyección de Capital - Portafolio {tipo_portafolio} ({moneda})")
        ax.set_xlabel("Años")
        ax.set_ylabel("Monto acumulado")
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)

        # 3. Tabla de Sensibilidad (Solo USD)
        if moneda.upper() == 'USD':
            st.subheader("Sensibilidad al Tipo de Cambio (USD)")
            df_tc = pd.DataFrame({
                'Escenario': ['TC -5%', 'TC Actual', 'TC +5%'],
                'Tipo de Cambio': [tipo_cambio*0.95, tipo_cambio, tipo_cambio*1.05],
                'Valor Final (USD)': [
                    monto * (1+mean)**meses * 0.95,
                    monto * (1+mean)**meses,
                    monto * (1+mean)**meses * 1.05
                ]
            })
            st.table(df_tc.style.format({
                'Tipo de Cambio': '{:.2f}',
                'Valor Final (USD)': '{:,.2f}'
            }))

    except Exception as e:
        st.error(f"Ocurrió un error en el cálculo: {str(e)}")
        st.info("Intenta recargar la página o verificar tu conexión a internet para descargar los datos de Yahoo Finance.")

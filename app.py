import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from datetime import datetime

# ===== CONFIGURACIÓN DE PÁGINA =====
st.set_page_config(page_title="Simulador de Portafolio", layout="wide")

st.title("📊 Simulador de Portafolio de Inversión")

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
    # Manejar estructura de columnas de yfinance (MultiIndex o SingleIndex)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            adj_close = data['Close']
        else:
            # Fallback si la estructura es diferente
            adj_close = data
    else:
        if 'Close' in data.columns:
            adj_close = data['Close']
        else:
            adj_close = data
            
    return adj_close.dropna()

# ===== 3. FUNCIÓN PARA CALCULAR ESTADÍSTICAS DEL PORTAFOLIO =====
def portafolio_stats(tickers, pesos):
    adj_close = get_adj_close(tickers)
    returns = adj_close.pct_change().dropna()
    
    # Asegurar que los pesos coincidan con las columnas disponibles
    # A veces yfinance puede fallar en descargar algun ticker
    available_tickers = returns.columns.intersection(tickers)
    
    if len(available_tickers) != len(tickers):
        st.warning(f"Algunos tickers no se pudieron descargar. Usando: {available_tickers.tolist()}")
        # Re-normalizar pesos si faltan tickers (simplificación para evitar crash)
        indices = [tickers.index(t) for t in available_tickers]
        pesos = pesos[indices]
        pesos = pesos / pesos.sum()
        
    mean_mensual = returns[available_tickers].mean().dot(pesos)
    cov_matrix = returns[available_tickers].cov()
    port_var_mensual = pesos.T @ cov_matrix @ pesos
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
            
        ax.axhline(y=monto_inicial if moneda.upper()=='PEN' else monto_inicial/tipo_cambio, color='r', linestyle='--', label='Monto Inicial')
        
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

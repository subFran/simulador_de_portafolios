import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from datetime import datetime

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

st.title("Simulador de Portafolios — (Monte Carlo + Proyección)")

# ========================
#     PORTAFOLIOS PREDEFINIDOS
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
# SIDEBAR: CONTROLES (basados en tu notebook original)
# ------------------------
st.sidebar.header("Configuración")

selected_portfolio = st.sidebar.selectbox('Tipo de portafolio', list(portafolios.keys()))
tickers = portafolios[selected_portfolio]['tickers']
preset_weights = portafolios[selected_portfolio]['pesos']

# Opción para editar o bloquear pesos (respeta tu comportamiento anterior)
edit_weights = st.sidebar.checkbox("Editar pesos manualmente", value=False, help="Si está desactivado se usan los pesos predefinidos y no se pueden editar.")
weights = {}
if edit_weights:
    st.sidebar.markdown("Pesos (sumar 1.0)")
    for i, tk in enumerate(tickers):
        weights[tk] = st.sidebar.number_input(f"{tk}", min_value=0.0, max_value=1.0, value=float(preset_weights[i]), step=0.01, format="%.2f")
else:
    st.sidebar.markdown("Pesos (predefinidos)")
    for i, tk in enumerate(tickers):
        st.sidebar.write(f"{tk}: {preset_weights[i]:.2f}")
    weights = {tickers[i]: float(preset_weights[i]) for i in range(len(tickers))}

# Moneda y tipo de cambio (como pediste)
st.sidebar.markdown("---")
currency = st.sidebar.selectbox("Moneda de visualización", ["PEN", "USD"])
exchange_rate = None
if currency == "PEN":
    exchange_rate = st.sidebar.number_input("Tipo de cambio (PEN por USD)", min_value=0.0, value=3.8, step=0.01)

# Monto inicial y contribuciones (separados)
st.sidebar.markdown("---")
initial_capital = st.sidebar.number_input("Monto inicial (en la moneda seleccionada)", min_value=0.0, value=10000.0, step=100.0)

st.sidebar.markdown("---")
st.sidebar.header("Contribuciones")
contrib_bool = st.sidebar.radio("¿Agregar contribuciones periódicas?", ["No", "Sí"])
monthly_contrib = 0.0
if contrib_bool == "Sí":
    monthly_contrib = st.sidebar.number_input("Monto de contribución (mensual, en la moneda seleccionada)", min_value=0.0, value=0.0, step=10.0)

# Periodo y fechas
st.sidebar.markdown("---")
years = st.sidebar.slider("Horizonte (años)", min_value=1, max_value=30, value=5)
start_date = st.sidebar.date_input("Fecha de inicio para datos (ej: 2015-01-01)", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("Fecha final para datos (ej: hoy)", value=pd.to_datetime(datetime.today().date()))

# Tipo de tasa y simulaciones (como tu notebook)
st.sidebar.markdown("---")
tipo_tasa = st.sidebar.selectbox("Tipo de tasa", ['Mensual histórica', 'Anual equivalente'])
simulations = st.sidebar.number_input("Número de simulaciones (Monte Carlo)", min_value=100, max_value=20000, value=1000, step=100)

# Botón de correr
run_btn = st.sidebar.button("Ejecutar simulación")

# ========================
#    FUNCIONES AUXILIARES
# ========================
@st.cache_data
def get_adj_close(tickers, start, end):
    """Descarga precios ajustados mensuales usando yfinance (Close con interval 1mo)."""
    # yfinance returns multiindex for multiple tickers; request 'Close' then dropna
    data = yf.download(tickers, start=start, end=end, interval='1mo', auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            adj = data['Close']
        else:
            # try 'Adj Close' fallback
            if 'Adj Close' in data.columns.get_level_values(0):
                adj = data['Adj Close']
            else:
                adj = data.dropna(axis=1, how='all')
    else:
        if 'Close' in data.columns:
            adj = data['Close']
        elif 'Adj Close' in data.columns:
            adj = data['Adj Close']
        else:
            adj = data
    if isinstance(adj, pd.Series):
        adj = adj.to_frame(name=tickers[0])
    return adj.dropna(how='all')

def compute_portfolio_stats(adj_close, pesos):
    """Calcula mean y std mensual del portafolio (basado en retornos históricos)."""
    returns = adj_close.pct_change().dropna()
    if returns.empty:
        return None, None, None, None
    # Alinear columnas con pesos
    weights_series = pd.Series(pesos, index=adj_close.columns).reindex(columns=adj_close.columns, fill_value=0.0)
    portfolio_returns = returns.mul(weights_series, axis=1).sum(axis=1)
    mean_mensual = portfolio_returns.mean()
    std_mensual = portfolio_returns.std()
    var_mensual = portfolio_returns.var()
    return mean_mensual, std_mensual, var_mensual, portfolio_returns

# ========================
#    LÓGICA AL PRESIONAR "Ejecutar simulación"
# ========================
if run_btn:
    # Validaciones de pesos
    weights_series = pd.Series(weights)
    if abs(weights_series.sum() - 1.0) > 1e-6:
        st.error("Los pesos deben sumar 1.0. Ajusta los pesos o usa los predefinidos.")
    else:
        st.info("Descargando datos y calculando estadísticas...")
        adj = get_adj_close(tickers, start_date, end_date)
        if adj.empty:
            st.error("No hay datos para los tickers y rango seleccionado.")
        else:
            # Si algunos tickers no están disponibles, ajustar
            available = [c for c in adj.columns if c in weights_series.index and weights_series[c] > 0]
            if len(available) != len(weights_series):
                missing = set(weights_series.index) - set(available)
                if missing:
                    st.warning(f"Se excluirán estos tickers sin datos: {missing}")
                # keep only available and re-normalize
                weights_series = weights_series.loc[[c for c in available if c in weights_series.index]]
                if weights_series.sum() == 0:
                    st.error("No hay pesos válidos tras excluir tickers sin datos.")
                    st.stop()
                weights_series = weights_series / weights_series.sum()
                adj = adj[weights_series.index]

            mean_m, std_m, var_m, portfolio_returns = compute_portfolio_stats(adj, weights_series.values)
            if mean_m is None:
                st.error("No se pudo calcular estadísticas con los datos descargados.")
            else:
                # Ajuste de tasa si el usuario pidió anual equivalente
                if tipo_tasa == 'Anual equivalente':
                    mean = (1 + mean_m)**12 - 1
                    std = std_m * np.sqrt(12)
                    # convert back to monthly equivalents for the simulation if needed:
                    # We will keep simulation in monthly space: convert annual mean/std to monthly:
                    mean_monthly_for_sim = (1 + mean)**(1/12) - 1
                    std_monthly_for_sim = std / np.sqrt(12)
                else:
                    mean_monthly_for_sim = mean_m
                    std_monthly_for_sim = std_m

                # Moneda: convertimos monto inicial y contribución a USD internamente porque los tickers están en USD
                if currency == "PEN":
                    if not exchange_rate or exchange_rate <= 0:
                        st.error("Ingresa un tipo de cambio válido mayor a 0.")
                        st.stop()
                    initial_usd = float(initial_capital) / float(exchange_rate)
                    contrib_usd = float(monthly_contrib) / float(exchange_rate) if monthly_contrib else 0.0
                else:
                    initial_usd = float(initial_capital)
                    contrib_usd = float(monthly_contrib) if monthly_contrib else 0.0

                months = int(years * 12)
                # Proyección determinística con contribuciones mensuales añadidas al final de cada periodo
                tiempo = np.arange(0, months + 1)
                deterministico = np.zeros(months + 1)
                deterministico[0] = initial_usd
                for t in range(1, months + 1):
                    deterministico[t] = deterministico[t - 1] * (1 + mean_monthly_for_sim) + contrib_usd

                # Monte Carlo
                np.random.seed(42)
                sims = np.zeros((simulations, months + 1))
                sims[:, 0] = initial_usd
                for t in range(1, months + 1):
                    # generar rendimientos mensuales
                    rs = np.random.normal(loc=mean_monthly_for_sim, scale=std_monthly_for_sim, size=simulations)
                    sims[:, t] = sims[:, t - 1] * (1 + rs)
                    if contrib_usd > 0:
                        sims[:, t] += contrib_usd

                p5 = np.percentile(sims, 5, axis=0)
                p50 = np.percentile(sims, 50, axis=0)
                p95 = np.percentile(sims, 95, axis=0)

                # Convertir de USD a moneda seleccionada para mostrar
                if currency == "PEN":
                    deterministico_display = deterministico * exchange_rate
                    p5_disp = p5 * exchange_rate
                    p50_disp = p50 * exchange_rate
                    p95_disp = p95 * exchange_rate
                    axis_label = f"Monto ({currency})"
                    initial_display = initial_usd * exchange_rate
                else:
                    deterministico_display = deterministico.copy()
                    p5_disp = p5.copy()
                    p50_disp = p50.copy()
                    p95_disp = p95.copy()
                    axis_label = f"Monto ({currency})"
                    initial_display = initial_usd

                # ===== Mostrar resultados =====
                st.subheader("Estadísticas históricas (mensual)")
                st.write(pd.DataFrame({
                    "Mean mensual": [f"{mean_m:.4f}"],
                    "Std mensual": [f"{std_m:.4f}"],
                    "Var mensual": [f"{var_m:.6f}"]
                }, index=[f"Portafolio {selected_portfolio}"]))

                st.subheader("Proyección y Monte Carlo")
                fig, ax = plt.subplots(figsize=(10, 5))
                years_axis = tiempo / 12
                # Banda MC 5-95%
                ax.fill_between(years_axis, p5_disp, p95_disp, color="gray", alpha=0.25, label="Banda MC 5-95%")
                # Mediana MC
                ax.plot(years_axis, p50_disp, color="#1f77b4", linestyle='--', label="Mediana MC")
                # Determinístico
                ax.plot(years_axis, deterministico_display, color="#00FF7F", marker='o', linewidth=2, label="Proyección determinística")
                # Inicial
                ax.axhline(y=initial_display, color='r', linestyle='--', label="Monto Inicial")
                ax.set_title(f"Proyección de Capital - Portafolio {selected_portfolio} ({currency})")
                ax.set_xlabel("Años")
                ax.set_ylabel(axis_label)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend()
                ax.set_facecolor("#000000")
                fig.patch.set_facecolor("#060606")
                ax.tick_params(colors="#00FF7F")
                for spine in ax.spines.values():
                    spine.set_color("#00FF7F")
                st.pyplot(fig, use_container_width=True)

                # Tabla resumen
                final_det = deterministico_display[-1]
                final_p5 = p5_disp[-1]
                final_p50 = p50_disp[-1]
                final_p95 = p95_disp[-1]

                st.subheader("Resumen final esperado")
                df_res = pd.DataFrame({
                    "Escenario": ["Determinístico (mean)", "MC 5%", "MC 50% (mediana)", "MC 95%"],
                    "Valor final": [f"{final_det:,.2f}", f"{final_p5:,.2f}", f"{final_p50:,.2f}", f"{final_p95:,.2f}"]
                })
                st.write(df_res.set_index("Escenario"))

                # Sensibilidad tipo de cambio (si aplica)
                if currency == "PEN":
                    st.subheader("Sensibilidad al tipo de cambio (±5%)")
                    tc_down = exchange_rate * 0.95
                    tc_up = exchange_rate * 1.05
                    df_tc = pd.DataFrame({
                        'Tipo de cambio': [tc_down, exchange_rate, tc_up],
                        f'Valor final ({currency})': [final_det * 0.95, final_det, final_det * 1.05]
                    })
                    st.write(df_tc)

                # Mostrar últimos valores proyectados (tabla)
                st.subheader("Valores proyectados (últimos 5 puntos)")
                df_vals = pd.DataFrame({
                    "Año": np.round(years_axis, 2),
                    "Determinístico": deterministico_display,
                    "MC P5": p5_disp,
                    "MC P50": p50_disp,
                    "MC P95": p95_disp
                })
                st.write(df_vals.tail(5).set_index("Año"))

                st.success("Simulación completada.")

else:
    st.info("Ajusta los parámetros en la barra lateral y presiona 'Ejecutar simulación' para ver proyecciones y Monte Carlo.")

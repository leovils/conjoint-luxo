import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from conjoint_engine import ConjointEngine

st.set_page_config(page_title="Painel Analítico Avançado - Conjoint", layout="wide", page_icon="📊")

st.title("📊 Painel Avançado de Análise Conjoint")
st.markdown("Neste laboratório, você explora as decisões matemáticas dos respondentes. Filtre demograficamente, avalie utilidades zero-centradas e simule novos lançamentos no mercado.")

CONFIG_FILE = "survey_config.json"
if not os.path.exists(CONFIG_FILE):
    st.error("O arquivo 'survey_config.json' não foi encontrado na pasta. Por favor, crie e trave a sua pesquisa no 'app.py' primeiro.")
    st.stop()

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = json.load(f)
    attributes = cfg.get("attributes", {})
    forbidden = cfg.get("forbidden", [])
    
st.success(f"Configuração Base Identificada: {len(attributes)} atributos em jogo.")

# SIDEBAR: Upload & Filtros
st.sidebar.header("📁 Dados e Segmentação")
uploaded_file = st.sidebar.file_uploader("Upload do CSV (Nuvem)", type=["csv"])

if uploaded_file is None:
    st.info("👈 Faça o upload do seu `.csv` gerado pelo Supabase ou Google Sheets na barra lateral para destravar os relatórios!")
    st.stop()

try:
    df_cloud = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python')
except UnicodeDecodeError:
    uploaded_file.seek(0)
    df_cloud = pd.read_csv(uploaded_file, encoding='latin1', sep=None, engine='python')

# Dynamic Demographic Filters
st.sidebar.markdown("---")
st.sidebar.subheader("👥 Filtros Demográficos")
demo_cols = [c for c in df_cloud.columns if c.startswith("Perfil_")]

filtered_df = df_cloud.copy()

if demo_cols:
    st.sidebar.caption("Filtre o relatório selecionando grupos específicos. O modelo recalculará instantaneamente.")
    for col in demo_cols:
        unique_vals = df_cloud[col].dropna().unique().tolist()
        if unique_vals:
            # Shorten column name for the UI label
            lbl = col.replace("Perfil_", "")
            selected_vals = st.sidebar.multiselect(f"{lbl}", options=unique_vals, default=unique_vals)
            if selected_vals:
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

st.sidebar.markdown("---")
st.sidebar.metric("Linhas (Respostas) Ativas no Modelo", len(filtered_df))

if len(filtered_df) < 5:
    st.warning("Poucas rodadas encontradas para esse corte demográfico. Resultados podem ter alta variância.")

# ----------------- REGRESSION ENGINE REBUILD ----------------- #
engine = ConjointEngine(attributes, forbidden)
history_rebuilt = []

for _, row in filtered_df.iterrows():
    raw_A = {}
    raw_B = {}
    for attr in attributes.keys():
        colA = f"OpA_{attr}"
        colB = f"OpB_{attr}"
        if colA in row and colB in row:
            raw_A[attr] = str(row[colA])
            raw_B[attr] = str(row[colB])
            
    if raw_A and raw_B and "choice" in row:
        choice = str(row["choice"]).strip()
        diff = engine._encode_profile(raw_A) - engine._encode_profile(raw_B)
        history_rebuilt.append({
            'raw_A': raw_A,
            'raw_B': raw_B,
            'diff_vector': diff,
            'choice_A': 1 if choice == 'A' else 0
        })

engine.history = history_rebuilt
engine.betas = engine._calculate_betas()

# DATA PREPARATION FOR DASHBOARDS
imp_df = engine.get_importance_df()
util_df = engine.get_utilities_df()

# ----------------- TABS REPORTING ----------------- #
tab1, tab2, tab3 = st.tabs(["📌 Importância Relativa", "⚖️ Utilidades (Part-Worths)", "🔮 Simulador de Mercado"])

with tab1:
    st.header("1. Qual Atributo pesa mais na decisão?")
    st.markdown("A **Importância Relativa** traduz o quanto a diferença entre a 'melhor' e a 'pior' opção de um atributo representa no peso total da decisão.")
    
    fig_imp = px.pie(imp_df, values="Relative Importance (%)", names="Atributo", 
                     title="Importância Relativa dos Atributos (%)", hole=0.4,
                     color_discrete_sequence=px.colors.sequential.Teal)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.dataframe(imp_df, use_container_width=True)

with tab2:
    st.header("2. Qual o valor psicológico de cada Nível?")
    st.markdown("As **Utilidades Zero-Centradas** mostram se um nível agrega (+, verde) ou destrói (-, vermelho) a atratividade do produto frente à média (0).")
    
    for attr in attributes.keys():
        sub_df = util_df[util_df["Atributo"] == attr].copy()
        
        # Plotly Bar Chart centered at 0
        sub_df["Cor"] = np.where(sub_df["Utilidade Zero-Centrada"] > 0, "#2ecc71", "#e74c3c")
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=sub_df["Nível"],
            y=sub_df["Utilidade Zero-Centrada"],
            marker_color=sub_df["Cor"],
            text=np.round(sub_df["Utilidade Zero-Centrada"], 2),
            textposition='auto'
        ))
        
        fig_bar.update_layout(
            title=f"Utilidades: {attr}",
            yaxis_title="Part-Worth Utility",
            xaxis_title="Níveis do Atributo",
            shapes=[dict(type='line', y0=0, y1=0, x0=-0.5, x1=len(sub_df)-0.5, line=dict(color='black', width=1))],
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.header("3. Simulador de Preferência de Mercado")
    st.markdown("Crie produtos hipotéticos e jogue-os na arena. O algoritmo Logit dirá o **Market Share** exato de cada um.")
    
    num_prods = st.slider("Qtd. de Concorrentes na Arena", min_value=2, max_value=8, value=3)
    
    # Gerar cols pros produtos
    prod_cols = st.columns(num_prods)
    sim_profiles = []
    
    for i in range(num_prods):
        with prod_cols[i]:
            st.subheader(f"Produto {i+1}")
            p_config = {}
            for attr, levels in attributes.items():
                p_config[attr] = st.selectbox(f"{attr} ##{i}", levels, key=f"sim_{attr}_{i}")
            sim_profiles.append(p_config)
            
    st.markdown("---")
    if st.button("Lutar! Calcular Market Share", type="primary"):
        shares = engine.simulate_market_share_n(sim_profiles)
        
        share_df = pd.DataFrame({
            "Produto": [f"Produto {i+1}" for i in range(num_prods)],
            "Market Share (%)": [round(s * 100, 2) for s in shares]
        })
        
        colChat, colData = st.columns([2, 1])
        with colChat:
            fig_share = px.bar(share_df, x="Produto", y="Market Share (%)", color="Produto", text="Market Share (%)", title="Market Share Simulado (%)")
            fig_share.update_traces(textposition='outside')
            st.plotly_chart(fig_share, use_container_width=True)
            
        with colData:
            st.write("Configurações Deste Cenário:")
            # Display configuration table
            conf_df = pd.DataFrame(sim_profiles)
            conf_df.index = [f"Produto {i+1}" for i in range(num_prods)]
            st.dataframe(conf_df.T, use_container_width=True)

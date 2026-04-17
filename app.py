import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import os
import requests
import uuid
import threading
from datetime import datetime

from conjoint_engine import ConjointEngine

st.set_page_config(page_title="Pesquisa - Conjoint Analysis", layout="wide")

CONFIG_FILE = "survey_config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

if "config" not in st.session_state:
    st.session_state.config = load_config()

respondent_mode = st.session_state.config is not None

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

if "profiling_completed" not in st.session_state:
    st.session_state.profiling_completed = False

if "profile_answers" not in st.session_state:
    st.session_state.profile_answers = {}

if "engine" not in st.session_state:
    if respondent_mode:
        cfg = st.session_state.config
        st.session_state.engine = ConjointEngine(cfg["attributes"], cfg["forbidden"])
        st.session_state.intro_text = cfg.get("intro_text", "")
        st.session_state.scenario_text = cfg.get("scenario_text", "")
        st.session_state.webhook_url = cfg.get("webhook_url", "")
        st.session_state.supabase_url = cfg.get("supabase_url", "")
        st.session_state.supabase_key = cfg.get("supabase_key", "")
        st.session_state.supabase_table = cfg.get("supabase_table", "conjoint_responses")
        if not cfg.get("profile_questions"):
            st.session_state.profiling_completed = True
    else:
        st.session_state.engine = None
        st.session_state.intro_text = "Bem-vindo à pesquisa! Por favor, escolha a opção que mais prefere."
        st.session_state.scenario_text = ""
        st.session_state.webhook_url = ""
        st.session_state.supabase_url = ""
        st.session_state.supabase_key = ""
        st.session_state.supabase_table = "conjoint_responses"
        st.session_state.attributes = {}
        st.session_state.forbidden = []

if "setup_done" not in st.session_state:
    st.session_state.setup_done = respondent_mode

if "current_pair" not in st.session_state:
    if st.session_state.setup_done and respondent_mode:
        st.session_state.current_pair = st.session_state.engine.generate_pair()
    else:
        st.session_state.current_pair = None

if "survey_finished" not in st.session_state:
    st.session_state.survey_finished = False

def _bg_post(webhook_url, sup_url, sup_key, sup_table, payload):
    if webhook_url:
        try: requests.post(webhook_url, json=payload, timeout=15)
        except: pass
    if sup_url and sup_key:
        try:
            from supabase import create_client
            supabase = create_client(sup_url, sup_key)
            supabase.table(sup_table).insert(payload).execute()
        except: pass

def send_to_webhook(pair, chosen):
    webhook = st.session_state.get("webhook_url", "")
    sup_url = st.session_state.get("supabase_url", "")
    sup_key = st.session_state.get("supabase_key", "")
    sup_table = st.session_state.get("supabase_table", "conjoint_responses")
    
    if not webhook and not (sup_url and sup_key): return
    
    payload = {
        "user_id": st.session_state.user_id,
        "timestamp": datetime.now().isoformat(),
        "round": len(st.session_state.engine.history),
        "choice": chosen
    }
    for k, v in pair['A'].items(): payload[f"OpA_{k}"] = v
    for k, v in pair['B'].items(): payload[f"OpB_{k}"] = v
    for q, a in st.session_state.profile_answers.items(): payload[f"Perfil_{q}"] = a
        
    threading.Thread(target=_bg_post, args=(webhook, sup_url, sup_key, sup_table, payload), daemon=True).start()
    st.session_state.webhook_error = None

st.title("Conjoint Analysis - Pesquisa Interativa")

def render_profile_form():
    st.write(st.session_state.intro_text)
    cfg = st.session_state.config
    questions = cfg.get("profile_questions", [])
    st.subheader("Sobre você")
    with st.form("profiling_form"):
        answers = {}
        for idx, q in enumerate(questions):
            answers[q["question"]] = st.selectbox(f"**{q['question']}**", q["options"], key=f"ans_{idx}")
        submit = st.form_submit_button("Continuar para a Pesquisa")
        if submit:
            st.session_state.profile_answers = answers
            st.session_state.profiling_completed = True
            st.rerun()

def render_survey():
    if not st.session_state.setup_done:
        st.info("Por favor, conclua a configuração (aba 1).")
        return
    if st.session_state.survey_finished:
        st.balloons()
        st.success("A pesquisa foi concluida! Processamos as suas preferencias em tempo real.")
        
        try:
            imp_df = st.session_state.engine.get_importance_df()
            util_df = st.session_state.engine.get_utilities_df()
            if len(imp_df) > 0:
                if imp_df["Range Absoluto"].sum() == 0.0:
                    st.warning("🤖 **Aviso do Sistema:** O algoritmo detectou um padrão linear excessivo (ex: selecionou sempre a mesma posição ou não variou escolhas). Não foi possível mapear matematicamente o seu padrão de conflito real. Obrigado pela participação!")
                else:
                    st.markdown("### 🧠 O Seu Mapa de Decisao Pessoal")
                    import plotly.express as px
                    fig = px.bar(
                        imp_df.sort_values(by="Relative Importance (%)", ascending=True), 
                        x="Relative Importance (%)", y="Atributo", orientation='h', 
                        title="O que mais peso teve nas suas escolhas?", color="Atributo"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    from pdf_generator import create_user_report
                    pdf_bytes = create_user_report(imp_df, util_df)
                    
                    st.download_button(
                        label="📥 Baixar Meu Relatorio Tecnico (PDF)",
                        data=pdf_bytes,
                        file_name="meu_perfil_conjoint.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
        except Exception as e:
            st.error(f"Seus dados foram salvos. Erro visual: {e}")
            
        if st.session_state.get("webhook_error"):
            st.error("Aviso de salvamento: " + st.session_state.webhook_error)
        return
        
    if st.session_state.get("webhook_error"):
        st.error("Erro da Rodada Anterior: " + st.session_state.webhook_error)
        
    cfg = st.session_state.config or {}
    scen_text = st.session_state.get("scenario_text", "")
    if scen_text and scen_text.strip():
        st.info(f"💡 **Atenção ao Cenário:** {scen_text}")
    elif not cfg.get("profile_questions"): 
        st.write(st.session_state.intro_text)
    else: 
        st.write("Avalie as opções e escolha a que você mais prefere em cada par:")
    
    history_len = len(st.session_state.engine.history)
    st.progress(min(history_len / 18.0, 1.0))
    st.caption(f"Rodada: {history_len + 1} (Mínimo: 10 pares)")
    
    pair = st.session_state.current_pair
    if pair:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Opção A")
            for k, v in pair['A'].items(): st.write(f"**{k}:** {v}")
            if st.button("Escolher Opção A ✅", use_container_width=True):
                st.session_state.survey_finished = st.session_state.engine.register_choice(pair, 'A')
                send_to_webhook(pair, 'A')
                st.session_state.current_pair = st.session_state.engine.generate_pair()
                st.rerun()
        with colB:
            st.subheader("Opção B")
            for k, v in pair['B'].items(): st.write(f"**{k}:** {v}")
            if st.button("Escolher Opção B ✅", use_container_width=True):
                st.session_state.survey_finished = st.session_state.engine.register_choice(pair, 'B')
                send_to_webhook(pair, 'B')
                st.session_state.current_pair = st.session_state.engine.generate_pair()
                st.rerun()

if respondent_mode:
    if not st.session_state.profiling_completed: render_profile_form()
    else: render_survey()
else:
    tab_config, tab_survey, tab_report = st.tabs(["1. Configuração", "2. Visualizar/Coletar", "3. Relatórios Experimentais"])
    
    with tab_config:
        st.header("Upload Mágico (Excel)")
        st.write("Tem uma configuração pronta no Excel? Arraste seu `.xlsx` (com Aba1, Aba2 e Aba3) aqui:")
        uploaded_xls = st.file_uploader("Upload do Template (.xlsx)", type=["xlsx"])
        if uploaded_xls:
            try:
                xls = pd.ExcelFile(uploaded_xls)
                if "Aba1" in xls.sheet_names:
                    df1 = pd.read_excel(xls, "Aba1")
                    for _, row in df1.iterrows():
                        k = str(row.iloc[0]).strip()
                        v = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
                        if k == "Webhook URL": st.session_state.webhook_url = v
                        elif k == "Supabase URL": st.session_state.supabase_url = v
                        elif k == "Supabase API Key": st.session_state.supabase_key = v
                        elif k == "Supabase Table": st.session_state.supabase_table = v
                        elif k == "Texto Convite": st.session_state.intro_text = v
                        elif k == "Texto Cenário": st.session_state.scenario_text = v
                        
                if "Aba2" in xls.sheet_names:
                    df2 = pd.read_excel(xls, "Aba2")
                    valid_attrs = df2.dropna(subset=[df2.columns[0]])
                    st.session_state.ui_num_attrs = len(valid_attrs)
                    for i, row in valid_attrs.reset_index(drop=True).iterrows():
                        st.session_state[f"attr_name_{i}"] = str(row.iloc[0]).strip()
                        levels_list = [str(row.iloc[c]).strip() for c in range(1, len(row)) if pd.notna(row.iloc[c]) and str(row.iloc[c]).strip()]
                        st.session_state[f"lvl_{i}"] = ",".join(levels_list)
                        
                if "Aba3" in xls.sheet_names:
                    df3 = pd.read_excel(xls, "Aba3")
                    valid_profs = df3.dropna(subset=[df3.columns[0]])
                    st.session_state.ui_num_profile = len(valid_profs)
                    for i, row in valid_profs.reset_index(drop=True).iterrows():
                        st.session_state[f"pq_{i}"] = str(row.iloc[0]).strip()
                        st.session_state[f"po_{i}"] = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else "Opção 1"
                        
                st.success("🎉 Planilha importada com sucesso! Dê F5 para ver tudo na tela.")
            except Exception as e:
                st.error(f"Erro Excel: {e}")

        st.markdown("---")
        st.header("1. Banco de Dados e Textos")
        st.session_state.intro_text = st.text_area("Texto de Convite (Aba1)", value=st.session_state.intro_text)
        st.session_state.scenario_text = st.text_area("Texto Cenário das Escolhas (Aba1)", value=st.session_state.get("scenario_text", ""))
        st.session_state.webhook_url = st.text_input("Webhook URL (Google Sheets)", value=st.session_state.webhook_url)
        st.session_state.supabase_url = st.text_input("Supabase Project URL", value=st.session_state.supabase_url)
        st.session_state.supabase_key = st.text_input("Supabase API Key (Anon)", value=st.session_state.supabase_key, type="password")
        st.session_state.supabase_table = st.text_input("Nome da Tabela no Supabase (ex: conjoint_responses)", value=st.session_state.supabase_table)
        
        st.markdown("---")
        st.header("2. Perguntas de Perfil Sócio-Demográficas (Aba3)")
        num_profile = st.number_input("Número de perguntas", min_value=0, max_value=20, value=st.session_state.get("ui_num_profile", 0))
        profile_config = []
        if num_profile > 0:
            for i in range(num_profile):
                st.markdown(f"**Pergunta {i+1}**")
                colQ, colO = st.columns(2)
                with colQ:
                    if f"pq_{i}" not in st.session_state: st.session_state[f"pq_{i}"] = f"Sua Pergunta {i+1}?"
                    q_text = st.text_input("Texto", key=f"pq_{i}")
                with colO:
                    if f"po_{i}" not in st.session_state: st.session_state[f"po_{i}"] = "Opção 1,Opção 2"
                    q_opts = st.text_input("Opções (separadas por virgula)", key=f"po_{i}")
                if q_text: profile_config.append({"question": q_text, "options": [opt.strip() for opt in q_opts.split(",") if opt.strip()]})
        
        st.markdown("---")
        st.header("3. Atributos Formadores (Aba2)")
        num_attrs = st.number_input("Quantidade de Atributos", min_value=2, max_value=25, value=st.session_state.get("ui_num_attrs", 5))
        attr_inputs = {}
        for i in range(num_attrs):
            colA1, colA2 = st.columns(2)
            with colA1:
                if f"attr_name_{i}" not in st.session_state: st.session_state[f"attr_name_{i}"] = f"Atributo {i+1}"
                attr_name = st.text_input(f"Atributo {i+1}", key=f"attr_name_{i}")
            with colA2:
                if f"lvl_{i}" not in st.session_state: st.session_state[f"lvl_{i}"] = "Nível 1,Nível 2,Nível 3"
                levels_str = st.text_input("Níveis (separados por vírgula)", key=f"lvl_{i}")
            attr_inputs[attr_name] = [l.strip() for l in levels_str.split(",") if l.strip()]

        st.markdown("---")
        st.header("4. Combinações Proibidas")
        all_levels = [f"{k}: {lvl}" for k, v in attr_inputs.items() for lvl in v]
        forbidden_pairs = st.multiselect("Pares proibidos", options=[f"{a} + {b}" for i, a in enumerate(all_levels) for b in all_levels[i+1:]])

        st.markdown("---")
        colSave1, colSave2 = st.columns(2)
        with colSave1:
            if st.button("Aplicar Configurações (Teste)"):
                st.session_state.attributes = attr_inputs
                st.session_state.forbidden = forbidden_pairs
                st.session_state.engine = ConjointEngine(attr_inputs, forbidden_pairs)
                st.session_state.setup_done = True
                
                st.session_state.config = {
                    "intro_text": st.session_state.intro_text,
                    "scenario_text": st.session_state.scenario_text,
                    "profile_questions": profile_config,
                    "attributes": attr_inputs, "forbidden": forbidden_pairs
                }
                st.session_state.profiling_completed = not profile_config
                st.session_state.profile_answers = {}
                st.session_state.current_pair = st.session_state.engine.generate_pair()
                st.session_state.survey_finished = False
                st.success("Salvo para teste na Aba 2.")
        
        with colSave2:
            if st.button("Travar Configuração & Preparar para Nuvem"):
                valid = True
                for k, v in attr_inputs.items():
                    if len(v) < 2:
                        st.error(f"'{k}' precisa ter no mínimo 2 níveis.")
                        valid = False
                if valid:
                    config_data = {
                        "intro_text": st.session_state.intro_text,
                        "scenario_text": st.session_state.scenario_text,
                        "webhook_url": st.session_state.webhook_url,
                        "supabase_url": st.session_state.supabase_url,
                        "supabase_key": st.session_state.supabase_key,
                        "supabase_table": st.session_state.supabase_table,
                        "profile_questions": profile_config,
                        "attributes": attr_inputs,
                        "forbidden": forbidden_pairs
                    }
                    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, ensure_ascii=False, indent=4)
                    st.session_state.config = config_data
                    st.session_state.engine = ConjointEngine(attr_inputs, forbidden_pairs)
                    st.session_state.setup_done = True
                    st.session_state.profiling_completed = not profile_config
                    st.session_state.profile_answers = {}
                    st.session_state.current_pair = st.session_state.engine.generate_pair()
                    st.session_state.survey_finished = False
                    st.rerun()

    with tab_survey:
        if not st.session_state.profiling_completed and st.session_state.config and st.session_state.config.get("profile_questions"):
            render_profile_form()
        else: render_survey()
        
    with tab_report:
        st.header("Análise Consolidada (Via Analise.py)")
        st.info("Por favor, use o aplicativo independente Iniciar_Analise.bat para calcular a matemática usando o CSV!")
        
        st.markdown("---")
        st.header("Teste Local Oculto")
        if not st.session_state.setup_done: st.info("Teste na Aba 2 antes.")
        elif len(st.session_state.engine.history) == 0: st.info("Nenhuma escolha local.")
        else:
            st.dataframe(st.session_state.engine.get_history_df())

import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
import numpy as np
import io

class ConjointReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        # Title
        self.cell(0, 10, 'Seu Perfil Exclusivo de Consumidor', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, 'Gerado pelo Sistema Conjoint Analytics Oculto', 0, 0, 'C')

def create_user_report(importance_df: pd.DataFrame, util_df: pd.DataFrame) -> bytes:
    pdf = ConjointReport()
    pdf.add_page()
    pdf.set_font('helvetica', '', 12)
    
    # [SHIELD ANTI-CACHE DO STREAMLIT]
    # Se o Streamlit Cloud ainda estiver usando o `conjoint_engine.py` antigo da RAM,
    # as colunas 'Atributo' e 'Nível' não existirão. Vamos forçar a criação delas:
    if "Atributo" not in util_df.columns and "Nível (Feature)" in util_df.columns:
        util_df["Atributo"] = util_df["Nível (Feature)"].apply(lambda x: x.split("::")[0])
        util_df["Nível"] = util_df["Nível (Feature)"].apply(lambda x: x.split("::")[1])
        util_df["Utilidade Zero-Centrada"] = util_df["Utilidade (Beta)"]
        
    # Text intro
    intro = "Muito obrigado por concluir a nossa pesquisa! Analisando fielmente as escolhas que você acabou de fazer, o nosso Algoritmo de Inteligencia Artificial mediu exatamente o peso psicologico de cada fator na sua Cabeca."
    # Ensure fpdf2 can map accents securely if they fall out
    pdf.multi_cell(0, 8, intro)
    pdf.ln(5)
    
    # Drawing matplotlib chart to BytesIO
    plt.figure(figsize=(8, 4))
    
    # Sort importance starting from lowest to highest for horizontal barchart
    df_sorted = importance_df.sort_values(by="Relative Importance (%)", ascending=True)
    
    plt.barh(df_sorted["Atributo"], df_sorted["Relative Importance (%)"], color='#2980b9')
    plt.title('Quais atributos dominam a sua decisao? (%)')
    plt.xlabel('Peso Relativo (%)')
    plt.tight_layout()
    
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=150)
    img_bytes.seek(0)
    plt.close()
    
    # Insert image to PDF (fpdf2 accepts BytesIO directly)
    pdf.image(img_bytes, x=15, w=180)
    pdf.ln(10)
    
    # Most important attribute text
    if len(importance_df) > 0:
        top_attr = importance_df.iloc[0]["Atributo"]
        top_perc = importance_df.iloc[0]["Relative Importance (%)"]
        pdf.set_font('helvetica', 'B', 12)
        pdf.multi_cell(0, 10, f"Maior Influencia: Para voce, o Atributo [{top_attr}] e o fator mais importante, correspondendo sozinho a {top_perc}% do peso da sua escolha!")
        pdf.ln(5)
        
        # Best Levels
        pdf.set_font('helvetica', '', 11)
        pdf.multi_cell(0, 8, "De todos os produtos concorrentes na arena, estas foram as configuracoes vitoriosas (com a maior utilidade matematica) para o seu perfil:")
        pdf.ln(3)
        for attr in importance_df["Atributo"]:
            sub = util_df[util_df["Atributo"] == attr]
            if len(sub) > 0:
                best_level = sub.loc[sub["Utilidade Zero-Centrada"].idxmax()]["Nível"]
                # Convert anything to compatible latin1
                try: 
                    clean_best = best_level.encode('latin1', 'ignore').decode('latin1')
                    clean_attr = attr.encode('latin1', 'ignore').decode('latin1')
                except:
                    clean_best = best_level; clean_attr = attr
                pdf.cell(0, 8, f"- {clean_attr}: {clean_best}", ln=True)
    
    pdf.ln(10)
    pdf.set_font('helvetica', 'I', 11)
    pdf.multi_cell(0, 8, "As Respostas geradas por este painel ajudarao inumeras marcas a calibrar produtos exatos para padroes parecidos com o seu.")
    
    # Convert FPDF to bytes safely
    return bytes(pdf.output())

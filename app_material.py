import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Rede Neural - Previsão de Propriedades Mecanicas PLA",
    page_icon="🔬",
    layout="wide"
)

# Estilo CSS customizado para os cartões de métricas
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #007bff;
    }
    .metric-label {
        font-size: 16px;
        color: #555;
    }s
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CARREGAMENTO DE MODELOS (CACHE)
# ==========================================
@st.cache_resource
def carregar_modelos():
    # Nomes dos arquivos gerados no treino
    model_path = "modelo_material_otimizado.pkl"
    sx_path = "scaler_X.pkl"
    sy_path = "scaler_y.pkl"
    
    if os.path.exists(model_path) and os.path.exists(sx_path) and os.path.exists(sy_path):
        model = joblib.load(model_path)
        sx = joblib.load(sx_path)
        sy = joblib.load(sy_path)
        return model, sx, sy
    else:
        return None, None, None

model, sx, sy = carregar_modelos()

# ==========================================
# BARRA LATERAL - ENTRADAS (INPUTS)
# ==========================================
st.sidebar.header("🔬 Parâmetros de Processo")
st.sidebar.write("Ajuste os valores para prever as propriedades:")

# Valores baseados nos limites do seu dataset (ajuste se necessário)
temp = st.sidebar.slider("Temperatura (°C)", min_value=180.0, max_value=240.0, value=200.0, step=0.1)
espessura = st.sidebar.slider("Espessura (mm)", min_value=0.05, max_value=0.6, value=0.3, step=0.01)
velocidade = st.sidebar.slider("Velocidade (m/min)", min_value=20.0, max_value=60.0, value=40.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.info("Este app utiliza uma Rede Neural MLP treinada com dados experimentais.")

# ==========================================
# ÁREA PRINCIPAL - PREVISÃO E VISUALIZAÇÃO
# ==========================================
st.title("Previsão de Propriedades de Materiais de Impressão 3D via rede neural")
st.write("Ajuste os parâmetros na barra lateral para ver a previsão instantânea das propriedades mecânicas.")

if model is not None:
    # 1. Realizar a Previsão
    entrada = np.array([[temp, espessura, velocidade]])
    entrada_scaled = sx.transform(entrada)
    previsao_scaled = model.predict(entrada_scaled)
    previsao = sy.inverse_transform(previsao_scaled)[0]
    
    young, ultimate, maximum = previsao[0], previsao[1], previsao[2]

    # 2. Exibir Métricas em Colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Módulo de Young</div>
            <div class="metric-value">{young:.4f}</div>
        </div>""", unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Tensão Ultimate</div>
            <div class="metric-value">{ultimate:.4f}</div>
        </div>""", unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Tensão Máxima (Maximum)</div>
            <div class="metric-value">{maximum:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.write("")

    # 3. Gráfico de Radar (Visualização de Perfil)
    st.subheader("📊 Perfil do Material")
    
    # Normalizar para o gráfico (escala 0-100 para visualização)
    # Aqui usamos valores de referência aproximados para o gráfico
    categorias = ['Young', 'Ultimate', 'Maximum']
    valores = [young, ultimate, maximum]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
          r=valores,
          theta=categorias,
          fill='toself',
          name='Propriedades Previstas',
          line_color='#007bff'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(visible=True, range=[0, max(valores)*1.2])
      ),
      showlegend=False,
      height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # 4. Exportação de Dados
    st.subheader("💾 Exportar Resultado")
    df_res = pd.DataFrame({
        "Parâmetro": ["Temperatura", "Espessura", "Velocidade", "Young", "Ultimate", "Maximum"],
        "Valor": [temp, espessura, velocidade, young, ultimate, maximum]
    })
    
    csv = df_res.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar Previsão em CSV",
        data=csv,
        file_name=f"previsao_{temp}_{espessura}_{velocidade}.csv",
        mime='text/csv',
    )

else:
    st.error("❌ Erro: Arquivos do modelo não encontrados!")
    st.warning("Certifique-se de que 'modelo_material_otimizado.pkl', 'scaler_X.pkl' e 'scaler_y.pkl' estão na mesma pasta deste script.")
    st.info("Execute o script de treinamento primeiro para gerar esses arquivos.")

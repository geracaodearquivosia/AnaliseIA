"""
Analisador de Dados Interativo com IA (Versão com Login Integrado e Sugestões Avançadas)

Esta versão aprimora a seção de "Sugestões e Gráficos Recomendados" para incluir
análises bivariadas e passos mais concretos para a próxima fase da EDA.
"""

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import hashlib
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from io import BytesIO

# --- 2. FUNÇÕES DE AUTENTICAÇÃO E UTILIDADE ---

def hash_password(password: str) -> str:
    """Criptografa a senha usando SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def init_session_state():
    """Inicializa as variáveis de estado de sessão para o app e autenticação."""
    if 'users' not in st.session_state:
        st.session_state['users'] = {}
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'current_user' not in st.session_state:
        st.session_state['current_user'] = None
    if 'current_api_key' not in st.session_state:
        st.session_state['current_api_key'] = None
    if 'mode' not in st.session_state:
        st.session_state['mode'] = 'login'
    if 'llm' not in st.session_state:
        st.session_state['llm'] = None

def get_langchain_model(api_key: str):
    """Inicializa o LLM usando a chave de API fornecida pelo usuário."""
    if not api_key:
        load_dotenv()
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Chave de API do Google não encontrada. Certifique-se de ter cadastrado sua chave ou configurado o arquivo .env.")
        api_key = key
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    return llm

def get_ai_response(llm, prompt: str):
    """Invoca o modelo de linguagem com um prompt específico."""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Ocorreu um erro ao chamar a API via LangChain: {e}"

# --- 3. LÓGICA DE CADASTRO ---

def show_signup_form():
    """Exibe o formulário de cadastro, incluindo a solicitação da API Key."""
    st.title("Cadastrar Novo Usuário")

    st.markdown("### 🔑 Obtenha sua API Key do Gemini")
    st.info("""
    Para utilizar o assistente de IA, você precisará de uma chave de API gratuita do Google Gemini. Siga os passos:

    1. Clique no link: **[Google AI Studio - Criar API Key](https://aistudio.google.com/app/apikey)**.
    2. Clique no botão **`Create API key`**.
    3. Copie a chave gerada e cole no campo abaixo.
    """)

    with st.form("signup_form"):
        new_username = st.text_input("1. Nome de Usuário (Username):").strip()
        new_password = st.text_input("2. Senha:", type="password")
        confirm_password = st.text_input("3. Confirme a Senha:", type="password")
        new_api_key = st.text_input("4. Gemini API Key:", type="password").strip()

        submitted = st.form_submit_button("Cadastrar")

        if submitted:
            if not new_username or not new_password or not confirm_password or not new_api_key:
                st.warning("Preencha todos os campos, incluindo a API Key.")
            elif new_password != confirm_password:
                st.error("As senhas não coincidem.")
            elif new_username in st.session_state['users']:
                st.error("Nome de usuário já cadastrado.")
            else:
                hashed_pw = hash_password(new_password)
                st.session_state['users'][new_username] = {
                    'password': hashed_pw, 
                    'api_key': new_api_key
                }
                st.success("Cadastro realizado com sucesso! Faça o login.")
                st.session_state['mode'] = 'login'
                st.rerun()

    st.markdown("---")
    if st.button("Já sou cadastrado (Ir para Login)", key="go_to_login_from_signup"):
        st.session_state['mode'] = 'login'
        st.rerun()


# --- 4. LÓGICA DE LOGIN ---

def show_login_form():
    """Exibe o formulário de login."""
    st.title("Login de Usuário")

    with st.form("login_form"):
        username = st.text_input("Nome de Usuário (Username):").strip()
        password = st.text_input("Senha:", type="password")

        submitted = st.form_submit_button("Entrar")

        if submitted:
            if not username or not password:
                st.warning("Preencha todos os campos.")
            elif username not in st.session_state['users']:
                st.error("Usuário não encontrado.")
            else:
                user_data = st.session_state['users'][username]
                stored_hashed_pw = user_data['password']
                input_hashed_pw = hash_password(password)

                if input_hashed_pw == stored_hashed_pw:
                    st.session_state['logged_in'] = True
                    st.session_state['current_user'] = username
                    st.session_state['current_api_key'] = user_data['api_key'] 
                    st.success(f"Bem-vindo(a), {username}!")
                    st.rerun()
                else:
                    st.error("Senha incorreta.")

    st.markdown("---")
    st.caption("Ainda não tem conta?")
    if st.button("Ir para Cadastro", key="go_to_signup_from_login"):
        st.session_state['mode'] = 'signup'
        st.rerun()


# --- 5. LÓGICA DO DATA READER (FUNÇÕES AUXILIARES) ---

def ler_json_robusto(file_object) -> pd.DataFrame:
    try:
        file_object.seek(0)
        df = pd.read_json(file_object, lines=True)
        if not df.empty: return df
    except Exception: pass
    try:
        file_object.seek(0)
        dados = json.load(file_object)
        if isinstance(dados, list):
            df = pd.json_normalize(dados)
        elif isinstance(dados, dict):
            chave_da_lista = next(
                (chave for chave, valor in dados.items() if isinstance(valor, list)), None)
            if chave_da_lista:
                df = pd.json_normalize(dados[chave_da_lista])
            else:
                df = pd.DataFrame([dados])
        else: return pd.DataFrame()
        return df
    except Exception: return pd.DataFrame()

def ler_arquivo_para_dataframe(arquivo):
    try:
        if arquivo.name.endswith('.csv'):
            arquivo.seek(0)
            df = pd.read_csv(arquivo)
        elif arquivo.name.endswith('.xlsx') or arquivo.name.endswith('.xls'):
            arquivo.seek(0)
            bytes_data = arquivo.read()
            xls = pd.ExcelFile(BytesIO(bytes_data))
            sheets = xls.sheet_names
            if len(sheets) > 1:
                sheet = st.selectbox("Selecione a aba do arquivo Excel:", sheets, key="excel_sheet_select")
            else:
                sheet = sheets[0]
            df = pd.read_excel(BytesIO(bytes_data), sheet_name=sheet)
        elif arquivo.name.endswith('.json'):
            df = ler_json_robusto(arquivo)
        else:
            st.error("Formato de arquivo não suportado.")
            return None
        if df is None: return None
        mask = [not str(col).startswith('Unnamed') for col in df.columns]
        df = df.loc[:, mask]
        return df
    except Exception as e:
        st.error(f"Ocorreu um erro ao ler o arquivo: {e}")
        return None


# --- 6. LÓGICA PRINCIPAL DO APP DE ANÁLISE DE DADOS (COM MELHORIAS) ---

def show_data_analysis_app():
    """
    Função que encapsula toda a lógica do analisador de dados original.
    Inclui a lógica aprimorada de sugestões e gráficos recomendados.
    """
    
    # Exibe informações de usuário logado e botão de Logout
    st.sidebar.title("Sessão")
    st.sidebar.markdown(f"**Usuário:** `{st.session_state['current_user']}`")
    st.sidebar.caption("API Key do Gemini está ativa.")
    
    if st.sidebar.button("Sair (Logout)"):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = None
        st.session_state['current_api_key'] = None
        st.session_state['mode'] = 'login'
        st.session_state['llm'] = None 
        st.rerun()

    st.title("Analisador Interativo de Dados com IA")

    # --- Inicialização da IA ---
    if st.session_state['llm'] is None:
        try:
            st.session_state['llm'] = get_langchain_model(st.session_state['current_api_key'])
        except ValueError as e:
            st.error(f"Erro de configuração da IA: {e}")
            st.session_state['logged_in'] = False 
            st.rerun()
            return
    llm = st.session_state['llm']
    
    # Widget principal para o usuário fazer o upload do arquivo.
    arquivo = st.file_uploader(
        "Faça o upload do seu arquivo (CSV, Excel ou JSON)", type=["csv", "xlsx", "json"])

    if arquivo is not None:
        
        # Lógica de leitura de arquivo (mantida)
        def _clean_unnamed(df):
            mask = [not str(col).startswith('Unnamed') for col in df.columns]
            try: return df.loc[:, mask]
            except Exception: return df
            
        df = None
        df_compare = None
        
        if arquivo.name.endswith(('.xlsx', '.xls')):
            arquivo.seek(0)
            bytes_data = arquivo.read()
            xls = pd.ExcelFile(BytesIO(bytes_data))
            sheets = xls.sheet_names

            if len(sheets) > 1:
                c1, c2 = st.columns(2)
                with c1: sheet1 = st.selectbox("Selecione a Aba 1:", sheets, index=0, key="sheet1")
                with c2: 
                    default_idx = 1 if len(sheets) > 1 else 0
                    sheet2 = st.selectbox("Selecione a Aba 2:", sheets, index=default_idx, key="sheet2")
                try:
                    df1 = pd.read_excel(BytesIO(bytes_data), sheet_name=sheet1)
                    df2 = pd.read_excel(BytesIO(bytes_data), sheet_name=sheet2)
                    df = _clean_unnamed(df1)
                    df_compare = _clean_unnamed(df2)
                except Exception as e:
                    st.error(f"Erro ao ler abas selecionadas: {e}")
                    df = None; df_compare = None
            else:
                try:
                    arquivo.seek(0)
                    df = pd.read_excel(BytesIO(bytes_data), sheet_name=sheets[0])
                    df = _clean_unnamed(df)
                except Exception as e:
                    st.error(f"Erro ao ler o arquivo Excel: {e}")
                    df = None
        else:
            df = ler_arquivo_para_dataframe(arquivo)

        if df is not None and not df.empty:
            tab_dados, tab_visualizacao, tab_ia = st.tabs(
                ["Tabela de Dados", "Gráficos", "Análise com IA"])

            # --- ABA 1, 2 (Dados e Visualização) (Conteúdo Omitido para Foco) ---
            with tab_dados:
                st.header("Exploração do Conjunto de Dados")
                # Lógica de exibição de dados e sliders...
                if df_compare is not None:
                    # Lógica de duas abas...
                    st.write(f"Duas abas detectadas. Use os sliders abaixo para visualizar os dados.")
                    left, right = st.columns(2)
                    with left:
                        st.subheader("Aba 1")
                        num_linhas = len(df)
                        linha_inicio_1, linha_fim_1 = st.slider("Intervalo Aba 1 (linhas):", 0, max(0, num_linhas - 1), (0, min(24, num_linhas - 1)), key="slider_aba1")
                        st.dataframe(df.iloc[linha_inicio_1:linha_fim_1 + 1], width="stretch")
                    with right:
                        st.subheader("Aba 2")
                        num_linhas2 = len(df_compare)
                        linha_inicio_2, linha_fim_2 = st.slider("Intervalo Aba 2 (linhas):", 0, max(0, num_linhas2 - 1), (0, min(24, num_linhas2 - 1)), key="slider_aba2")
                        st.dataframe(df_compare.iloc[linha_inicio_2:linha_fim_2 + 1], width="stretch")
                else:
                    st.write(f"Linhas: {len(df):,} | Colunas: {df.shape[1]}")
                    num_linhas = len(df)
                    linha_inicio, linha_fim = st.slider("Intervalo de Linhas:", 0, max(0, num_linhas - 1), (0, min(24, num_linhas - 1)), key="slider_dados_geral")
                    st.dataframe(df.iloc[linha_inicio:linha_fim + 1], width="stretch")

            with tab_visualizacao:
                st.header("Geração de Gráficos Personalizados")

                # Separação de colunas por tipo para facilitar a escolha
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                all_cols = df.columns.tolist()

                col1, col2 = st.columns([1, 3])

                with col1:
                    st.subheader("Configurações")
                    tipo_grafico = st.selectbox(
                        "Tipo de Gráfico:",
                        ["Histograma", "Gráfico de Barras", "Boxplot", "Gráfico de Pizza", "Gráfico de Dispersão (Scatter)"]
                    )

                    # --- Lógica de Seleção de Colunas baseada no Tipo ---
                    selected_x = None
                    selected_y = None
                    selected_color = None
                    bins = 20
                    
                    if tipo_grafico == "Histograma":
                        if numeric_cols:
                            selected_x = st.selectbox("Selecione a Coluna (Numérica):", numeric_cols)
                            bins = st.slider("Número de Bins (Intervalos):", 5, 100, 20)
                            selected_color = st.selectbox("Agrupar por cor (Hue - Opcional):", ["(Nenhum)"] + categorical_cols)
                        else:
                            st.warning("Não há colunas numéricas para histograma.")

                    elif tipo_grafico == "Gráfico de Barras":
                        if categorical_cols and numeric_cols:
                            selected_x = st.selectbox("Eixo X (Categoria):", categorical_cols)
                            selected_y = st.selectbox("Eixo Y (Numérico - Média):", numeric_cols)
                            selected_color = st.selectbox("Agrupar por cor (Hue - Opcional):", ["(Nenhum)"] + categorical_cols)
                        else:
                            st.warning("Necessário ter colunas categóricas e numéricas.")

                    elif tipo_grafico == "Boxplot":
                        if numeric_cols:
                            selected_y = st.selectbox("Eixo Y (Numérico - Distribuição):", numeric_cols)
                            selected_x = st.selectbox("Eixo X (Agrupamento - Opcional):", ["(Nenhum)"] + categorical_cols)
                        else:
                            st.warning("Não há colunas numéricas para boxplot.")

                    elif tipo_grafico == "Gráfico de Pizza":
                        if categorical_cols:
                            selected_x = st.selectbox("Categoria (Rótulos):", categorical_cols)
                            # Para pizza, geralmente contamos a ocorrência ou somamos um valor
                            metodo_pizza = st.radio("Método:", ["Contagem de Registros", "Soma de Valor"])
                            if metodo_pizza == "Soma de Valor" and numeric_cols:
                                selected_y = st.selectbox("Valor a Somar:", numeric_cols)
                            else:
                                selected_y = None # Indica contagem
                        else:
                            st.warning("Necessário colunas categóricas.")

                    elif tipo_grafico == "Gráfico de Dispersão (Scatter)":
                        if len(numeric_cols) >= 2:
                            selected_x = st.selectbox("Eixo X:", numeric_cols, index=0)
                            selected_y = st.selectbox("Eixo Y:", numeric_cols, index=min(1, len(numeric_cols)-1))
                            selected_color = st.selectbox("Legenda (Cor - Opcional):", ["(Nenhum)"] + categorical_cols)
                        else:
                            st.warning("Necessário pelo menos 2 colunas numéricas.")

                    btn_gerar = st.button("Gerar Gráfico")

                with col2:
                    st.subheader("Visualização")
                    if btn_gerar:
                        try:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.set_palette("viridis")
                            
                            # Conversão do Hue ("Nenhum" -> None)
                            hue_val = selected_color if selected_color != "(Nenhum)" else None
                            
                            # --- Plotagem ---
                            if tipo_grafico == "Histograma" and selected_x:
                                sns.histplot(data=df, x=selected_x, kde=True, bins=bins, hue=hue_val, ax=ax)
                                ax.set_title(f"Histograma de {selected_x}")

                            elif tipo_grafico == "Gráfico de Barras" and selected_x and selected_y:
                                sns.barplot(data=df, x=selected_x, y=selected_y, hue=hue_val, ax=ax, errorbar=None)
                                ax.set_title(f"Média de {selected_y} por {selected_x}")
                                plt.xticks(rotation=45)

                            elif tipo_grafico == "Boxplot" and selected_y:
                                x_val = selected_x if selected_x != "(Nenhum)" else None
                                sns.boxplot(data=df, x=x_val, y=selected_y, hue=hue_val, ax=ax)
                                ax.set_title(f"Boxplot de {selected_y}")
                                if x_val: plt.xticks(rotation=45)

                            elif tipo_grafico == "Gráfico de Pizza" and selected_x:
                                if selected_y: # Soma
                                    data_pie = df.groupby(selected_x)[selected_y].sum()
                                else: # Contagem
                                    data_pie = df[selected_x].value_counts()
                                
                                ax.pie(data_pie, labels=data_pie.index, autopct='%1.1f%%', startangle=90)
                                ax.axis('equal') # Garante que é um círculo
                                ax.set_title(f"Distribuição de {selected_x}")

                            elif tipo_grafico == "Gráfico de Dispersão (Scatter)" and selected_x and selected_y:
                                sns.scatterplot(data=df, x=selected_x, y=selected_y, hue=hue_val, ax=ax)
                                ax.set_title(f"Correlação: {selected_x} vs {selected_y}")

                            st.pyplot(fig)
                            
                            # Botão de download da imagem (opcional)
                            fn = f"grafico_{tipo_grafico.lower()}.png"
                            img = BytesIO()
                            plt.savefig(img, format='png')
                            st.download_button(label="Baixar Imagem", data=img, file_name=fn, mime="image/png")
                            
                            plt.close(fig)

                        except Exception as e:
                            st.error(f"Erro ao gerar gráfico: {e}")
                    else:
                        st.info("Configure as opções à esquerda e clique em 'Gerar Gráfico'.")

            # --- ABA 3: ASSISTENTE COM IA (LANGCHAIN) ---
            with tab_ia:
                st.header("Análise com IA")

                col_left, col_right = st.columns([2, 1])

                # ----- Coluna ESQUERDA: Análise Geral e Sugestões (MELHORADA) -----
                with col_left:
                    session_key_geral = f"analise_geral_{arquivo.name}"

                    st.subheader("Análise Geral Automática")
                    # ... Lógica de geração da análise da IA (mantida) ...
                    info_placeholder = st.empty()
                    if session_key_geral not in st.session_state:
                        info_placeholder.info("A IA está realizando a análise automática do arquivo. Aguarde alguns segundos...")
                        with st.spinner("IA: analisando resumo estatístico..."):
                            resumo_estatistico = df.describe(include='all').to_string()
                            contexto_geral = """Você é um analista de dados. Produza, em Markdown, as seções: 1) Interpretação Geral. 2) Insights Numéricos: para cada coluna numérica informe média, mediana e porcentagens relevantes. 3) Insights Categóricos: para cada coluna categórica mostre porcentagem dos principais valores. 4) Qualidade dos Dados: porcentagem de valores ausentes por coluna. 5) Conclusão Principal. Não inicie com frases de apresentação."""
                            prompt_completo = f"{contexto_geral}\n\nResumo estatístico:\n```\n{resumo_estatistico}\n```\n\nInicie a análise."
                            raw_response = get_ai_response(llm, prompt_completo) if llm else "IA indisponível."
                            cleaned = re.sub(r'(?i)^\s*como analista de dados[^\n.]*[.\n]?\s*', '', raw_response).strip()
                            st.session_state[session_key_geral] = cleaned
                        info_placeholder.empty()

                    analysis_text = st.session_state.get(session_key_geral, "_Aguardando análise..._")
                    st.text_area("Resultado da Análise", value=analysis_text, height=480, disabled=True)

                    st.markdown("---")
                    
                    # -----------------------------------------------------
                    # NOVO: LÓGICA DE SUGESTÕES E GRÁFICOS INTERESSANTES
                    # -----------------------------------------------------

                    st.subheader("Sugestões e Próximos Passos Recomendados")
                    total_rows = len(df)
                    num_cols = df.select_dtypes(include='number').columns.tolist()
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    missing = df.isnull().sum()
                    missing = missing[missing > 0].sort_values(ascending=False)

                    # 1. Qualidade dos Dados e Próximos Passos
                    st.markdown("### 1. Próximos Passos e Qualidade")
                    
                    if not missing.empty:
                        st.markdown("**Tratamento de Dados:**")
                        for col, cnt in missing.items():
                            pct = cnt / total_rows * 100
                            st.markdown(f"- **`{col}`** ({pct:.1f}% ausente) — **Sugestão:** Use imputação de média/mediana para essa coluna ou remova os registros se a perda de dados for pequena.")
                    else:
                        st.markdown("- **Qualidade:** Nenhum valor ausente detectado. Prossiga para a análise de outliers.")

                    # 2. Análise de Variação e Outliers
                    important_num = []
                    if num_cols:
                        numeric_std = df[num_cols].std()
                        numeric_skew = df[num_cols].skew().abs().fillna(0)
                        std_norm = numeric_std / (numeric_std.max() + 1e-9)
                        skew_norm = numeric_skew / (numeric_skew.max() + 1e-9) if numeric_skew.max() > 0 else numeric_skew*0
                        importance_score = (std_norm + skew_norm).sort_values(ascending=False)
                        important_num = importance_score.index.tolist()
                        
                        st.markdown("\n**Ações em Numéricas:**")
                        for col in importance_score.head(3).index:
                             st.markdown(f"- **Investigar outliers em `{col}`** (Alto Std/Skew). **Sugestão:** Gere o **Boxplot** ou **Histograma** na aba Gráficos para confirmar a presença de valores extremos.")
                    
                    # 3. Análise Categórica e Agrupamento
                    important_cat = []
                    if cat_cols:
                        cat_counts_df = pd.DataFrame({c: df[c].nunique() for c in cat_cols}, index=['count']).T.sort_values(by='count')
                        important_cat = cat_counts_df.index.tolist()
                        st.markdown("\n**Ações em Categóricas:**")
                        for c in important_cat[:3]:
                            unique_count = df[c].nunique()
                            if unique_count > 50:
                                st.markdown(f"- **Agrupar/Limpar `{c}`** ({unique_count} categorias). **Sugestão:** Essa coluna tem alta cardinalidade. Considere agrupar categorias raras ou verificar erros de digitação.")
                            else:
                                st.markdown(f"- **Analisar o peso de `{c}`**. **Sugestão:** Use a aba Gráficos para um **Gráfico de Barras** para entender a distribuição de cada categoria.")


                    # -----------------------------------------------------
                    # NOVO: GRÁFICOS RECOMENDADOS (UNIVARIADO E BIVARIADO)
                    # -----------------------------------------------------

                    st.markdown("### 2. Gráficos Recomendados")
                    charts_shown = 0
                    max_charts = 3
                    sns.set_style("whitegrid")

                    # --- Prioridade 1: BIVARIADA (Numérica vs. Categórica) ---
                    if important_num and important_cat and charts_shown < max_charts:
                        num_col = important_num[0]
                        cat_col = important_cat[0]
                        try:
                            fig, ax = plt.subplots(figsize=(7, 4))
                            sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
                            ax.set_title(f"Boxplot: Distribuição de {num_col} por {cat_col}")
                            ax.tick_params(axis='x', rotation=45)
                            st.markdown(f"**Recomendação:** Comparação Bivariada: **Boxplot de `{num_col}` por `{cat_col}`**")
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                            charts_shown += 1
                        except Exception:
                            plt.close('all')

                    # --- Prioridade 2: UNIVARIADA (Distribuição) ---
                    for col in important_num:
                        if charts_shown >= max_charts: break
                        try:
                            fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
                            sns.histplot(data=df, x=col, kde=True, ax=axes[0], color="#2563eb")
                            axes[0].set_title(f"Histograma {col}")
                            sns.boxplot(data=df, x=col, ax=axes[1], color="#0ea5a4")
                            axes[1].set_title(f"Boxplot {col}")
                            st.markdown(f"**Recomendação:** Análise de Distribuição: **Histograma e Boxplot de `{col}`**")
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                            charts_shown += 1
                        except Exception:
                            plt.close('all')
                            continue
                        
                    # --- Prioridade 3: BIVARIADA (Numérica vs. Numérica - Correlação) ---
                    if len(important_num) >= 2 and charts_shown < max_charts:
                         col_x = important_num[0]
                         col_y = important_num[1]
                         try:
                             fig, ax = plt.subplots(figsize=(6, 4))
                             sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax)
                             ax.set_title(f"Scatter Plot: Correlação entre {col_x} e {col_y}")
                             st.markdown(f"**Recomendação:** Análise de Correlação: **Scatter Plot entre `{col_x}` e `{col_y}`**")
                             st.pyplot(fig, clear_figure=True)
                             plt.close(fig)
                             charts_shown += 1
                         except Exception:
                             plt.close('all')
                    
                    
                    if charts_shown == 0:
                        st.markdown("_Nenhum gráfico recomendado. Verifique se há colunas numéricas ou categóricas suficientes no dataset._")

                # ----- Coluna DIREITA: Perguntas detalhadas (MANTIDA) -----
                with col_right:
                    st.subheader("Perguntas Detalhadas")
                    st.caption(f"Total de linhas: {len(df):,}")
                    
                    # Lógica de input para linhas e pergunta...
                    total_rows = len(df)
                    ia_linha_inicio = st.number_input("Analisar da linha (índice inicial):", min_value=0, max_value=total_rows - 1, value=0, step=1, key="qa_start")
                    ia_linha_fim = st.number_input(f"Até a linha (máx {total_rows - 1}):", min_value=0, max_value=total_rows - 1, value=min(99, total_rows - 1), step=1, key="qa_end")
                    if ia_linha_inicio > ia_linha_fim: st.error("A 'Linha Inicial' não pode ser maior que a 'Linha Final'.")
                    st.markdown("---")

                    coluna = st.selectbox("Coluna (opcional):", ["(nenhuma)"] + df.columns.tolist())
                    prompt_usuario = st.text_area("Digite sua pergunta sobre o intervalo selecionado:", height=120, placeholder="Ex: Qual o valor máximo da coluna 'Vendas' neste intervalo?")

                    if st.button("Perguntar à IA"):
                        if not prompt_usuario:
                            st.warning("Digite uma pergunta.")
                        else:
                            with st.spinner(f"A IA está analisando as linhas {ia_linha_inicio} a {ia_linha_fim}..."):
                                df_para_analise = df.iloc[ia_linha_inicio:ia_linha_fim + 1]
                                foco_col = f" Concentre-se na coluna '{coluna}'." if coluna != "(nenhuma)" else ""
                                dados_brutos_str = df_para_analise.to_string()
                                contexto_pergunta = ("Você é um analista de dados. Use exclusivamente os dados abaixo para responder." + foco_col)
                                prompt_pergunta_completo = (
                                    f"{contexto_pergunta}\n\nTabela de dados brutos:\n```\n{dados_brutos_str}\n```\n\n"
                                    f"Pergunta do usuário: {prompt_usuario}\n\nResponda de forma direta e objetiva."
                                )
                                resposta_pergunta = get_ai_response(llm, prompt_pergunta_completo) if llm else "IA indisponível."
                                st.markdown("**Resposta da IA:**")
                                st.markdown(resposta_pergunta)


# --- 7. FUNÇÃO MAIN DE ORQUESTRAÇÃO ---

def main():
    """Função principal que controla a navegação entre login/cadastro e o app."""
    st.set_page_config(page_title="Analisador com Login", layout="wide")
    
    init_session_state()

    if st.session_state['logged_in']:
        show_data_analysis_app()
    else:
        if st.session_state['mode'] == 'login':
            show_login_form()
        elif st.session_state['mode'] == 'signup':
            show_signup_form()

if __name__ == "__main__":
    main()

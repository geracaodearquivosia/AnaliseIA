import streamlit as st
import hashlib

# --- 1. FUNÇÕES DE UTILIDADE E SEGURANÇA ---

def hash_password(password: str) -> str:
    """Criptografa a senha usando SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def init_user_storage():
    """Inicializa o armazenamento de usuários no Session State."""
    if 'users' not in st.session_state:
        # Armazena usuários no formato: {'username': 'hashed_password'}
        st.session_state['users'] = {}
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'current_user' not in st.session_state:
        st.session_state['current_user'] = None

# --- 2. LÓGICA DE CADASTRO ---

def show_signup_form():
    """Exibe o formulário de cadastro e processa a submissão."""
    st.title("Cadastrar Novo Usuário")

    with st.form("signup_form"):
        new_username = st.text_input("Nome de Usuário (Username):").strip()
        new_password = st.text_input("Senha:", type="password")
        confirm_password = st.text_input("Confirme a Senha:", type="password")

        submitted = st.form_submit_button("Cadastrar")

        if submitted:
            if not new_username or not new_password or not confirm_password:
                st.warning("Preencha todos os campos.")
            elif new_password != confirm_password:
                st.error("As senhas não coincidem.")
            elif new_username in st.session_state['users']:
                st.error("Nome de usuário já cadastrado.")
            else:
                # Armazena a senha criptografada
                hashed_pw = hash_password(new_password)
                st.session_state['users'][new_username] = hashed_pw
                st.success("Cadastro realizado com sucesso! Faça o login.")
                # Alterna para a tela de login
                st.session_state['mode'] = 'login'
                st.rerun() # Reruna para atualizar a tela

# --- 3. LÓGICA DE LOGIN ---

def show_login_form():
    """Exibe o formulário de login e processa a submissão."""
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
                stored_hashed_pw = st.session_state['users'][username]
                input_hashed_pw = hash_password(password)

                if input_hashed_pw == stored_hashed_pw:
                    st.session_state['logged_in'] = True
                    st.session_state['current_user'] = username
                    st.success(f"Bem-vindo(a), {username}!")
                    st.rerun() # Reruna para mostrar a tela principal
                else:
                    st.error("Senha incorreta.")

    st.markdown("---")
    st.caption("Ainda não tem conta?")
    if st.button("Ir para Cadastro"):
        st.session_state['mode'] = 'signup'
        st.rerun()

# --- 4. TELA PRINCIPAL (APÓS LOGIN) ---

def show_main_app():
    """Exibe o conteúdo principal da aplicação após o login."""
    st.sidebar.title("Informações da Sessão")
    st.sidebar.markdown(f"**Usuário:** {st.session_state['current_user']}")
    
    if st.sidebar.button("Sair (Logout)"):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = None
        st.session_state['mode'] = 'login'
        st.rerun()

    st.title(f"Página Principal")
    st.header("Conteúdo Exclusivo para Usuários Logados")
    st.write("Aqui você integraria a lógica do seu analisador de dados ou qualquer outra funcionalidade principal.")
    
    st.markdown("---")
    # Apenas para debug/demonstração: Mostra a lista de usuários em memória
    st.subheader("Usuários Cadastrados (Apenas Demo)")
    st.json(st.session_state['users'])
    st.warning("Lembre-se: Estes dados são APENAS armazenados na memória do servidor e não persistem após o servidor ser desligado.")

# --- 5. LÓGICA DE NAVEGAÇÃO CENTRAL ---

def main():
    """Função principal que orquestra a navegação."""
    st.set_page_config(page_title="Login/Cadastro em Memória", layout="centered")
    init_user_storage()

    # Se estiver logado, exibe o app principal
    if st.session_state['logged_in']:
        show_main_app()
    else:
        # Se não estiver logado, alterna entre login e cadastro
        if 'mode' not in st.session_state:
            st.session_state['mode'] = 'login'

        if st.session_state['mode'] == 'login':
            show_login_form()
        elif st.session_state['mode'] == 'signup':
            show_signup_form()

if __name__ == "__main__":
    main()
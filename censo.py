import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="ChatBot Censo Escolar SP",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- ESTILIZAÇÃO CSS (Branco e Azul Moderno) ---
st.markdown("""
    <style>
    /* Fundo Branco Principal */
    .stApp {
        background-color: #FFFFFF;
        color: #333333;
    }
    
    /* Cabeçalhos em Azul */
    h1, h2, h3 {
        color: #0056b3 !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Botão Estilizado (Azul) */
    .stButton>button {
        background-color: #0056b3;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #004494;
        color: white;
    }

    /* Mensagens do Chat */
    .stChatMessage {
        background-color: #F0F8FF; /* Azul alice muito claro */
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- BASE DE CONHECIMENTO (DADOS DO USUÁRIO) ---
# Dicionário simples mapeando Pergunta -> Resposta
kb_data = {
    "Como cadastrar uma escola nova?": "A escola deve entrar em contato com a secretaria estadual de Educação (Fale Conosco), solicitar um questionário de escola nova, preenchê-lo e devolvê-lo à secretaria estadual.",
    "Quais são os perfis de acesso ao Sistema Educacenso?": "Os perfis são: Inep (todas as escolas), Secretaria Estadual, Setec/MEC (federais), Órgão Regional, Secretaria Municipal e Perfil Escola (apenas dados da escola).",
    "Quais são os níveis de acesso ao Sistema Educacenso?": "Os níveis são: Leitor (apenas visualiza), Executor (preenche e altera) e Superusuário (preenche, altera e gerencia usuários).",
    "Como cadastrar um novo usuário no Sistema Educacenso?": "O superusuário da entidade deve acessar o Sistema Educacenso, preencher as informações e cadastrar o novo usuário.",
    "Como proceder quando não houver Superusuário cadastrado na escola?": "A escola deve procurar a entidade hierarquicamente superior (secretaria municipal/estadual ou Setec/MEC) e solicitar o cadastramento de um responsável com nível de Superusuário.",
    "É preciso retirar o acesso de usuário que não trabalha mais com o Censo Escolar?": "Sim. O superusuário deve clicar no menu 'Usuário' > 'Gerenciar', encontrar o usuário e clicar em 'excluir vínculo'.",
    "Como alterar o e-mail de um usuário?": "O superusuário deve acessar 'Usuário', pesquisar o cadastro, clicar no ícone do lápis, alterar o e-mail e clicar em 'Salvar'.",
    "Esqueci minha senha. O que devo fazer?": "Na página inicial, clique em 'Esqueceu a senha?', preencha o CPF e clique em 'Enviar' para receber um link de redefinição.",
    "O que fazer se o link de ativação expirou?": "Acesse a tela de login, clique em 'Esqueceu a senha?' e insira seu CPF para receber um novo e-mail.",
    "Qual é o período de preenchimento do Censo Escolar 2025?": "A 1ª etapa (Matrícula Inicial) é de 28 de maio a 31 de julho de 2025. A data de referência é 28/05/2025.",
    "Como informar os dados do gestor da escola?": "Acesse a escola, pesquise pelo gestor. Se não achar, clique em 'Cadastrar Gestor Escolar'. Se achar, clique em 'Vincular'.",
    "Como cadastrar um novo aluno?": "Clique no menu 'Aluno', pesquise em 'Todo o Brasil' pelo CPF ou nome/nascimento. Se não encontrar, clique em 'Cadastrar aluno'.",
    "Como cadastrar um novo profissional escolar?": "Clique no menu 'Profissional escolar', pesquise em 'Todo o Brasil' (preferencialmente por CPF). Se não encontrar, clique em 'Cadastrar profissional escolar'.",
    "O que fazer se o aluno ou profissional foi cadastrado por engano?": "Entre em contato com a coordenação estadual do Censo Escolar e informe o ID para exclusão. Apenas o Inep pode excluir registros do banco de dados.",
    "Como editar dados cadastrais de alunos ou profissionais?": "No menu 'Aluno' ou 'Profissional', pesquise 'Apenas na escola', clique no ícone do lápis para editar. O nome fica bloqueado se vinculado ao CPF (deve alterar na Receita Federal).",
    "Quais transtornos de aprendizagem são coletados a partir de 2025?": "São coletados: TDAH, Dislexia, Disgrafia, Disortografia, Discalculia, Dislalia e TPAC.",
    "O que é TPAC?": "Transtorno do Processamento Auditivo Central: dificuldade em interpretar informações sonoras, embora a detecção do som seja normal.",
    "Qual é o período da Situação do Aluno 2024?": "A coleta ocorre de 3 de fevereiro a 14 de março de 2025.",
    "Quem deve responder à Situação do Aluno?": "Todas as escolas que informaram matrículas na 1ª etapa (Matrícula Inicial), exceto as exclusivas de AEE/atividade complementar.",
    "Como informar escolaridade do gestor ou profissional?": "Declare o 'Maior nível de escolaridade concluído'. Se estiver cursando superior, declare Ensino Médio. Se tiver superior, informe até 3 cursos.",
    "A educação a distância é coletada?": "Sim, deve ser declarada no campo 'Tipo de mediação didático-pedagógica' para Ensino Regular, EJA e Educação Profissional."
}

# Separa perguntas e respostas para o modelo
questions = list(kb_data.keys())
answers = list(kb_data.values())

# --- LÓGICA DE INTELIGÊNCIA (BUSCA) ---
@st.cache_resource
def setup_search_engine():
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = setup_search_engine()

def get_best_answer(user_query):
    # Vetoriza a pergunta do usuário
    user_vec = vectorizer.transform([user_query])
    # Calcula similaridade
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    best_match_idx = np.argmax(similarities)
    score = similarities[0][best_match_idx]
    
    # Limiar de confiança (ajustável)
    if score > 0.3:
        return answers[best_match_idx]
    else:
        return None

# --- GERENCIAMENTO DE ESTADO (NAVEGAÇÃO) ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'messages' not in st.session_state:
    st.session_state.messages = []

def go_to_chat():
    st.session_state.page = 'chat'

# --- PÁGINA 1: HOME ---
if st.session_state.page == 'home':
    st.markdown("<div style='text-align: center; margin-top: 50px;'>", unsafe_allow_html=True)
    st.title("Bem-vindo ao ChatBot Censo Escolar SP")
    st.markdown("### Tire suas dúvidas sobre o Sistema Educacenso de forma rápida e simples.")
    st.write("Este assistente virtual utiliza a base de dados oficial para orientar sobre cadastros, perfis, prazos e correções.")
    st.write("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("INICIAR CONVERSA", on_click=go_to_chat)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='position: fixed; bottom: 20px; width: 100%; text-align: center; color: #888;'>
        Projeto de Apoio à Educação - SP
    </div>
    """, unsafe_allow_html=True)

# --- PÁGINA 2: CHAT ---
elif st.session_state.page == 'chat':
    st.title("💬 Atendimento Censo Escolar")
    st.caption("Pergunte sobre cadastros, prazos, perfis, etc.")

    # Botão para voltar (opcional, pequeno no sidebar ou topo)
    if st.button("⬅ Voltar ao Início", key="back"):
        st.session_state.page = 'home'
        st.rerun()

    # Exibe histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada do usuário
    if prompt := st.chat_input("Digite sua dúvida aqui..."):
        
        # Validação de Caracteres (Regra 1.1)
        if len(prompt) > 100:
            st.warning(f"Sua mensagem tem {len(prompt)} caracteres. O limite é 100.")
        else:
            # Mostra mensagem do usuário
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Lógica de Resposta
            answer = get_best_answer(prompt)
            
            if answer:
                response_text = answer
            else:
                response_text = ("Desculpe, não encontrei uma resposta conclusiva em minha base. "
                                 "Por favor, entre em contato com **atendimento.educacao.sp.gov.br**")

            # Mostra resposta do bot
            with st.chat_message("assistant"):
                st.markdown(response_text)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})

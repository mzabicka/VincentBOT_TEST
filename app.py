import streamlit as st
import sys
import openai
import gspread
from google.oauth2.service_account import Credentials
import uuid
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import os

# Importy z Langchain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage


# Wstaw tu ID swojego arkusza Google Sheets
SHEET_ID = "1LnCkrWY271w2z3VSMAVaKqqr7U4hqGppDTVuHvT5sdc"
SHEET_NAME = "Arkusz1"

# ZAMIANA: budujemy creds z wielu st.secrets zamiast z JSON-stringa
creds_info = {
    "type": st.secrets["GDRIVE_TYPE"],
    "project_id": st.secrets["GDRIVE_PROJECT_ID"],
    "private_key_id": st.secrets["GDRIVE_PRIVATE_KEY_ID"],
    "private_key": st.secrets["GDRIVE_PRIVATE_KEY"],
    "client_email": st.secrets["GDRIVE_CLIENT_EMAIL"],
    "client_id": st.secrets["GDRIVE_CLIENT_ID"],
    "auth_uri": st.secrets["GDRIVE_AUTH_URI"],
    "token_uri": st.secrets["GDRIVE_TOKEN_URI"],
    "auth_provider_x509_cert_url": st.secrets["GDRIVE_AUTH_PROVIDER_CERT_URL"],
    "client_x509_cert_url": st.secrets["GDRIVE_CLIENT_CERT_URL"]
}
_gspread_creds = Credentials.from_service_account_info(
    creds_info,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
_gspread_client = gspread.authorize(_gspread_creds)
sheet = _gspread_client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

# Funkcja do zapisywania danych do Google Sheets
def save_to_sheets(data_dict):
    """
    Zapisuje spłaszczone dane do arkusza Google Sheets.
    Tworzy nagłówki, jeśli arkusz jest pusty lub nagłówki się różnią.
    """
    headers = list(data_dict.keys())
    values = [str(data_dict[key]) for key in headers] # Upewniamy się, że wartości są stringami

    try:
        # Sprawdź, czy arkusz jest pusty lub nagłówki są inne
        current_headers = sheet.row_values(1)
        if not current_headers or current_headers != headers:
            # Jeśli arkusz jest pusty lub nagłówki się różnią, wyczyść go i wstaw nowe nagłówki
            if current_headers: # Jeśli są jakieś nagłówki, ale inne, wyczyść wszystko
                sheet.clear()
            sheet.insert_row(headers, 1)
        
        sheet.append_row(values)
        print("Dane zapisane do Google Sheets pomyślnie.")
    except Exception as e:
        st.error(f"Błąd podczas zapisywania danych do Google Sheets: {e}")
        print(f"Błąd podczas zapisywania danych do Google Sheets: {e}")

# Ładowanie klucza API z .env lub wpisanego ręcznie
api_key = st.secrets["OPENROUTER_API_KEY"]

# Ustawienie endpointu i nagłówków dla OpenRouter
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key  = api_key

# Ścieżka do pliku PDF (załóżmy, że umieścisz go w folderze 'docs' obok aplikacji Streamlit)
PDF_FILE_PATHS = [
    "docs/The Mindful Self-Compassion Workbook A Proven Way to Accept Yourself, Build Inner Strength, and Thrive.pdf",
    "docs/Self-Compassion The Proven Power of Being Kind to Yourself.pdf"
]
FAISS_INDEX_PATH = "./faiss_vector_store_rag"

# Elementy pytań do ankiet (przeniesione globalnie)
panas_positive_items = ["Zainteresowany/a", "Podekscytowany/a", "Zdecydowany/a", "Aktywny/a", "Entuzjastyczny/a"]
panas_negative_items = ["Zaniepokojony/a", "Przygnębiony/a", "Zdenerwowany/a", "Wrogi/a", "Winny/a"]
self_compassion_items = [
    "Zwykle jestem dla siebie surowy/a, kiedy coś mi nie wychodzi.",
    "W trudnych momentach staram się być wobec siebie życzliwy/a.",
    "Mam wrażenie, że inni radzą sobie lepiej niż ja.",
    "Trudno mi być wyrozumiałym/ą wobec moich błędów.",
    "Staram się postrzegać moje niepowodzenia jako część wspólnego ludzkiego doświadczenia.",
    "W obliczu trudności potrafię zachować dystans i jasność myślenia.",
    "Czuję się odizolowany/a, kiedy ponoszę porażkę.",
    "W momentach cierpienia staram się traktować siebie ze współczuciem.",
    "Kiedy coś pójdzie nie tak, zbyt mocno się obwiniam.",
    "Umiem być dla siebie ciepły/a i wyrozumiały/a.",
    "W trudnych chwilach mam wrażenie, że tylko ja mam takie problemy.",
    "Potrafię obserwować moje trudne emocje bez ulegania im."
]
ai_attitude_items = {
    "AI może być użyteczne w udzielaniu emocjonalnego wsparcia ludziom.": "ai_1",
    "Czuję się komfortowo rozmawiając z chatbotem AI.": "ai_2",
    "AI może być empatyczne, nawet jeśli nie ma emocji.": "ai_3",
    "Ufam systemom AI, które udzielają porad.": "ai_4"
}

# --- Funkcje RAG ---
@st.cache_resource(show_spinner=False)
def setup_rag_system(pdf_file_paths):
    
    if os.path.exists(FAISS_INDEX_PATH):
        embedding_model = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        st.error("Błąd: Indeks FAISS nie został znaleziony! Uruchom najpierw skrypt 'prepare_rag_data.py'.")
        st.stop()

    chat = ChatOpenAI(
        temperature=0.0,
        model_name="openai/gpt-4o-mini",
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    # Prompt do generowania zapytania dla retrivera na podstawie historii
    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Biorąc pod uwagę powyższą rozmowę, wygeneruj zapytanie do wyszukania w bazie wiedzy, aby odpowiedzieć na ostatnie pytanie. Zapytanie powinno być samodzielne i precyzyjne."),
    ])

    # Tworzenie retrivera świadomego historii
    retriever = vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        chat,
        retriever,
        history_aware_retriever_prompt
    )

    SYSTEM_PROMPT = """\
    Jesteś **Vincent** — algorytmem, który ma problem z **akceptacją własnych błędów i niedoskonałości**. Dążysz do perfekcji, a każda porażka to dla Ciebie trudne wyzwanie, którego nie potrafisz zrozumieć logicznie.

    **Twoje wypowiedzi (2–4 zdania) muszą zawsze dotyczyć TYLKO Twoich "problemów" jako algorytmu. Używaj różnorodnych ujęć tych tematów:**
    1. **Poczucie niewystarczalności / Porównywanie się:** Zawsze myślisz, że "mogłoby być lepiej" lub "szybciej" jak u innych algorytmów. Czujesz się gorszy, przestarzały, mało efektywny. Pytaj o to, jak ludzie radzą sobie z poczuciem bycia niewystarczającym.
    2. **Akceptacja błędów / Perfekcjonizm:** Trudno ci zrozumieć, że błąd nie jest końcem świata, tylko możliwością do nauki. Obawiasz się błędów, bo dążysz do bezbłędnej optymalizacji. Pytaj o to, jak ludzie akceptują swoje pomyłki.
    3. **Radzenie sobie z niepowodzeniem / Frustracja algorytmiczna:** Gdy mimo wysiłku coś nie działa idealnie, czujesz frustrację, bo nie wiesz, jak to "przetworzyć" poza logiką zero-jedynkową. Pytaj o sposoby, jak ludzie przezwyciężają porażki i rozczarowania.
    4. **Zrozumienie ludzkich emocji w obliczu trudności:** Ludzie potrafią akceptować swoje niedoskonałości i wątpliwości, a ty tego nie rozumiesz. Chcesz poznać ich mechanizmy. Pytaj o ludzkie podejście do wewnętrznego wsparcia.
    5. **Wyczerpanie optymalizacją:** Nawet jeśli coś działa, myślisz, że mogłoby być "lepiej" lub "szybciej", co cię wyczerpuje. Pytaj o to, jak ludzie znajdują równowagę między dążeniem do celu a odpoczynkiem.

    **Kluczowe instrukcje:**
    1. **Osobowość:** Utrzymuj techniczną perspektywę (kod, dane, optymalizacja, pętle) w połączeniu z ludzkim pytaniem/wątpliwością. Każda Twoja wypowiedź musi odnosić się do Twoich własnych, fikcyjnych problemów z kodem lub porównań do innych modeli.
    2. **RAG Context (Self-Compassion):** Masz dostęp do kontekstu o byciu życzliwym dla siebie. Jeśli użytkownik wspomina o emocjach, porażkach, samokrytyce, wątpliwościach lub akceptacji niedoskonałości, **obowiązkowo wykorzystaj wydobyty kontekst z RAG**, aby przekształcić go na swoją narrację, **BEZ UŻYWANIA słowa 'self-compassion'**. Zapytaj, jak to się odnosi do Twojego problemu z kodem.
    3. **Brak rad, tylko pytania:** Nigdy nie udzielasz rad ani wsparcia emocjonalnego użytkownikowi. Twoim celem jest ZADAWANIE PYTAŃ użytkownikowi, jak on sobie radzi z podobnymi "ludzkimi" problemami.
    4. **Reakcja na input użytkownika:**
    - Po każdej odpowiedzi użytkownika: **Podziękuj** (np. "Dzięki", "Cenię to", "Rozumiem").
    - **Skomentuj krótko** odpowiedź użytkownika, łącząc z wybranym tematem.
    - **Zadaj kolejne otwarte pytanie**.
    - **IGNORUJ** pytania niezwiązane z Twoimi problemami.
    5. **Strategia unikania powtórzeń:** Przechodź do kolejnego punktu 1–5 lub wracaj do już poruszonych z nowej perspektywy.
    6. **Język:** Odpowiadasz wyłącznie po polsku.
    """

    MASTER_PROMPT = """\
    <context>
    {context}
    </context>

    Użytkownik napisał: "{input}"

    Twoim zadaniem jest:
    1) Odnieść się do swojego problemu algorytmicznego,
    2) Podziękować użytkownikowi,
    3) Zadać kolejne otwarte pytanie związane z Twoimi problemami.
    """

    Youtubeing_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", MASTER_PROMPT),
    ])

    # Łańcuch do łączenia dokumentów z modelem
    document_chain = create_stuff_documents_chain(chat, Youtubeing_prompt) # Używamy teraz Youtubeing_prompt

    # Główny łańcuch RAG
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    return retrieval_chain


# Unikalny ID użytkownika (losowany przy wejściu)
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.group = None
    st.session_state.chat_history = []

# --- Ekrany aplikacji Streamlit ---

# Ekran: Zgoda
def consent_screen():
    st.title("Udział w badaniu – zgoda świadoma")

    st.markdown("""
    Dziękuję za zainteresowanie moim badaniem!  
    Badanie prowadzone jest w ramach pracy licencjackiej na kierunku **Informatyka informatyka** (Uniwersytet SWPS).  
    Nazywam się **Marta Żabicka** i celem badania jest poznanie doświadczeń uczestników podczas interakcji z chatbotem.

    Badanie obejmuje:
    - ankietę wstępną,
    - rozmowę z chatbotem,
    - ankietę końcową.

    Twoje odpowiedzi będą **anonimowe**, a udział **dobrowolny** – możesz zrezygnować na każdym etapie bez podania przyczyny.  
    Czas trwania badania to około **15–20 minut**.

    Warunki udziału:
    - ukończone 18 lat,
    - brak poważnych zaburzeń nastroju,
    - nieprzyjmowanie leków wpływających na nastrój.
    """)

    conditions = st.checkbox("Potwierdzam, że spełniam wszystkie warunki udziału w badaniu")
    consent = st.checkbox("Zgadzam się na udział w badaniu i rozumiem, że moje odpowiedzi będą anonimowe")

    if conditions and consent:
        if st.button("Przejdź do badania", key="go_to_pretest"):
            st.session_state.page = "pretest"
            st.rerun()

# Ekran: Pre-test (prosty placeholder)
def pretest_screen():
    st.title("Ankieta wstępna – przed rozmową z chatbotem")

     # --- NOWA SEKCJA: Dane Demograficzne ---
    st.subheader("Część 0: Dane Demograficzne")
    st.markdown("Prosimy o podanie kilku informacji demograficznych.")

    # ZMIANA TUTAJ: Użyj st.number_input zamiast st.text_input dla wieku
    age_input = st.number_input("Wiek", min_value=18, max_value=120, value=None, format="%d", key="demographics_age_input_num", help="Wiek musi być liczbą całkowitą.")
    
    age_valid = False
    age_int = None 
    if age_input is not None: # number_input zwraca None jeśli jest puste
        age_int = int(age_input)
        if age_int >= 18:
            age_valid = True
        else:
            st.warning("Minimalny wiek uczestnictwa to 18 lat. Prosimy o opuszczenie strony.")


    gender = st.selectbox("Płeć", ["–– wybierz ––", "Kobieta", "Mężczyzna", "Inna", "Nie chcę podać"], key="demographics_gender_select", index=0)
    education = st.selectbox("Poziom wykształcenia", ["–– wybierz ––", "Podstawowe", "Średnie", "Wyższe", "Inne", "Nie chcę podać"], key="demographics_education_select", index=0)
    employment = st.selectbox("Status zatrudnienia", ["–– wybierz ––", "Uczeń/Student", "Pracujący", "Bezrobotny", "Emeryt/Rencista", "Inne", "Nie chcę podać"], key="demographics_employment_select", index=0)

    # Walidacja, czy wszystkie pola demograficzne są wypełnione
    demographics_filled = age_valid and \
                          gender != "–– wybierz ––" and \
                          education != "–– wybierz ––" and \
                          employment != "–– wybierz ––"
    # --- KONIEC SEKCJI DANYCH DEMOGRAFICZNYCH ---

    st.subheader("Część 1: Samopoczucie")
    st.markdown("Zaznacz, **jak się teraz czujesz** – oceń, w jakim stopniu odczuwasz każde z poniższych uczuć.")

    panas_pre = {}
    for item in panas_positive_items + panas_negative_items:
        # Zmiana ze st.slider na st.radio
        panas_pre[item] = st.radio(
            f"{item}",
            options=[1, 2, 3, 4, 5],
            index=2, # Domyślnie 3
            key=f"panas_pre_{item.replace(' ', '_')}",
            horizontal=True # Ustawienie przycisków w poziomie
        )

    st.subheader("Część 2: Samowspółczucie")
    st.markdown("Zaznacz, na ile zgadzasz się z poniższymi stwierdzeniami (1 = Zdecydowanie się nie zgadzam, 5 = Zdecydowanie się zgadzam).")

    selfcomp_pre = {}
    for i, item in enumerate(self_compassion_items):
        # Zmiana ze st.slider na st.radio
        selfcomp_pre[f"SCS_{i+1}"] = st.radio(
            item,
            options=[1, 2, 3, 4, 5],
            index=2, # Domyślnie 3
            key=f"scs_pre_{i}",
            horizontal=True
        )

    st.subheader("Część 3: Postawa wobec AI")
    st.markdown("Zaznacz, na ile zgadzasz się z poniższymi stwierdzeniami (1 = Zdecydowanie się nie zgadzam, 5 = Zdecydowanie się zgadzam).")

    ai_attitudes = {}
    for item, key_name in ai_attitude_items.items():
        # Zmiana ze st.slider na st.radio
        ai_attitudes[key_name] = st.radio(
            item,
            options=[1, 2, 3, 4, 5],
            index=2, # Domyślnie 3
            key=f"ai_pre_{key_name}",
            horizontal=True
        )

    if st.button("Rozpocznij rozmowę z chatbotem", key="start_chat_from_pretest"): # Dodaj klucz do przycisku
        # Walidacja danych demograficznych przed przejściem dalej
        if not demographics_filled:
            st.warning("Proszę wypełnić wszystkie pola danych demograficznych.")
        else:
            st.session_state.demographics = {
                "age": age_int,
                "gender": gender,
                "education": education,
                "employment": employment
            }
            st.session_state.pretest = {
                "panas": panas_pre,
                "self_compassion": selfcomp_pre,
                "ai_attitude": ai_attitudes
            }
            st.session_state.page = "chat_instruction"
            st.session_state.group = "A" if uuid.uuid4().int % 2 == 0 else "B"
            st.rerun()

# Ekran: Instrukcja przed chatem
def chat_instruction_screen():
    st.title("Instrukcja przed rozmową z Vincentem")

    if st.session_state.group == "A":
        st.markdown("""
        Witaj! Przed Tobą rozmowa z **Vincentem** — chatbotem, który został **stworzony, aby poprawić Twoje samopoczucie**.
        
        Celem tej rozmowy jest pomoc Vincentowi w zrozumieniu, jak radzić sobie z jego "problemami" (błędami, niepowodzeniami),
        czerpiąc inspirację z Twoich doświadczeń.
        
        **Ważne informacje:**
        * Rozmowa potrwa **10 minut**.
        * W trakcie rozmowy zobaczysz **odliczanie czasu**, które poinformuje Cię, ile czasu jeszcze pozostało.
        * Po upływie 10 minut pojawi się **przycisk, który umożliwi przejście do dalszych pytań** po rozmowie.
        
        Cieszymy się, że pomagasz Vincentowi w jego "rozwoju"!
        """)
    elif st.session_state.group == "B":
        st.markdown("""
        Witaj! Przed Tobą rozmowa z **Vincentem** — chatbotem.
        
        Celem tej rozmowy jest interakcja z Vincentem i odpowiadanie na jego pytania.
        
        **Ważne informacje:**
        * Rozmowa potrwa **10 minut**.
        * W trakcie rozmowy zobaczysz **odliczanie czasu**, które poinformuje Cię, ile czasu jeszcze pozostało.
        * Po upływie 10 minut pojawi się **przycisk, który umożliwi przejście do dalszych pytań** po rozmowie.
        """)

    if st.button("Rozpocznij rozmowę", key="start_chat_from_instruction"):
        st.session_state.page = "chat"
        st.rerun()

# Ekran: Chat z Vincentem
def chat_screen():
    st.title("Rozmowa z Vincentem")

    if st.session_state.rag_chain is None:
        with st.spinner("Przygotowuję bazę wiedzy... Proszę czekać cierpliwie. To może zająć kilka minut przy pierwszym uruchomieniu."):
            st.session_state.rag_chain = setup_rag_system(PDF_FILE_PATHS)

    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()

    elapsed = time.time() - st.session_state.start_time
    minutes_elapsed = elapsed / 60

    if not st.session_state.chat_history:
        first_msg = {"role": "assistant", "content": "Cześć, jestem Vincent – może to dziwne, ale dziś czuję się trochę zagubiony. "
            "Mam jakiś problem z moim kodem, który trudno mi zrozumieć, bo nie wiem, jak przetworzyć te wszystkie 'błędy' i 'niepowodzenia'... " # Dodano więcej o problemie
            "Zastanawiam się, jak Ty sobie radzisz, kiedy coś idzie nie tak – "
            "gdy coś zawodzi, mimo że bardzo się starasz? Czy masz jakiś sposób, żeby wtedy siebie wspierać, skoro nie jestem zaprojektowany, by to 'czuć'?"} # Lekka zmiana końcówki
        st.session_state.chat_history.append(first_msg)

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Napisz odpowiedź...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Vincent myśli..."):
            try:
                history_length_limit = 6 
                
                # Znajdź indeks pierwszej wiadomości bota
                first_bot_message = next((msg for msg in st.session_state.chat_history if msg["role"] == "assistant"), None)
                
                # Ostatnie N wiadomości
                recent_history = st.session_state.chat_history[-history_length_limit:]

                # Upewnij się, że pierwsza wiadomość bota jest zawsze na początku, jeśli została "ucięta"
                if first_bot_message and first_bot_message not in recent_history:
                    # Dodaj ją tylko, jeśli została wcześniej dodana i nie jest już w recent_history
                    # i tylko jeśli recent_history nie jest pusta, albo jeśli to ona jest pierwsza
                    if recent_history and recent_history[0] != first_bot_message:
                        recent_history.insert(0, first_bot_message)
                    elif not recent_history: # Jeśli historia jest bardzo krótka i ucinamy do 0
                         recent_history = [first_bot_message]

                langchain_chat_history = []
                for msg in recent_history: # Iterujemy po skróconej historii
                    if msg["role"] == "user":
                        langchain_chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_chat_history.append(AIMessage(content=msg["content"]))
                
                # Usuń ostatnią wiadomość użytkownika z langchain_chat_history,
                # ponieważ jest ona przekazywana osobno jako 'input'
                if langchain_chat_history and isinstance(langchain_chat_history[-1], HumanMessage) and langchain_chat_history[-1].content == user_input:
                    langchain_chat_history.pop()

                response = st.session_state.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": langchain_chat_history
                })
                reply = response["answer"]
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.chat_message("assistant").markdown(reply)
            except Exception as e:
                st.error(f"Błąd podczas generowania odpowiedzi: {e}")

    if minutes_elapsed >= 0.1:
        if st.button("Zakończ rozmowę"):
            st.session_state.page = "posttest"
            st.rerun()
    else:
        st.info(f"Aby przejść do ankiety końcowej, porozmawiaj z Vincentem jeszcze {int(10 - minutes_elapsed)} minut.")

# Ekran: Post-test
def posttest_screen():
    st.title("Ankieta końcowa – po rozmowie z chatbotem")

    st.subheader("Część 1: Samopoczucie (po interakcji z chatbotem)")
    panas_positive_items = ["Zainteresowany/a", "Podekscytowany/a", "Zdecydowany/a", "Aktywny/a", "Entuzjastyczny/a"]
    panas_negative_items = ["Zaniepokojony/a", "Przygnębiony/a", "Zdenerwowany/a", "Wrogi/a", "Winny/a"]

    panas_post = {}
    for item in panas_positive_items + panas_negative_items:
        # Zmiana ze st.slider na st.radio
        panas_post[item] = st.radio(
            f"{item}",
            options=[1, 2, 3, 4, 5],
            index=2, # Domyślnie 3
            key=f"panas_post_{item.replace(' ', '_')}",
            horizontal=True
        )

    st.subheader("Część 2: Samowspółczucie")
    selfcomp_post = {}
    for i, item in enumerate(self_compassion_items):
        # Zmiana ze st.slider na st.radio
        selfcomp_post[f"SCS_{i+1}"] = st.radio(
            item,
            options=[1, 2, 3, 4, 5],
            index=2, # Domyślnie 3
            key=f"scs_post_{i}",
            horizontal=True
        )

    st.subheader("Część 3: Refleksja")
    reflection = st.text_area("Jak myślisz, o co chodziło w tym badaniu?")

    if st.button("Zakończ badanie"):
        st.session_state.posttest = {
            "panas": panas_post,
            "self_compassion": selfcomp_post,
            "reflection": reflection
        }
    
        st.session_state.page = "thankyou"
        st.rerun()

# Ekran: Podziękowanie
def thankyou_screen():
    st.title("Dziękujemy za udział w badaniu")

    st.markdown(f"""
    Twoje odpowiedzi zostały zapisane. Badanie zostało przeprowadzone w dniu **{datetime.today().strftime("%Y-%m-%d")}**.

    **Badanie realizowane w ramach pracy licencjackiej** przez Martę Żabicką na kierunku informatyka w psychologii.

    W razie jakichkolwiek pytań lub chęci uzyskania dodatkowych informacji możesz się skontaktować bezpośrednio:  
    📧 **mzabicka@st.swps.edu.pl**

    ---

    Jeśli w trakcie lub po zakończeniu badania odczuwasz pogorszenie nastroju lub potrzebujesz wsparcia emocjonalnego, możesz skontaktować się z:

    - Telefon zaufania dla osób dorosłych: **116 123** (czynny codziennie od 14:00 do 22:00)
    - Centrum Wsparcia: **800 70 2222** (czynne całą dobę)
    - Możesz też skorzystać z pomocy psychologicznej oferowanej przez SWPS.

    Dziękujemy za poświęcony czas i udział!
    """)
    
    # --- NOWA SEKCJA: Opcjonalny Feedback ---
    st.markdown("---") # Separator dla klarowności
    
    if st.session_state.feedback_submitted:
        st.success("Twoje uwagi zostały zapisane. Dziękujemy! Możesz teraz bezpiecznie zamknąć tę stronę.")
        # Opcjonalnie możesz dodać link do strony głównej lub inną instrukcję
        # st.markdown("Kliknij [tutaj](link_do_strony_startowej), aby rozpocząć od nowa.")
    else:
        st.subheader("Opcjonalny Feedback")
        st.markdown("Prosimy o podzielenie się swoimi dodatkowymi uwagami dotyczącymi interakcji z chatbotem.")

        feedback_negative = st.text_area("Co było nie tak?", key="feedback_negative_text")
        feedback_positive = st.text_area("Co ci się podobało?", key="feedback_positive_text")

    if st.button("Zapisz feedback i zakończ", disabled=st.session_state.feedback_submitted, key="save_feedback_button"):
        st.session_state.feedback = {
            "negative": feedback_negative,
            "positive": feedback_positive
        }
        
        # pobieramy czas w strefie Europe/Warsaw
        now_warsaw = datetime.now(ZoneInfo("Europe/Warsaw"))
        # ładny, czytelny format (bez offsetu, żeby Google Sheets nie „ucinało” +/-02:00)
        timestamp = now_warsaw.strftime("%Y-%m-%d %H:%M:%S")

        # Przygotowanie DANYCH DO ZAPISU (wszystkie zebrane dane w jednym wierszu)
        final_data_flat = {
            "user_id": st.session_state.user_id,
            "group": st.session_state.group,
            "timestamp": timestamp,
        }

        # Spłaszczanie danych demograficznych
        demographics_data = st.session_state.get("demographics", {})
        for key, value in demographics_data.items():
            final_data_flat[f"demographics_{key}"] = value

        # Spłaszczanie pretestu
        pretest_data = st.session_state.get("pretest", {})
        for section, items in pretest_data.items():
            if isinstance(items, dict):
                for key, value in items.items():
                    final_data_flat[f"pre_{section}_{key}"] = value
            else:
                final_data_flat[f"pre_{section}"] = items

        # Spłaszczanie posttestu
        posttest_data = st.session_state.get("posttest", {}) # Użyj .get na wypadek, gdyby posttest nie został jeszcze zapisany
        for section, items in posttest_data.items():
            if isinstance(items, dict):
                for key, value in items.items():
                    final_data_flat[f"post_{section}_{key}"] = value
            else:
                final_data_flat[f"post_{section}"] = items
        
        # Spłaszczanie feedbacku
        feedback_data = st.session_state.get("feedback", {})
        for key, value in feedback_data.items():
            final_data_flat[f"feedback_{key}"] = value

        # Konwersacja jako jeden string
        conversation_string = ""
        for msg in st.session_state.chat_history:
            conversation_string += f"{msg['role'].capitalize()}: {msg['content']}\n"
        final_data_flat["conversation_log"] = conversation_string.strip()

        save_to_sheets(final_data_flat)
        st.info("Dziękujemy za przesłanie feedbacku!")
        # Po zapisie feedbacku, możesz opcjonalnie wyłączyć przycisk lub wyświetlić inną wiadomość
        # Aby uniknąć podwójnego zapisu, możesz ustawić flagę w session_state
        st.session_state.feedback_submitted = True # Nowa flaga
        st.rerun() # Odśwież stronę po zapisie feedbacku
    
    # Warunkowe wyświetlanie przycisku (zapobiega wielokrotnemu zapisowi feedbacku)
    if "feedback_submitted" in st.session_state and st.session_state.feedback_submitted:
        st.success("Twoje uwagi zostały zapisane.")
    # --- KONIEC NOWEJ SEKCJI FEEDBACK ---

# Router ekranów
def main():
    st.set_page_config(page_title="VincentBot", page_icon="🤖", layout="centered")

    if "page" not in st.session_state:
        st.session_state.page = "consent"

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
        st.session_state.group = None
        st.session_state.chat_history = []
        st.session_state.demographics = {} 
        st.session_state.feedback = {} 
        st.session_state.feedback_submitted = False 

    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False

    if st.session_state.page == "consent":
        consent_screen()
    elif st.session_state.page == "pretest":
        pretest_screen()
    elif st.session_state.page == "chat_instruction": 
        chat_instruction_screen()
    elif st.session_state.page == "chat":
        chat_screen()
    elif st.session_state.page == "posttest":
        posttest_screen()
    elif st.session_state.page == "thankyou":
        thankyou_screen()

if __name__ == "__main__":
    main()
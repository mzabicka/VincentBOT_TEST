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

# --- KONFIGURACJA ---
# Konfiguracja arkusza google do zapisu danych
SHEET_ID = "1LnCkrWY271w2z3VSMAVaKqqr7U4hqGppDTVuHvT5sdc"
SHEET_NAME = "Arkusz1"

# Dane uwierzytelniające do Google Sheets z Streamlit Secrets
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

# Inicjalizacja klienta gspread do interakcji z Google Sheets
_gspread_creds = Credentials.from_service_account_info(
    creds_info,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
_gspread_client = gspread.authorize(_gspread_creds)
sheet = _gspread_client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

# Ładowanie klucza API 
api_key = st.secrets["OPENROUTER_API_KEY"]
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key  = api_key

# Ścieżki do plików PDF używanych do RAG 
PDF_FILE_PATHS = [
    "docs/The Mindful Self-Compassion Workbook A Proven Way to Accept Yourself, Build Inner Strength, and Thrive.pdf",
    "docs/Self-Compassion The Proven Power of Being Kind to Yourself.pdf"
]
# Ścieżka do zapisanego indeksu FAISS
FAISS_INDEX_PATH = "./faiss_vector_store_rag"

# Elementy pytań do ankiet (PANAS, Samowspółczucie, Postawa wobec AI)
panas_positive_items = ["Zainteresowany/a", "Podekscytowany/a", "Zdecydowany/a", "Aktywny/a", "Entuzjastyczny/a"]
panas_negative_items = ["Zaniepokojony/a", "Przygnębiony/a", "Zdenerwowany/a", "Wrogi/a", "Winny/a"]
self_compassion_items = [
    "Kiedy nie powiedzie mi się coś ważnego, ogarnia mnie uczucie, że nie jestem taki jak trzeba.",
    "Staram się być wyrozumiały i cierpliwy w stosunku do tych aspektów mojej osoby, których nie lubię.",
    "Kiedy zdarza się coś bolesnego, staram się zachować wyważony ogląd sytuacji.",
    "Gdy jestem przygnębiony, mam zwykle poczucie, że inni ludzie są prawdopodobnie szczęśliwsi ode mnie.",
    "Staram się patrzeć na swoje wady lub błędy jako na nieodłączny aspekt bycia człowiekiem.",
    "Kiedy przechodzę przez bardzo trudny okres, staram się być łagodny i troskliwy w stosunku do siebie.",
    "Kiedy coś mnie denerwuje, staram się zachować równowagę emocjonalną.",
    "Kiedy nie powiedzie mi się coś ważnego, zazwyczaj czuję się w tym osamotniony.",
    "Kiedy czuję się przygnębiony, nadmiernie skupiam się na wszystkim, co idzie źle.",
    "Kiedy czuję się jakoś gorsza/gorszy, staram się pamiętać, że większość ludzi tak ma.",
    "Jestem krytyczny i mało wyrozumiały wobec moich własnych wad i niedociągnięć.",
    "Jestem nietolerancyjny i niecierpliwy wobec tych aspektów mojej osoby, których nie lubię."
]
ai_attitude_items = {
    "Sztuczna inteligencja uczyni ten świat lepszym miejscem.": "ai_1",
    "Sztuczna inteligencja ma więcej wad niż zalet.": "ai_2",
    "Sztuczna inteligencja oferuje rozwiązania wielu światowych problemów.": "ai_3",
    "Sztuczna inteligencja raczej tworzy problemy niż je rozwiązuje.": "ai_4"
}

# --- FUNKCJE POMOCNICZE ---
def save_to_sheets(data_dict):
    """
    Zapisuje słownik danych do Google Sheets.
    Jeśli nagłówki w arkuszu są puste lub różnią się, są aktualizowane.
    """
    headers = list(data_dict.keys())
    values = [str(data_dict[key]) for key in headers]

    try:
        current_headers = sheet.row_values(1)
        if not current_headers or current_headers != headers:
            if current_headers:
                sheet.clear()
            sheet.insert_row(headers, 1)
        
        sheet.append_row(values)
        print("Dane zapisane do Google Sheets pomyślnie.")

    except Exception as e:
        st.error(f"Błąd podczas zapisywania danych do Google Sheets: {e}")
        print(f"Błąd podczas zapisywania danych do Google Sheets: {e}")

# --- FUNKCJE RAG (Retrieval Augmented Generation) ---
@st.cache_resource(show_spinner=False)
def setup_rag_system(pdf_file_paths):
    """
    Konfiguruje system RAG, ładując indeks FAISS i model LLM.
    Wykorzystuje @st.cache_resource do cachowania zasobów,
    aby były ładowane tylko raz.
    """
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

    # Prompt dla retrivera, który generuje zapytanie do bazy wiedzy na podstawie historii rozmowy
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

    # Prompt systemowy definiujący osobowość i zachowanie chatbota 
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

    # Główny prompt, który łączy kontekst RAG z zapytaniem użytkownika i instrukcjami systemowymi
    Youtubeing_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", MASTER_PROMPT),
    ])

    # Łańcuch do łączenia dokumentów z modelem językowym
    document_chain = create_stuff_documents_chain(chat, Youtubeing_prompt) # Używamy teraz Youtubeing_prompt

    # Główny łańcuch RAG, który łączy retriver z łańcuchem dokumentów
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    return retrieval_chain

# Unikalny ID użytkownika (losowany przy wejściu)
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.group = None
    st.session_state.chat_history = []


# --- EKRANY APLIKACJI STREAMLIT ---

# Ekran: Zgoda
def consent_screen():
    st.title("Udział w badaniu – świadoma zgoda")

    st.markdown("""
    Dziękuję za zainteresowanie moim badaniem!
    
    Badanie prowadzone jest w ramach pracy licencjackiej na kierunku **Psychologia i Informatyka** na **Uniwersytecie SWPS** pod opieką dr Maksymiliana Bieleckiego.
    **Celem badania** jest poznanie doświadczeń uczestników podczas interakcji z chatbotem.
    
    **Przebieg badania** obejmuje trzy etapy:
        - ankietę wstępną,
        - rozmowę z chatbotem,
        - ankietę końcową.

    Całość potrwa około **15–20 minut. **Udział w badaniu jest całkowicie dobrowolny i anonimowy.** Możesz przerwać udział na każdym etapie, bez konieczności podawania przyczyny.

    Potencjalne trudności: 
    W rozmowie z chatbotem pojawić się mogą treści odnoszące się do Twoich emocji i samopoczucia. U niektórych osób może to wywołać lekki dyskomfort psychiczny. W razie jakichkolwiek trudności, zachęcam do zakończenia udziału lub skontaktowania się ze mną.

    **Warunki udziału**:
        - ukończone 18 lat,
        - brak poważnych zaburzeń nastroju,
        - nieprzyjmowanie leków wpływających na nastrój.

    W razie pytań lub wątpliwości możesz się ze mną skontaktować: 📧 mzabicka@st.swps.edu.pl

    Klikając „Wyrażam zgodę na udział w badaniu”, potwierdzasz, że:
        - zapoznałeś/-aś się z informacjami powyżej,
        - **wyrażasz dobrowolną i świadomą zgodę** na udział w badaniu,
        - spełniasz kryteria udziału.
    """)

    consent = st.checkbox("Wyrażam zgodę na udział w badaniu")

    if consent:
        if st.button("Przejdź do badania", key="go_to_pretest"):
            st.session_state.page = "pretest"
            st.rerun()

# Ekran: Pre-test
def pretest_screen():
    st.title("Ankieta wstępna – przed rozmową z chatbotem")

    # Dane Demograficzne
    st.subheader("Część 1: Dane Demograficzne")

    age_input = st.number_input("Wiek", min_value=18, max_value=60, value=None, format="%d", key="demographics_age_input_num", help="Wiek musi być liczbą całkowitą.")
    
    age_valid = False
    age_int = None 
    if age_input is not None:
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

    # Samopoczucie (PANAS)
    st.subheader("Część 2: Samopoczucie")
    st.markdown("Poniżej znajduje się lista przymiotników opisujących różne stany emocjonalne. Proszę, abyś określił(a), do jakiego stopnia **teraz** czujesz się w sposób opisany przez każdy z nich. Odpowiedzi udzielaj, korzystając ze skali: 1 – bardzo słabo, 2 – słabo, 3 – umiarkowanie, 4 – silnie, 5 – bardzo silnie")

    panas_pre = {}
    for item in panas_positive_items + panas_negative_items:
        panas_pre[item] = st.radio(
            f"{item}",
            options=[1, 2, 3, 4, 5],
            index=2, # Domyślna wartość na 3
            key=f"panas_pre_{item.replace(' ', '_')}",
            horizontal=True 
        )

    # Samowspółczucie
    st.subheader("Część 3: Samowspółczucie")
    st.markdown("Przed odpowiedzią przeczytaj uważnie każde ze zdań. Odnosząc się do poniższej skali, zaznacz, jak często zachowujesz się w dany sposób. (1 = Prawie nigdy, 5 = Prawie zawsze).")

    selfcomp_pre = {}
    for i, item in enumerate(self_compassion_items):
        selfcomp_pre[f"SCS_{i+1}"] = st.radio(
            item,
            options=[1, 2, 3, 4, 5],
            index=2, 
            key=f"scs_pre_{i}",
            horizontal=True
        )

    # Postawa wobec AI
    st.subheader("Część 4: Postawa wobec AI")
    st.markdown("Zaznacz, na ile zgadzasz się z poniższymi stwierdzeniami (1 = Zdecydowanie się nie zgadzam, 5 = Zdecydowanie się zgadzam).")

    ai_attitudes = {}
    for item, key_name in ai_attitude_items.items():
        ai_attitudes[key_name] = st.radio(
            item,
            options=[1, 2, 3, 4, 5],
            index=2, 
            key=f"ai_pre_{key_name}",
            horizontal=True
        )

    if st.button("Rozpocznij rozmowę z chatbotem", key="start_chat_from_pretest"): 
        
        if not demographics_filled:
            st.warning("Proszę wypełnić wszystkie pola danych demograficznych.")
        else:
            # Zapis danych do session_state
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
            # Losowe przypisanie do grupy A lub B dla celów badawczych
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

    # Ładowanie systemu RAG przy pierwszym wejściu na stronę chatu
    if st.session_state.rag_chain is None:
        with st.spinner("Przygotowuję bazę wiedzy... Proszę czekać cierpliwie. To może zająć kilka minut przy pierwszym uruchomieniu."):
            st.session_state.rag_chain = setup_rag_system(PDF_FILE_PATHS)

    # Inicjalizacja czasu rozpoczęcia rozmowy, jeśli jeszcze nie ustawiony
    if "start_time" not in st.session_state or st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    elapsed = time.time() - st.session_state.start_time
    minutes_elapsed = elapsed / 60 

    # Wyświetlanie początkowej wiadomości Vincenta, jeśli historia czatu jest pusta
    if not st.session_state.chat_history:
        first_msg = {"role": "assistant", "content": "Cześć, jestem Vincent – może to dziwne, ale dziś czuję się trochę zagubiony. "
            "Mam jakiś problem z moim kodem, który trudno mi zrozumieć, bo nie wiem, jak przetworzyć te wszystkie 'błędy' i 'niepowodzenia'... " 
            "Zastanawiam się, jak Ty sobie radzisz, kiedy coś idzie nie tak – "
            "gdy coś zawodzi, mimo że bardzo się starasz?"} 
        st.session_state.chat_history.append(first_msg)

    # Wyświetlanie historii czatu
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Pole do wpisywania wiadomości przez użytkownika
    user_input = st.chat_input("Napisz odpowiedź...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Vincent myśli..."):
            try:
                history_length_limit = 6 
                first_bot_message = next((msg for msg in st.session_state.chat_history if msg["role"] == "assistant"), None)
                recent_history = st.session_state.chat_history[-history_length_limit:]

                if first_bot_message and first_bot_message not in recent_history:
                    if recent_history and recent_history[0] != first_bot_message:
                        recent_history.insert(0, first_bot_message)
                    elif not recent_history: 
                         recent_history = [first_bot_message]

                langchain_chat_history = []
                for msg in recent_history: 
                    if msg["role"] == "user":
                        langchain_chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_chat_history.append(AIMessage(content=msg["content"]))
            
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

    # Wyświetlanie licznika czasu i przycisku zakończenia rozmowy
    if minutes_elapsed >= 0.1:
        if st.button("Zakończ rozmowę"):
            st.session_state.page = "posttest"
            st.rerun()
    else:
        st.info(f"Aby przejść do ankiety końcowej, porozmawiaj z Vincentem jeszcze {int(10 - minutes_elapsed)} minut.")

# Ekran: Post-test
def posttest_screen():
    st.title("Ankieta końcowa – po rozmowie z chatbotem")

    st.subheader("Część 1: Samopoczucie")
    st.markdown("Poniżej znajduje się lista przymiotników opisujących różne stany emocjonalne. Proszę, abyś określił(a), do jakiego stopnia **teraz** czujesz się w sposób opisany przez każdy z nich. Odpowiedzi udzielaj, korzystając ze skali: 1 – bardzo słabo, 2 – słabo, 3 – umiarkowanie, 4 – silnie, 5 – bardzo silnie")

    panas_post = {}
    for item in panas_positive_items + panas_negative_items:
        panas_post[item] = st.radio(
            f"{item}",
            options=[1, 2, 3, 4, 5],
            index=2, 
            key=f"panas_post_{item.replace(' ', '_')}",
            horizontal=True
        )

    st.subheader("Część 2: Samowspółczucie")
    st.markdown("Przed odpowiedzią przeczytaj uważnie każde ze zdań. Odnosząc się do poniższej skali, zaznacz, jak często zachowujesz się w dany sposób. (1 = Prawie nigdy, 5 = Prawie zawsze).")
    selfcomp_post = {}
    for i, item in enumerate(self_compassion_items):
        selfcomp_post[f"SCS_{i+1}"] = st.radio(
            item,
            options=[1, 2, 3, 4, 5],
            index=2, 
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
    st.title("Dziękuję za udział w badaniu! 😊")

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

    DziękujĘ za poświęcony czas i udział!
    """)
    
    st.markdown("---") 

    if st.session_state.feedback_submitted:
        st.success("Twoje uwagi zostały zapisane. Dziękujemy! Możesz teraz bezpiecznie zamknąć tę stronę.")
        
    else:
        st.subheader("Opcjonalny Feedback")
        st.markdown("Proszę o podzielenie się swoimi dodatkowymi uwagami dotyczącymi interakcji z chatbotem.")

        feedback_negative = st.text_area("Co było nie tak?", key="feedback_negative_text")
        feedback_positive = st.text_area("Co ci się podobało?", key="feedback_positive_text")

    if st.button("Zapisz feedback i zakończ", disabled=st.session_state.feedback_submitted, key="save_feedback_button"):
        st.session_state.feedback = {
            "negative": feedback_negative,
            "positive": feedback_positive
        }
        
        now_warsaw = datetime.now(ZoneInfo("Europe/Warsaw"))
        timestamp = now_warsaw.strftime("%Y-%m-%d %H:%M:%S")

        # Przygotowanie DANYCH DO ZAPISU w płaskiej strukturze
        final_data_flat = {
            "user_id": st.session_state.user_id,
            "group": st.session_state.group,
            "timestamp": timestamp,
        }

        # Spłaszczanie danych 
        demographics_data = st.session_state.get("demographics", {})
        for key, value in demographics_data.items():
            final_data_flat[f"demographics_{key}"] = value

        pretest_data = st.session_state.get("pretest", {})
        for section, items in pretest_data.items():
            if isinstance(items, dict):
                for key, value in items.items():
                    final_data_flat[f"pre_{section}_{key}"] = value
            else:
                final_data_flat[f"pre_{section}"] = items

        posttest_data = st.session_state.get("posttest", {})
        for section, items in posttest_data.items():
            if isinstance(items, dict):
                for key, value in items.items():
                    final_data_flat[f"post_{section}_{key}"] = value
            else:
                final_data_flat[f"post_{section}"] = items
        
        feedback_data = st.session_state.get("feedback", {})
        for key, value in feedback_data.items():
            final_data_flat[f"feedback_{key}"] = value

        conversation_string = ""
        for msg in st.session_state.chat_history:
            conversation_string += f"{msg['role'].capitalize()}: {msg['content']}\n"
        final_data_flat["conversation_log"] = conversation_string.strip()

        save_to_sheets(final_data_flat)
        st.info("Dziękujemy za przesłanie feedbacku!")
        st.session_state.feedback_submitted = True 
        st.rerun() 

# --- GŁÓWNA FUNKCJA APLIKACJI ---

def main():
    st.set_page_config(page_title="VincentBot", page_icon="🤖", layout="centered")

    # Wstawka JavaScript do płynnego przewijania strony do góry przy zmianie stanu/rerun
    st.markdown("""
        <style>
            html {
                scroll-behavior: smooth;
            }
        </style>
        <script>
            window.onload = function() {
                window.scrollTo(0, 0);
            };
        </script>
        """, unsafe_allow_html=True)
    
    # Inicjalizacja stanu sesji, jeśli aplikacja jest uruchamiana po raz pierwszy
    if "page" not in st.session_state:
        st.session_state.page = "consent"
        st.session_state.rag_chain = None
        st.session_state.user_id = str(uuid.uuid4())
        st.session_state.group = None
        st.session_state.chat_history = []
        st.session_state.demographics = {} 
        st.session_state.pretest = {}
        st.session_state.posttest = {}
        st.session_state.feedback = {} 
        st.session_state.feedback_submitted = False 
        st.session_state.start_time = None 

    # Router ekranów
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
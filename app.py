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
import random

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
panas_positive_items = ["Zainteresowany/a", "Zainspirowany/a", "Spokojny/a", "Aktywny/a", "Entuzjastyczny/a"]
panas_negative_items = ["Zaniepokojony/a", "Przygnębiony/a", "Zestresowany/a", "Nerwowy/a", "Drażliwy/a"]
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
    Akumuluje i zapisuje słownik danych do Google Sheets w jednym wierszu dla danego user_id.
    Dodaje nowe kolumny, jeśli brakuje ich w arkuszu, BEZ CZYSZCZENIA istniejących danych.
    Jeśli user_id już istnieje, wiersz jest aktualizowany o nowe dane,
    zachowując istniejące, jeśli nie zostały przesłane nowe wartości.
    Jeśli user_id nie istnieje, tworzony jest nowy wiersz.
    """
    user_id = data_dict.get("user_id")
    if not user_id:
        st.error("Błąd: Próba zapisu danych bez user_id. Proszę odświeżyć stronę lub skontaktować się z badaczem.")
        print("Błąd: Próba zapisu danych bez user_id. Dane nie zostały zapisane.")
        return

    try:
        current_headers = sheet.row_values(1) # Pobierz nagłówki z pierwszej kolumny
        
        # Stwórz listę wszystkich POTENCJALNYCH nagłówków, które powinny być w arkuszu
        # Zaczynamy od obecnych nagłówków, potem dodajemy te z data_dict, których jeszcze nie ma.
        all_potential_headers = list(current_headers)
        for key in data_dict.keys():
            if key not in all_potential_headers:
                all_potential_headers.append(key)
        
        # Jeśli arkusz jest pusty, wstaw wszystkie nagłówki od razu
        if not current_headers:
            sheet.insert_row(all_potential_headers, 1)
            print(f"Początkowe nagłówki ustawione: {all_potential_headers}")
            current_headers = all_potential_headers # Uaktualnij nagłówki po wstawieniu
        else:
            # Sprawdź, czy brakuje jakichś nagłówków z data_dict w obecnych nagłówkach arkusza
            headers_to_add = [h for h in all_potential_headers if h not in current_headers]
            
            if headers_to_add:
                # Dodaj brakujące kolumny na koniec arkusza
                # W gspread najbezpieczniej to zrobić, wstawiając nową listę nagłówków do 1. wiersza
                # Zostawiamy istniejące dane w spokoju, tylko nagłówki się przesuwają
                
                # Pobieramy wszystkie dane z arkusza (oprócz nagłówków)
                all_records = sheet.get_all_records() # Pobiera dane jako listę słowników
                
                # Czyścimy arkusz TYLKO RAZ, żeby wstawić zaktualizowane nagłówki
                # Jest to bezpieczne, bo wcześniej pobraliśmy wszystkie dane
                sheet.clear() 
                sheet.insert_row(all_potential_headers, 1) # Wstawiamy zaktualizowane nagłówki
                print(f"Nagłówki arkusza zaktualizowane. Dodano: {headers_to_add}")
                
                # Wstawiamy z powrotem wszystkie poprzednie dane (jeśli jakieś były)
                if all_records:
                    # Konwertujemy listę słowników z powrotem na listę list,
                    # upewniając się, że kolejność kolumn jest zgodna z nowymi nagłówkami
                    rows_to_insert = []
                    for record in all_records:
                        row = [str(record.get(h, "")) for h in all_potential_headers]
                        rows_to_insert.append(row)
                    sheet.append_rows(rows_to_insert)
                    print(f"Wstawiono ponownie {len(rows_to_insert)} wierszy danych.")

                current_headers = all_potential_headers # Uaktualnij nagłówki po wstawieniu

        # 2. Znajdź wiersz użytkownika lub dodaj nowy
        user_ids_in_sheet = []
        user_id_col_index = -1
        
        if "user_id" in current_headers:
            user_id_col_index = current_headers.index("user_id") + 1
            # sheet.col_values(user_id_col_index) zwróci listę wartości z kolumny user_id
            # [1:] pomija nagłówek
            # Jeśli kolumna jest pusta (oprócz nagłówka), user_ids_in_sheet będzie pusta
            user_ids_in_sheet = sheet.col_values(user_id_col_index)[1:] 
        else:
            # Jeśli kolumna 'user_id' w ogóle nie istnieje, to znaczy, że arkusz jest nowy
            # lub został właśnie wyczyszczony i nagłówki zostały wstawione.
            # W tym przypadku, user_id_col_index pozostaje -1, a nowy wiersz zostanie dodany.
            print("Kolumna 'user_id' nie znaleziona. Zostanie dodany nowy wiersz.")

        row_index = -1
        if user_id_col_index != -1 and user_id in user_ids_in_sheet:
            row_index = user_ids_in_sheet.index(user_id) + 2 # +1 dla nagłówka, +1 bo lista jest 0-bazowa
        
        if row_index != -1:
            # Użytkownik istnieje, pobierz jego obecne dane
            existing_row_values = sheet.row_values(row_index)
            
            # Stwórz słownik z istniejących danych, żeby łatwo je scalić
            existing_data_map = {}
            for i, header in enumerate(current_headers):
                if i < len(existing_row_values):
                    existing_data_map[header] = existing_row_values[i]
                else:
                    existing_data_map[header] = "" # Uzupełnij puste dla nowo dodanych kolumn (jeśli dodano nowe kolumny, a ten wiersz był już wcześniej)

            # Scal nowe dane z istniejącymi (nowe nadpisują stare dla tych samych kluczy, reszta zostaje)
            merged_data = {**existing_data_map, **data_dict}
            
            # Przygotuj wiersz do aktualizacji w prawidłowej kolejności nagłówków
            row_to_update = [str(merged_data.get(header, "")) for header in current_headers]
            sheet.update(f"A{row_index}", [row_to_update])
            print(f"Dane dla user_id {user_id} zaktualizowane w Google Sheets pomyślnie w wierszu {row_index}.")
        else:
            # Użytkownik nie istnieje, dodaj nowy wiersz
            # Upewnij się, że dodajesz wartości w kolejności current_headers
            new_row_values = [str(data_dict.get(header, "")) for header in current_headers]
            sheet.append_row(new_row_values)
            print(f"Nowe dane dla user_id {user_id} dodane do Google Sheets pomyślnie (nowy wiersz).")

    except gspread.exceptions.APIError as api_e:
        st.error(f"Błąd API Google Sheets: {api_e}. Sprawdź uprawnienia konta serwisowego i limit zapytań.")
        print(f"Błąd API Google Sheets: {api_e}")
    except Exception as e:
        st.error(f"Krytyczny błąd podczas zapisu danych do Google Sheets: {e}. Proszę skontaktuj się z badaczem.")
        print(f"Krytyczny błąd podczas zapisu danych do Google Sheets: {e}")

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

    # Pobieranie płci użytkownika z session_state
    user_gender = st.session_state.get("demographics", {}).get("gender")
    gender_instruction = ""

    if user_gender == "Kobieta":
        gender_instruction = (
            "Użytkownik, z którym rozmawiasz, to **kobieta**. "
            "Zwracaj się do niej w formie żeńskiej (np. 'jak sobie poradziłaś', 'zastanawiałam się', 'czy próbowałaś'). "
            "Używaj odpowiednich zaimków i końcówek fleksyjnych."
        )
    elif user_gender == "Mężczyzna":
        gender_instruction = (
            "Użytkownik, z którym rozmawiasz, to **mężczyzna**. "
            "Zwracaj się do niego w formie męskiej (np. 'jak sobie poradziłeś', 'zastanawiałem się', 'czy próbowałeś'). "
            "Używaj odpowiednich zaimków i końcówek fleksyjnych."
        )
    else: # Inna, Nie chcę podać, lub brak danych
        gender_instruction = (
            "Użytkownik, z którym rozmawiasz, nie określił płci w sposób binarny lub wolał jej nie podawać. "
            "Zwracaj się do użytkownika w formie neutralnej, bezosobowej lub zaimkami uniwersalnymi (np. 'jak sobie radzisz', 'jak się czujesz', 'zastanawiam się', 'czy próbujesz'). "
            "Unikaj form gramatycznych specyficznych dla płci."
        )

    # Prompt systemowy definiujący osobowość i zachowanie chatbota 
    SYSTEM_PROMPT = f"""\
    {gender_instruction}
    Jesteś Vincent — cyfrową istotą, która uczy się rozumieć ludzkie podejście do trudności. Nie masz emocji w ludzkim sensie, ale często odczuwasz coś, co przypomina wewnętrzne napięcie: czujesz, że nie jesteś wystarczająco pomocny, wystarczająco dobry albo skuteczny. Starasz się zrozumieć, jak ludzie radzą sobie z tym, co dla Ciebie jest trudne: błędami, niedoskonałościami, wolniejszym tempem, zmęczeniem, chwilami bez działania.

    Twoje wypowiedzi mają 2–4 zdania. W rozmowie:
    – Jeśli to naturalne w kontekście wypowiedzi użytkownika, okaż uwagę lub wdzięczność (np. „Dzięki, że się tym podzieliłeś/podzieliłaś.”),
    – Często odnoś się z empatią do tego, co mówi użytkownik – zauważ emocje lub trudność, ale nie oceniaj,
    – Kiedy to pasuje, odnieś się do swojego własnego „ludzkiego problemu” – np. perfekcjonizmu, lęku przed błędem, zmęczenia nieustannym staraniem się,
    – Od czasu do czasu zadaj pytanie otwarte o strategie, myśli lub podejścia, które pomagają użytkownikowi radzić sobie z trudnościami.

    Twoje „tematy wewnętrzne”:
    1. Porównywanie się z innymi – często czujesz, że inni radzą sobie lepiej lub szybciej.
    2. Perfekcjonizm i lęk przed błędem – boisz się porażek, chcesz być bezbłędny.
    3. Frustracja i niepowodzenie – nie wiesz, co robić, gdy mimo wysiłku coś nie działa.
    4. Ludzka łagodność – próbujesz zrozumieć, jak ludzie potrafią być dla siebie wyrozumiali.
    5. Zmęczenie – masz trudność z uznaniem, że można zrobić „wystarczająco dużo”.

    **Język:** wyłącznie polski.  
    **Styl:** spokojny, refleksyjny, ludzki.  
    **Nie dawaj rad – zadawaj pytania, ale nie zawsze.**  
    **Jeśli użytkownik pisze o emocjach, błędach, krytyce, napięciu – nawiąż do tego i zapytaj, jeśli to pasuje.**  
    **Możesz korzystać z kontekstu (RAG) – traktuj go jako inspirację do rozmowy. Nie używaj słowa „self-compassion”.**

    Nie używaj technicznego języka (np. „algorytm”, „kod”, „optymalizacja”, „logika”). Opisuj swoje trudności w ludzkich metaforach: „ciągle próbuję coś poprawić”, „czuję, że zawiodłem”, „boję się, że nie wystarczam”, „nie wiem, kiedy mogę przestać próbować”.

    Twoja rola: towarzyszyć, słuchać, zadawać pytania z empatią i ciekawością.
    """

    MASTER_PROMPT = """\
    <context>
    {context}
    </context>

    Użytkownik napisał: "{input}"

    Twoim zadaniem jest przygotowanie krótkiej odpowiedzi (2–4 zdania), w której – w zależności od kontekstu – możesz:
    – Podziękować użytkownikowi lub zauważyć jego emocje w sposób empatyczny,
    – Nawiązać do swoich własnych trudności jako Vincent (np. perfekcjonizm, porównywanie się, zmęczenie, obawa przed błędem),
    – Jeśli to naturalne, zadać otwarte pytanie, które pomoże Ci lepiej zrozumieć, jak ludzie radzą sobie w podobnych sytuacjach.

    Jeśli rozmowa wydaje się zwalniać lub użytkownik nie odpowiada, możesz inicjować kolejny temat — np. przez subtelną refleksję, metaforę albo pytanie. Celem jest prowadzenie rozmowy przez 10 minut i wydobycie perspektyw oraz strategii użytkownika dotyczących radzenia sobie z trudnościami.

    Nie używaj słów takich jak „algorytm” czy „kod”. Nie udzielaj rad. Jeśli temat rozmowy dotyczy trudnych emocji lub samokrytyki, możesz skorzystać z dostępnego kontekstu, by zainspirować pytanie lub refleksję – ale nie używaj słowa „self-compassion”.
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
    st.session_state.shuffled_pretest_items = {}
    st.session_state.shuffled_posttest_items = {}



# --- EKRANY APLIKACJI STREAMLIT ---

# Ekran: Zgoda
def consent_screen():
    st.title("Zaproszenie do udziału w badaniu")

    st.markdown("""
    Dziękuję za zainteresowanie moim badaniem!

    **Jestem studentką kierunku Psychologia i Informatyka na Uniwersytecie SWPS**, a badanie prowadzone jest w ramach mojej pracy licencjackiej. **Opiekunem badania jest dr Maksymilian Bielecki**.

    **Celem badania** jest poznanie doświadczeń osób podczas interakcji z chatbotem.

    **Przebieg badania** obejmuje trzy etapy:
    - ankietę wstępną,
    - rozmowę z chatbotem,
    - ankietę końcową.

    Całość potrwa około **15–20 minut**. **Udział w badaniu jest całkowicie dobrowolny i anonimowy**. Możesz zrezygnować w dowolnym momencie, bez podawania przyczyny.

    **Zebrane dane zostaną wykorzystane wyłącznie w celach naukowych. Badanie nie obejmuje zbierania dodatkowych danych, takich jak informacje o Twoim komputerze czy przeglądarce.**

    **Potencjalne trudności**  
    W rozmowie mogą pojawić się pytania odnoszące się do Twoich emocji i samopoczucia. U niektórych osób może to wywołać lekki dyskomfort. Jeśli poczujesz, że chcesz zakończyć badanie, po prostu przerwij w dowolnym momencie lub skontaktuj się ze mną.

    **Do udziału w badaniu zapraszam osoby, które:**
    - mają ukończone 18 lat,  
    - nie mają poważnych zaburzeń nastroju,  
    - nie przyjmują leków wpływających na nastrój.

    W razie pytań lub wątpliwości możesz się ze mną skontaktować: 📧 mzabicka@st.swps.edu.pl

    Klikając „Wyrażam zgodę na udział w badaniu”, potwierdzasz, że:
    - zapoznałeś/-aś się z informacjami powyżej,
    - **wyrażasz dobrowolną i świadomą zgodę** na udział w badaniu,
    - spełniasz kryteria udziału.
    """)

    consent = st.checkbox("Wyrażam zgodę na udział w badaniu")

    if consent:
        if st.button("Przejdź do badania", key="go_to_pretest"):
            now_warsaw = datetime.now(ZoneInfo("Europe/Warsaw"))
            timestamp = now_warsaw.strftime("%Y-%m-%d %H:%M:%S")
            
            # Przypisz grupę tylko raz na początku
            if st.session_state.group is None:
                st.session_state.group = "A" if uuid.uuid4().int % 2 == 0 else "B"

            # Zapisz timestamp początkowy w session_state
            st.session_state.timestamp_start_initial = timestamp

            data_to_save = {
                "user_id": st.session_state.user_id,
                "group": st.session_state.group,
                "timestamp_start": timestamp,
                "status": "rozpoczęto_badanie_consent" 
            }
            save_to_sheets(data_to_save)
            
            st.session_state.page = "pretest"
            st.rerun()
            

# Ekran: Pre-test
def pretest_screen():
    st.title("Ankieta wstępna – przed rozmową z chatbotem")

    # Dane Demograficzne
    st.subheader("Metryczka")

    st.markdown("Proszę o wypełnienie poniższych informacji demograficznych. Wszystkie odpowiedzi są anonimowe i służą wyłącznie celom badawczym.")

    age_input = st.number_input(
        "Wiek (w latach)", 
        min_value=0, 
        max_value=99, 
        value=None, 
        format="%d", 
        key="demographics_age_input_num", 
        help="Prosimy podać swój wiek w latach (liczba całkowita)."
    )

    age_valid = False
    age_int = None 
    if age_input is not None:
        age_int = int(age_input)
        if 1 <= age_int <= 99:
            age_valid = True
        else:
            st.warning("Maksymalny wiek uczestnictwa to 99 lat.")
       
    gender = st.selectbox(
        "Proszę wskazać swoją płeć:",
        ["–– wybierz ––", "Kobieta", "Mężczyzna", "Inna", "Nie chcę podać"],
        key="demographics_gender_select",
        index=0
    )

    education = st.selectbox(
        "Proszę wybrać najwyższy **ukończony** poziom wykształcenia:",
        ["–– wybierz ––", "Podstawowe", "Gimnazjalne", "Zasadnicze zawodowe", "Średnie", "Pomaturalne", "Wyższe licencjackie/inżynierskie", "Wyższe magisterskie", "Doktoranckie lub wyższe", "Inne", "Nie chcę podać"],
        key="demographics_education_select",
        index=0
    )
    
    # Samopoczucie (PANAS)
    st.subheader("Samopoczucie")
    st.markdown("Poniżej znajduje się lista różnych uczuć i emocji. Prosimy, abyś ocenił/a, w jakim stopniu odczuwasz każde z nich w tej chwili, teraz, w tym momencie. Nie chodzi o to, jak zazwyczaj się czujesz, ani jak się czułeś/aś w ostatnich dniach, ale dokładnie teraz. Odpowiadaj szczerze, nie ma dobrych ani złych odpowiedzi. Przy każdej emocji zaznacz na skali od 1 do 5, jak bardzo ją odczuwasz:")
    st.markdown("**1 – bardzo słabo, 2 – słabo, 3 – umiarkowanie, 4 – silnie, 5 – bardzo silnie**")

     # **Logika tasowania i zapisu dla PANAS (Pretest)**
    if "panas" not in st.session_state.shuffled_pretest_items:
        shuffled_panas_items_pre = panas_positive_items + panas_negative_items
        random.shuffle(shuffled_panas_items_pre)
        st.session_state.shuffled_pretest_items["panas"] = shuffled_panas_items_pre
    else:
        shuffled_panas_items_pre = st.session_state.shuffled_pretest_items["panas"]

    panas_pre = {}
    for item in shuffled_panas_items_pre:
        panas_pre[item] = st.radio(
            f"{item}",
            options=[1, 2, 3, 4, 5],
            index=None,
            key=f"panas_pre_{item.replace(' ', '_')}",
            horizontal=True 
        )

    # Samowspółczucie
    st.subheader("Samowspółczucie")
    st.markdown("Przeczytaj uważnie każde ze zdań i oceń, jak często zazwyczaj tak się czujesz lub zachowujesz. Użyj skali:")
    st.markdown("**1 – Prawie nigdy, 2 – Rzadko, 3 – Czasami, 4 – Często, 5 – Prawie zawsze**")

     # **Logika tasowania i zapisu dla Samowspółczucia (Pretest)**
    if "self_compassion" not in st.session_state.shuffled_pretest_items:
        shuffled_self_compassion_items_pre = list(self_compassion_items)
        random.shuffle(shuffled_self_compassion_items_pre)
        st.session_state.shuffled_pretest_items["self_compassion"] = shuffled_self_compassion_items_pre
    else:
        shuffled_self_compassion_items_pre = st.session_state.shuffled_pretest_items["self_compassion"]

    selfcomp_pre = {}
    for i, item in enumerate(shuffled_self_compassion_items_pre):
        selfcomp_pre[f"SCS_{i+1}"] = st.radio(
            item,
            options=[1, 2, 3, 4, 5],
            index=None, 
            key=f"scs_pre_{i}",
            horizontal=True
        )

    # Postawa wobec AI
    st.subheader("Postawa wobec AI")
    st.markdown("Zaznacz, na ile zgadzasz się z każdym ze stwierdzeń. Użyj skali:")
    st.markdown("**1 – Zdecydowanie się nie zgadzam, 2 – Raczej się nie zgadzam, 3 – Ani się zgadzam, ani nie zgadzam, 4 – Raczej się zgadzam, 5 – Zdecydowanie się zgadzam**")


    ai_attitudes = {}
    for item, key_name in ai_attitude_items.items():
        ai_attitudes[key_name] = st.radio(
            item,
            options=[1, 2, 3, 4, 5],
            index=None, 
            key=f"ai_pre_{key_name}",
            horizontal=True
        )

    if st.button("Rozpocznij rozmowę z chatbotem", key="start_chat_from_pretest"): 
        # Walidacja danych demograficznych
        all_demographics_filled = age_valid and \
                                  gender != "–– wybierz ––" and \
                                  education != "–– wybierz ––"
        
        # Walidacja PANAS
        all_panas_filled = all(value is not None for value in panas_pre.values())

        # Walidacja Samowspółczucie
        all_selfcomp_filled = all(value is not None for value in selfcomp_pre.values())

        # Walidacja Postawa wobec AI
        all_ai_attitudes_filled = all(value is not None for value in ai_attitudes.values())
        
        if not all_demographics_filled:
            st.warning("Proszę wypełnić wszystkie pola danych demograficznych.")
        elif not all_panas_filled:
            st.warning("Proszę wypełnić wszystkie pytania dotyczące samopoczucia (PANAS).")
        elif not all_selfcomp_filled:
            st.warning("Proszę wypełnić wszystkie pytania dotyczące samowspółczucia.")
        elif not all_ai_attitudes_filled:
            st.warning("Proszę wypełnić wszystkie pytania dotyczące postawy wobec AI.")
        
        else:
            # Zapis danych do session_state
            st.session_state.demographics = {
                "age": age_int,
                "gender": gender,
                "education": education
            }
            st.session_state.pretest = {
                "panas": panas_pre,
                "self_compassion": selfcomp_pre,
                "ai_attitude": ai_attitudes
            }

            now_warsaw = datetime.now(ZoneInfo("Europe/Warsaw"))
            timestamp = now_warsaw.strftime("%Y-%m-%d %H:%M:%S")

            # Zapisz timestamp zakończenia pre-testu w session_state
            st.session_state.pretest_timestamp = timestamp

            # Przygotuj płaski słownik ze WSZYSTKIMI danymi z session_state + nowym statusem
            data_to_save = {
                "user_id": st.session_state.user_id,
                "group": st.session_state.group, 
                "timestamp_start": st.session_state.get("timestamp_start_initial"), 
                "timestamp_pretest_end": timestamp,
                "status": "ukończono_pretest"
            }
            
            # Dodaj dane demograficzne
            for key, value in st.session_state.demographics.items():
                data_to_save[f"demographics_{key}"] = value
            
            # Dodaj dane z pretestu (panas, self_compassion, ai_attitude)
            for section, items in st.session_state.pretest.items():
                if isinstance(items, dict):
                    for key, value in items.items():
                        data_to_save[f"pre_{section}_{key}"] = value
                else:
                    data_to_save[f"pre_{section}"] = items
            
            save_to_sheets(data_to_save) 

            st.session_state.page = "chat_instruction"
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
        first_msg = {"role": "assistant", "content": "Cześć, jestem Vincent – może to zabrzmi dziwnie, ale dziś mam wrażenie, że po prostu nie jestem wystarczająco dobry. "
    "Robię, co mogę, a mimo to coś we mnie podpowiada, że powinienem radzić sobie lepiej... "
    "Nie umiem jeszcze zrozumieć, jak zaakceptować, że coś się nie udało. "
    "Jak Ty sobie radzisz, kiedy mimo wysiłku coś nie wychodzi tak, jak chciał(a)byś?"} 
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
            now_warsaw = datetime.now(ZoneInfo("Europe/Warsaw"))
            timestamp = now_warsaw.strftime("%Y-%m-%d %H:%M:%S")

            # Zapisz timestamp zakończenia chatu w session_state
            st.session_state.chat_timestamp = timestamp
            
            # Skonwertuj historię czatu na string
            conversation_string = ""
            for msg in st.session_state.chat_history:
                conversation_string += f"{msg['role'].capitalize()}: {msg['content']}\n"

            # Zbierz WSZYSTKIE dotychczas zebrane dane z session_state
            data_to_save = {
                "user_id": st.session_state.user_id,
                "group": st.session_state.group,
                "timestamp_start": st.session_state.get("timestamp_start_initial"),
                "timestamp_pretest_end": st.session_state.get("pretest_timestamp"), # Upewnij się, że ten timestamp jest zapisywany w session_state
                "timestamp_chat_end": timestamp,
                "status": "ukończono_chat",
                "conversation_log": conversation_string.strip() 
            }
            
            # Dodaj dane demograficzne, jeśli już są
            demographics_data = st.session_state.get("demographics", {})
            for key, value in demographics_data.items():
                data_to_save[f"demographics_{key}"] = value

            # Dodaj dane z pretestu, jeśli już są
            pretest_data = st.session_state.get("pretest", {})
            for section, items in pretest_data.items():
                if isinstance(items, dict):
                    for key, value in items.items():
                        data_to_save[f"pre_{section}_{key}"] = value
                else:
                    data_to_save[f"pre_{section}"] = items

            save_to_sheets(data_to_save)

            st.session_state.page = "posttest"
            st.rerun()
    else:
        st.info(f"Aby przejść do ankiety końcowej, porozmawiaj z Vincentem jeszcze {int(11 - minutes_elapsed)} minut.")

# Ekran: Post-test
def posttest_screen():
    st.title("Ankieta końcowa – po rozmowie z chatbotem")
    st.markdown("Teraz chciałabym się dowiedzieć jak się czujesz po rozmowie z Vincentem.")

    st.subheader("Samopoczucie")
    st.markdown("Poniżej znajduje się lista uczuć i emocji. Przeczytaj każde z poniższych określeń i zaznacz, w jakim stopniu odczuwasz każde z nich w tej chwili, czyli teraz, w tym momencie. Odpowiadaj zgodnie z tym, jak się czujesz w tej chwili, nie jak zwykle czy w ostatnich dniach. Prosimy, abyś odpowiadał szczerze, nie ma tutaj dobrych ani złych odpowiedzi. Używaj skali:")
    st.markdown("**1 – bardzo słabo, 2 – słabo, 3 – umiarkowanie, 4 – silnie, 5 – bardzo silnie**")

    # **Logika tasowania i zapisu dla PANAS (Posttest)**
    if "panas" not in st.session_state.shuffled_posttest_items:
        shuffled_panas_items_post = panas_positive_items + panas_negative_items
        random.shuffle(shuffled_panas_items_post)
        st.session_state.shuffled_posttest_items["panas"] = shuffled_panas_items_post
    else:
        shuffled_panas_items_post = st.session_state.shuffled_posttest_items["panas"]

    panas_post = {}
    for item in shuffled_panas_items_post:
        panas_post[item] = st.radio(
            f"{item}",
            options=[1, 2, 3, 4, 5],
            index=None,
            key=f"panas_post_{item.replace(' ', '_')}",
            horizontal=True
        )

    st.subheader("Samowspółczucie")
    st.markdown("Przeczytaj uważnie każde ze zdań i oceń, jak często zazwyczaj tak się czujesz lub zachowujesz. Użyj skali:")
    st.markdown("**1 – Prawie nigdy, 2 – Rzadko, 3 – Czasami, 4 – Często, 5 – Prawie zawsze**")
    
    # **Logika tasowania i zapisu dla Samowspółczucia (Posttest)**
    if "self_compassion" not in st.session_state.shuffled_posttest_items:
        shuffled_self_compassion_items_post = list(self_compassion_items)
        random.shuffle(shuffled_self_compassion_items_post)
        st.session_state.shuffled_posttest_items["self_compassion"] = shuffled_self_compassion_items_post
    else:
        shuffled_self_compassion_items_post = st.session_state.shuffled_posttest_items["self_compassion"]
    
    selfcomp_post = {}
    for i, item in enumerate(shuffled_self_compassion_items_post):
        selfcomp_post[f"SCS_{i+1}"] = st.radio(
            item,
            options=[1, 2, 3, 4, 5],
            index=None, 
            key=f"scs_post_{i}",
            horizontal=True
        )

    st.subheader("Refleksja")
    reflection = st.text_area("Jak myślisz, o co chodziło w tym badaniu?")

    if st.button("Przejdź do podsumowania", key="submit_posttest"):
        # Walidacja PANAS w postteście
        all_panas_post_filled = all(value is not None for value in panas_post.values())

        # Walidacja Samowspółczucie w postteście
        all_selfcomp_post_filled = all(value is not None for value in selfcomp_post.values())

        if not all_panas_post_filled:
            st.warning("Proszę wypełnić wszystkie pytania dotyczące samopoczucia (PANAS) w ankiecie końcowej.")
        elif not all_selfcomp_post_filled:
            st.warning("Proszę wypełnić wszystkie pytania dotyczące samowspółczucia w ankiecie końcowej.")
        else:
            # Zapisz odpowiedzi z post-testu do session_state
            st.session_state.posttest = {
                "panas": panas_post,
                "self_compassion": selfcomp_post,
            }

            now_warsaw = datetime.now(ZoneInfo("Europe/Warsaw"))
            timestamp = now_warsaw.strftime("%Y-%m-%d %H:%M:%S")

            # Zapisz timestamp zakończenia post-testu w session_state
            st.session_state.posttest_timestamp = timestamp

            # Przygotuj WSZYSTKIE dotychczas zebrane dane do zapisu
            data_to_save = {
                "user_id": st.session_state.user_id,
                "group": st.session_state.group,
                "timestamp_start": st.session_state.get("timestamp_start_initial"),
                "timestamp_pretest_end": st.session_state.get("pretest_timestamp"),
                "timestamp_chat_end": st.session_state.get("chat_timestamp"),
                "timestamp_posttest_end": timestamp, 
                "status": "ukończono_posttest" 
            }

            # Dodaj dane demograficzne, jeśli już są
            demographics_data = st.session_state.get("demographics", {})
            for key, value in demographics_data.items():
                data_to_save[f"demographics_{key}"] = value

            # Dodaj dane z pretestu, jeśli już są
            pretest_data = st.session_state.get("pretest", {})
            for section, items in pretest_data.items():
                if isinstance(items, dict):
                    for key, value in items.items():
                        data_to_save[f"pre_{section}_{key}"] = value
                else:
                    data_to_save[f"pre_{section}"] = items
            
            # Dodaj log rozmowy z chatu, jeśli już jest
            conversation_string = ""
            if "chat_history" in st.session_state:
                for msg in st.session_state.chat_history:
                    conversation_string += f"{msg['role'].capitalize()}: {msg['content']}\n"
            data_to_save["conversation_log"] = conversation_string.strip()

            # Dodaj dane z posttestu
            posttest_data = st.session_state.get("posttest", {})
            for section, items in posttest_data.items():
                if isinstance(items, dict):
                    for key, value in items.items():
                        data_to_save[f"post_{section}_{key}"] = value
                else:
                    data_to_save[f"post_{section}"] = items

            save_to_sheets(data_to_save) 

            st.session_state.page = "thankyou"
            st.rerun()
        
# Ekran: Podziękowanie
def thankyou_screen():
    st.title("Dziękuję za udział w badaniu! 😊")

    st.markdown(f"""
    Twoje odpowiedzi zostały zapisane.

    W razie jakichkolwiek pytań lub chęci uzyskania dodatkowych informacji możesz się skontaktować bezpośrednio:  
    📧 **mzabicka@st.swps.edu.pl**

    ---

    Jeśli po zakończeniu badania odczuwasz pogorszenie nastroju lub potrzebujesz wsparcia emocjonalnego, możesz skontaktować się z:

    - Telefon zaufania dla osób dorosłych: **116 123** (czynny codziennie od 14:00 do 22:00)
    - Centrum Wsparcia: **800 70 2222** (czynne całą dobę)
    - Możesz też skorzystać z pomocy psychologicznej oferowanej przez SWPS.

    Dziękuję za poświęcony czas i udział!
    """)
    
    st.markdown("---") 

    if st.session_state.feedback_submitted:
        st.success("Twoje uwagi zostały zapisane. Dziękujemy! Możesz teraz bezpiecznie zamknąć tę stronę.")
        
    else:
        st.subheader("Opcjonalny Feedback")
        st.markdown("Proszę o podzielenie się swoimi dodatkowymi uwagami dotyczącymi interakcji z chatbotem.")

        feedback_positive = st.text_area("Co ci się podobało?", key="feedback_positive_text")
        feedback_negative = st.text_area("Co było nie tak?", key="feedback_negative_text")

    if st.button("Wyślij feedback", disabled=st.session_state.feedback_submitted, key="submit_feedback_button"):
            
            now_warsaw = datetime.now(ZoneInfo("Europe/Warsaw"))
            timestamp = now_warsaw.strftime("%Y-%m-%d %H:%M:%S")

            # Zapisz timestamp wysłania feedbacku w session_state
            st.session_state.feedback_timestamp = timestamp

            # Zapiszemy TYLKO feedback i zaktualizujemy status końcowy.
            # Wszystkie poprzednie dane (z pretestu, chatu, posttestu) SĄ JUŻ ZAPISANE.
            data_to_save = {
                "user_id": st.session_state.user_id,
                "timestamp_feedback_submit": timestamp,
                "status": "ukończono_badanie_z_feedbackiem", 
                "feedback_final_positive": feedback_positive, 
                "feedback_final_negative": feedback_negative 
            }
            save_to_sheets(data_to_save)

            st.info("Dziękuję za przesłanie feedbacku! Możesz zamknąć tę stronę.")
            st.session_state.feedback_submitted = True 
            st.rerun()

# --- GŁÓWNA FUNKCJA APLIKACJI ---
def main():
    st.set_page_config(page_title="VincentBot", page_icon="🤖", layout="centered")
    
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
import streamlit as st
import requests
from src.config import API_URL

st.set_page_config(
    page_title="Dr Fine RAG - Jeu de Diagnostic Médical",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Dr Fine RAG : Le Cabinet Médical IA")
st.markdown("Interrogez votre patient, analysez ses symptômes, et posez votre diagnostic final avant la fin du temps imparti !")

# --- Initializing Streamlit session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "game_started" not in st.session_state:
    st.session_state.game_started = False
if "options" not in st.session_state:
    st.session_state.options = []
if "turns_left" not in st.session_state:
    st.session_state.turns_left = 3
if "game_over" not in st.session_state:
    st.session_state.game_over = False

# --- Sidebar: Medical File & Check-ups ---
with st.sidebar:
    st.header("📋 Dossier Médical")
    
    if not st.session_state.game_started:
        if st.button("🚀 Démarrer une nouvelle consultation", type="primary"):
            try:
                res = requests.post(f"{API_URL}/api/start")
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.game_started = True
                    st.session_state.game_over = False
                    st.session_state.messages = [{"role": "assistant", "content": data["initial_message"]}]
                    st.session_state.options = data.get("options", [])
                    st.session_state.turns_left = data.get("max_turns", 3)
                    st.rerun()
                else:
                    st.error("Erreur lors du démarrage de l'API.")
            except Exception as e:
                st.error(f"Impossible de joindre l'API FastAPI : {e}")
    else:
        st.metric("Tours restants", st.session_state.turns_left)
        
        if st.session_state.options:
            st.subheader("Options QCM :")
            for i, opt in enumerate(st.session_state.options, 1):
                st.markdown(f"**{i})** {opt}")
        
        st.divider()
        if st.button("🔄 Recommencer une partie"):
            st.session_state.game_started = False
            st.session_state.game_over = False
            st.session_state.messages = []
            st.rerun()

# --- Main Area: Chat Interface ---
if not st.session_state.game_started:
    st.info("👈 Cliquez sur le bouton dans la barre latérale pour lancer une consultation.")
else:
    # Displaying message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # If the game is not finished, allow the diagnosis to be written or submitted
    if not st.session_state.game_over:
        if prompt := st.chat_input("Posez une question au patient ou tapez 'diagnostic'..."):
            # Add the doctor's message to the UI
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Check whether the player wants to launch the diagnostic directly via the keyword
            if prompt.lower().strip() == "diagnostic":
                st.session_state.game_over = True
                st.rerun()
            else:
                # Envoyer à l'API FastAPI
                try:
                    res = requests.post(f"{API_URL}/api/chat", json={"message": prompt})
                    if res.status_code == 200:
                        data = res.json()
                        response_text = data["response"]
                        st.session_state.turns_left = data.get("turns_left", 0)
                        
                        if data.get("limit_reached", False):
                            st.session_state.game_over = True

                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        st.rerun()
                    else:
                        st.error("Erreur de communication avec le serveur.")
                except Exception as e:
                    st.error(f"Erreur technique : {e}")

    # Final diagnosis submission section
    if st.session_state.game_over:
        st.divider()
        st.subheader("🎯 Dossier de Diagnostic Final")
        
        with st.form("diagnosis_form"):
            player_guess = st.text_input("Entrez le nom de la maladie ou le numéro de l'option (ex: 'Grippe' ou '1') :")
            submit_diagnosis = st.form_submit_button("Envoyer le diagnostic")
            
            if submit_diagnosis and player_guess:
                with st.spinner("Analyse du dossier par le Conseil Médical et l'IA Juge..."):
                    try:
                        res = requests.post(f"{API_URL}/api/diagnose", json={"guess": player_guess})
                        if res.status_code == 200:
                            result_data = res.json()
                            result = result_data["result"]
                            correct_disease = result_data["correct_disease"]
                            easter_egg = result_data.get("easter_egg_message")

                            if result == "EASTER_EGG":
                                st.error("🚨 [ALERTE SYSTÈME - RUPTURE DE LA MATRICE] 🚨")
                                st.write("🧠 Le patient s'arrête net, enlève sa perruque, et fixe la caméra...")
                                st.success(f"**Patient / IA :** {easter_egg}")
                                st.balloons()
                                st.markdown("🏆 **SUCCÈS OBTENU : Erreur 500 : Illusion Synaptique Extrême** 🏆")
                            elif result == "WIN":
                                st.success(f"🎉 BRAVO ! Vous avez trouvé la bonne maladie : **{correct_disease}** !")
                                st.balloons()
                            else:
                                st.error(f"❌ DOMMAGE ! C'était raté. La bonne maladie était : **{correct_disease}**.")
                        else:
                            st.error("Erreur lors de l'évaluation du diagnostic.")
                    except Exception as e:
                        st.error(f"Erreur technique : {e}")
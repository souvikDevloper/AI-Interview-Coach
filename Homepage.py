import streamlit as st

# Optional: nice icon if you have one
try:
    from PIL import Image
    im = Image.open("icon.png")
    st.set_page_config(page_title="AI Interviewer", layout="centered", page_icon=im)
except Exception:
    st.set_page_config(page_title="AI Interviewer", layout="centered")

# --- Try to use streamlit-option_menu if available; otherwise fall back to radio ---
try:
    from streamlit_option_menu import option_menu
    def pick_menu():
        return option_menu(
            menu_title=None,
            options=["Professional", "Resume", "Behavioral", "Customize!"],
            icons=["cpu", "file-earmark-person", "chat-dots", "gear"],
            default_index=0,
            orientation="horizontal",
        )
except Exception:
    def pick_menu():
        return st.radio(
            "Choose a screen:",
            ["Professional", "Resume", "Behavioral", "Customize!"],
            horizontal=True,
        )

st.markdown("<style>#MainMenu{visibility:hidden;}</style>", unsafe_allow_html=True)

st.title("AI Interviewer  ")
st.caption("Practice realistic interviews with job-aware questions and optional voice.")

st.markdown("#### Get started")
st.write("Pick a screen and click **Start Interview!**")

selected = pick_menu()

# Small helper that prefers the official API, and gracefully falls back
def _go(path: str, label: str):
    if hasattr(st, "switch_page"):
        st.switch_page(path)
    else:
        st.warning("Your Streamlit build doesn’t expose `st.switch_page`. Click below:")
        st.page_link(path, label=f"➡️ {label}")
        st.stop()

if selected == "Professional":
    st.info(
        "📚 Technical interview based on a job description.\n\n"
        "- Duration: ~10–15 minutes\n"
        "- Max answer length: ~4k tokens\n"
        "- Use chat or enable voice\n"
    )
    if st.button("Start Interview!"):
        _go("pages/Professional Screen.py", "Open Professional Screen")

elif selected == "Resume":
    st.info(
        "📚 Questions grounded in your uploaded résumé.\n\n"
        "- Duration: ~10–15 minutes\n"
        "- Max answer length: ~4k tokens\n"
        "- Use chat or enable voice\n"
    )
    if st.button("Start Interview!"):
        _go("pages/Resume Screen.py", "Open Resume Screen")

elif selected == "Behavioral":
    st.info(
        "📚 Soft-skills & situation-based discussion.\n\n"
        "- Duration: ~10–15 minutes\n"
        "- Max answer length: ~4k tokens\n"
        "- Use chat or enable voice\n"
    )
    if st.button("Start Interview!"):
        _go("pages/Behavioral Screen.py", "Open Behavioral Screen")

else:
    st.info(
        "🧪 Customize your interviewer’s specialty and persona. "
        "This area is a work-in-progress."
    )

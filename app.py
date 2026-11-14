import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "salary_predictor.pkl"

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Headline figures from the training notebook: (label, target, suffix, duration in seconds)
STATS = [
    ("Model features", 28, "", 1.6),
    ("Input fields", 14, "", 2.2),
    ("Cities covered", 4, "", 2.8),
]

CSS = """
<style>
@property --num { syntax: "<integer>"; initial-value: 0; inherits: true; }
@keyframes count-up { from { --num: 0; } to { --num: var(--target); } }
@keyframes fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

.block-container { max-width: 800px; padding-top: 2.5rem; }

.hero { padding: 0 0 1.4rem; border-bottom: 1px solid rgba(127,127,127,.35); margin-bottom: 1.6rem; animation: fade .6s ease both; }
.eyebrow { font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; opacity: .6; margin-bottom: .6rem; }
.hero h1 { font-family: Georgia, 'Times New Roman', serif; font-weight: 500; font-size: 2.3rem; letter-spacing: -.01em; margin: 0 0 .5rem; padding: 0; }
.hero p { margin: 0; opacity: .75; max-width: 34rem; }

.stats { display: grid; grid-template-columns: repeat(3, 1fr); margin-bottom: 1.8rem; border-top: 1px solid rgba(127,127,127,.35); border-bottom: 1px solid rgba(127,127,127,.35); }
.stat { padding: 1.1rem .5rem; text-align: center; border-right: 1px solid rgba(127,127,127,.35); animation: count-up var(--dur) ease-out forwards; }
.stat:last-child { border-right: 0; }
.stat-value { counter-reset: num var(--num); font-family: Georgia, 'Times New Roman', serif; font-size: 2.3rem; font-weight: 400; line-height: 1.1; }
.stat-value::before { content: counter(num); }
.stat-value::after { content: var(--suffix); }
.stat-label { font-size: .7rem; letter-spacing: .16em; text-transform: uppercase; opacity: .6; margin-top: .35rem; }

div[data-testid="stForm"] { border: 1px solid rgba(127,127,127,.35); border-radius: 6px; padding: 1.4rem 1.4rem .8rem; }
div[data-testid="stFormSubmitButton"] button { background: #1f2a44; color: #fff; border: 0; border-radius: 4px; font-weight: 500; letter-spacing: .06em; padding: .65rem 0; transition: background .2s; }
div[data-testid="stFormSubmitButton"] button:hover { background: #33415f; color: #fff; }

.result { margin-top: 1.4rem; padding: 1.6rem 1rem; text-align: center; border: 1px solid rgba(127,127,127,.35); border-left: 4px solid #1f2a44; border-radius: 4px; animation: fade .5s ease both; }
.result .label { font-size: .72rem; letter-spacing: .2em; text-transform: uppercase; opacity: .6; }
.result .amount { font-family: Georgia, 'Times New Roman', serif; font-size: 2.8rem; line-height: 1.2; margin-top: .3rem; }
.result .whole { animation: count-up 1.4s ease-out forwards; counter-reset: num var(--num); }
.result .whole::before { content: counter(num); }
@media (max-width: 600px) { .stats { grid-template-columns: 1fr; } .stat { border-right: 0; border-bottom: 1px solid rgba(127,127,127,.35); } .stat:last-child { border-bottom: 0; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)  # no "Running load_model()" message
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file not found. Put salary_predictor.pkl in the same folder as app.py.")
    st.stop()
model = load_model()

st.markdown(
    '<div class="hero"><div class="eyebrow">Compensation Estimator</div>'
    "<h1>Employee Salary Prediction</h1>"
    "<p>Enter employee details to estimate the annual salary.</p></div>",
    unsafe_allow_html=True,
)

stats_html = "".join(
    f'<div class="stat" style="--target:{v}; --dur:{d}s; --suffix:\'{s}\'">'
    f'<div class="stat-value"></div><div class="stat-label">{label}</div></div>'
    for label, v, s, d in STATS
)
st.markdown(f'<div class="stats">{stats_html}</div>', unsafe_allow_html=True)

with st.form("salary_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=21, max_value=60, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        experience = st.number_input("Experience (years)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        education = st.selectbox("Education", ["Diploma", "Bachelor", "Master", "PhD"])
        department = st.selectbox("Department", ["Operations", "Marketing", "Sales", "IT", "HR"])
        job_level = st.selectbox("Job Level", ["Junior", "Lead", "Mid", "Senior", "Manager"])

    with col2:
        performance = st.slider("Performance Rating", min_value=1, max_value=5, value=3)
        certifications = st.number_input("Certifications", min_value=0, max_value=50, value=2, step=1)
        overtime = st.number_input("Overtime Hours", min_value=0.0, max_value=300.0, value=10.0, step=1.0)
        remote_work = st.selectbox("Remote Work", ["No", "Yes"])
        city = st.selectbox("City", ["Chennai", "Delhi", "Hyderabad", "Mumbai"])
        tenure = st.number_input("Company Tenure (years)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
        projects = st.number_input("Projects Completed", min_value=0, max_value=200, value=5, step=1)
        skill_score = st.number_input("Skill Score", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

    submitted = st.form_submit_button("Predict Annual Salary", use_container_width=True)

if submitted:
    # Keep these columns and their order identical to the training notebook's X dataframe.
    row = {
        "Age": age,
        "Gender": 0 if gender == "Male" else 1,
        "Experience_Years": experience,
        "Performance_Rating": performance,
        "Certifications": certifications,
        "Overtime_Hours": overtime,
        "Remote_Work": 1 if remote_work == "Yes" else 0,
        "Company_Tenure": tenure,
        "Projects_Completed": projects,
        "Skill_Score": skill_score,
        "Education_Master": int(education == "Master"),
        "Education_Bachelor": int(education == "Bachelor"),
        "Education_Diploma": int(education == "Diploma"),
        "Education_PhD": int(education == "PhD"),
        "Department_Operations": int(department == "Operations"),
        "Department_Marketing": int(department == "Marketing"),
        "Department_Sales": int(department == "Sales"),
        "Department_IT": int(department == "IT"),
        "Department_HR": int(department == "HR"),
        "Job_Level_Junior": int(job_level == "Junior"),
        "Job_Level_Lead": int(job_level == "Lead"),
        "Job_Level_Mid": int(job_level == "Mid"),
        "Job_Level_Senior": int(job_level == "Senior"),
        "Job_Level_Manager": int(job_level == "Manager"),
        "City_Chennai": int(city == "Chennai"),
        "City_Delhi": int(city == "Delhi"),
        "City_Hyderabad": int(city == "Hyderabad"),
        "City_Mumbai": int(city == "Mumbai"),
    }
    input_df = pd.DataFrame([row])

    try:
        prediction = float(model.predict(input_df)[0])
        whole, frac = divmod(round(prediction, 2), 1)
        frac_txt = f"{frac:.2f}"[1:]
        st.markdown(
            f'<div class="result"><div class="label">Predicted Annual Salary</div>'
            f'<div class="amount"><span class="whole" style="--target:{int(whole)}"></span>'
            f'{frac_txt} LPA</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("This is a model estimate, not a guaranteed salary offer.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "The saved model must have been trained with the same 28 features and column order "
            "used by this app. If your training notebook changed, retrain and save the model again."
        )

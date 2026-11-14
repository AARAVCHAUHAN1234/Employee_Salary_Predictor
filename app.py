import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "salary_predictor.pkl"

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")

STATS = [
    ("Input features", 28, "+", 1.6),
    ("Cities covered", 4, "", 2.2),
    ("Model accuracy (R²)", 92, "%", 2.8),  # placeholder: put your real score here
]

CSS = """
<style>
@property --num { syntax: "<integer>"; initial-value: 0; inherits: true; }

@keyframes count-up { from { --num: 0; } to { --num: var(--target); } }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }

.block-container { padding-top: 2rem; max-width: 820px; }

.hero {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
  color: #fff; border-radius: 20px; padding: 2rem 1.8rem; margin-bottom: 1.2rem;
  box-shadow: 0 12px 30px rgba(79,70,229,.30); animation: rise .6s ease both;
}
.hero h1 { margin: 0 0 .3rem; font-size: 2rem; color: #fff; }
.hero p  { margin: 0; opacity: .9; }

.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin-bottom: 1.4rem; }
.stat {
  background: rgba(127,127,127,.10); border: 1px solid rgba(127,127,127,.25);
  border-radius: 16px; padding: 1rem .6rem; text-align: center;
  animation: count-up var(--dur) ease-out forwards, rise .6s ease both;
}
.stat-value { counter-reset: num var(--num); font-size: 2.1rem; font-weight: 800; color: #7c3aed; }
.stat-value::before { content: counter(num); }
.stat-value::after  { content: var(--suffix); }
.stat-label { font-size: .8rem; opacity: .75; margin-top: .2rem; }

.result {
  background: linear-gradient(135deg, #059669, #10b981); color: #fff; text-align: center;
  border-radius: 20px; padding: 1.6rem 1rem; margin-top: 1rem;
  box-shadow: 0 10px 26px rgba(16,185,129,.30); animation: rise .5s ease both;
}
.result .label { opacity: .9; font-size: .95rem; }
.result .amount { font-size: 3rem; font-weight: 800; line-height: 1.1; }
.result .whole {
  --target: 0; animation: count-up 1.4s ease-out forwards;
  counter-reset: num var(--num);
}
.result .whole::before { content: counter(num); }

div[data-testid="stForm"] {
  border: 1px solid rgba(127,127,127,.25); border-radius: 18px; padding: 1.2rem 1.2rem .6rem;
}
.stFormSubmitButton button, div[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(90deg, #4f46e5, #db2777); color: #fff; border: 0;
  border-radius: 12px; font-weight: 700; padding: .65rem 0; transition: transform .15s, box-shadow .15s;
}
div[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(124,58,237,.4); color:#fff; }
@media (max-width: 600px) { .stats { grid-template-columns: 1fr; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# show_spinner=False removes the "Running load_model()" message.
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model file not found. Put salary_predictor.pkl in the same folder as app.py.")
    st.stop()
model = load_model()  # loaded once, silently, before the form is used

st.markdown(
    '<div class="hero"><h1>💼 Employee Salary Predictor</h1>'
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
    st.subheader("👤 Personal & Role")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 70, 30, 1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", ["Diploma", "Bachelor", "Master", "PhD"])
        city = st.selectbox("City", ["Chennai", "Delhi", "Hyderabad", "Mumbai"])
    with c2:
        department = st.selectbox("Department", ["Operations", "Marketing", "Sales", "IT", "HR"])
        job_level = st.selectbox("Job Level", ["Junior", "Lead", "Mid", "Senior", "Manager"])
        remote_work = st.selectbox("Remote Work", ["No", "Yes"])
        performance = st.slider("Performance Rating", 1, 5, 3)

    st.subheader("📈 Experience & Output")
    c3, c4 = st.columns(2)
    with c3:
        experience = st.number_input("Experience (years)", 0.0, 50.0, 5.0, 0.5)
        tenure = st.number_input("Company Tenure (years)", 0.0, 50.0, 3.0, 0.5)
        certifications = st.number_input("Certifications", 0, 50, 2, 1)
    with c4:
        projects = st.number_input("Projects Completed", 0, 200, 5, 1)
        overtime = st.number_input("Overtime Hours", 0.0, 300.0, 10.0, 1.0)
        skill_score = st.number_input("Skill Score", 0.0, 100.0, 70.0, 1.0)

    submitted = st.form_submit_button("✨ Predict Annual Salary", use_container_width=True)

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
        frac_txt = f"{frac:.2f}"[1:]  # ".xx"
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

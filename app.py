import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "salary_predictor.pkl"

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Put salary_predictor.pkl in the same folder as app.py. "
            "Run the training notebook's save step first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

st.title("Employee Salary Prediction")
st.write("Enter employee details to estimate the annual salary.")

with st.form("salary_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
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
    model = load_model()

    try:
        prediction = float(model.predict(input_df)[0])
        st.success(f"Predicted Annual Salary: {prediction:.2f} LPA")
        st.caption("This is a model estimate, not a guaranteed salary offer.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "The saved model must have been trained with the same 28 features and column order "
            "used by this app. If your training notebook changed, retrain and save the model again."
        )

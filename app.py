import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Career Predictor", layout="wide")

st.title("🎓 Student Career Predictor Monolithic")
st.markdown("Test the classification and regression tasks in different scenarios.")

@st.cache_resource
def load_pipelines():
    return joblib.load('best_classification_pipeline.pkl'), joblib.load('best_regression_pipeline.pkl')

cls_model, reg_model = load_pipelines()

def get_common_inputs():
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc = st.slider("SSC %", 0, 100, 75)
        hsc = st.slider("HSC %", 0, 100, 75)
        degree = st.slider("Degree %", 0, 100, 75)
    with col2:
        cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0)
        entrance = st.number_input("Entrance Score", 0, 100, 80)
        tech_skill = st.number_input("Tech Skill", 0, 100, 80)
        soft_skill = st.number_input("Soft Skill", 0, 100, 80)
    with col3:
        internships = st.number_input("Internships", 0, 5, 1)
        projects = st.number_input("Live Projects", 0, 5, 1)
        work_exp = st.number_input("Work Exp (Months)", 0, 60, 0)
        certifications = st.number_input("Certifications", 0, 10, 1)
        attendance = st.slider("Attendance %", 0, 100, 90)
        backlogs = st.number_input("Backlogs", 0, 10, 0)
        extra = st.selectbox("Extracurricular", ["Yes", "No"])
    
    return {
        'gender': gender, 'ssc_percentage': ssc, 'hsc_percentage': hsc,
        'degree_percentage': degree, 'cgpa': cgpa, 'entrance_exam_score': entrance,
        'technical_skill_score': tech_skill, 'soft_skill_score': soft_skill,
        'internship_count': internships, 'live_projects': projects,
        'work_experience_months': work_exp, 'certifications': certifications,
        'attendance_percentage': attendance, 'backlogs': backlogs,
        'extracurricular_activities': extra
    }

tab1, tab2 = st.tabs(["🔍 Scenario 1: Predict Placement", "💰 Scenario 2: Predict Salary"])

# --- TAB 1: CLASSIFICATION ---
with tab1:
    st.header("Predict Placement Status")
    st.write("Using features **including Salary** to predict Placement.")
    
    with st.form("form_class"):
        data = get_common_inputs()
        salary_input = st.number_input("Known Salary Package (LPA)", 0.0, 50.0, 5.0)
        btn_class = st.form_submit_button("Predict Placement")

    if btn_class:
        data['salary_package_lpa'] = salary_input
        input_df = pd.DataFrame([data]) 
        
        prediction = cls_model.predict(input_df)[0]
        
        if prediction == 1:
            st.success("### Result: PLACED 🎉")
        else:
            st.error("### Result: NOT PLACED ❌")

# --- TAB 2: REGRESSION ---
with tab2:
    st.header("Predict Salary Package")
    st.write("Using features **including Placement Status** to predict Salary.")
    
    with st.form("form_reg"):
        data = get_common_inputs()
        placement_input = st.selectbox("Placement Status", [1, 0], format_func=lambda x: "Placed" if x==1 else "Not Placed")
        btn_reg = st.form_submit_button("Predict Salary")

    if btn_reg:
        data['placement_status'] = placement_input
        input_df = pd.DataFrame([data])
        
        salary_pred = reg_model.predict(input_df)[0]
        
        st.success(f"### Predicted Salary: {salary_pred:.2f} LPA")
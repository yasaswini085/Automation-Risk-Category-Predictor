import pandas as pd
import streamlit as st
import pickle
model=pickle.load(open('model.pkl','rb'))
transformer=pickle.load(open('transformer.pkl','rb'))
st.title('Job Automation Risk Category Predictor')
job_role=st.selectbox("Select Job Role",['Data Analyst', 'Accountant', 'Teacher', 'Customer Support Rep',
       'Software Engineer', 'Marketing Specialist', 'Financial Analyst',
       'HR Manager', 'Mechanical Engineer', 'Truck Driver','Other'])
Industry=st.selectbox("Select Industry",['Technology', 'Finance', 'Manufacturing', 'Healthcare', 'Retail',
       'Education', 'Transportation', 'Energy','Other'])
ai_score=st.number_input('AI Replacement Score',min_value=0.0)
skill_gap=st.number_input('Skillgap Index',min_value=0.0)
ai_adoption=st.number_input('AI Adoption Level',min_value=0.0)
remote_score=st.number_input('Remote Feasibility Score',min_value=0.0)
edu_level=st.selectbox("Select Level",['1','2','3','4','5'])
salary_change=st.number_input('Salary Change Percent')
skill_growth=st.number_input('Skill Demad Growth Percent')
wage_volatility=st.number_input('Wage Volatility Index')
disruption_intensity=st.number_input('Disruption Intensity')

if st.button("Predict"):
    input_data=pd.DataFrame([[job_role,Industry,ai_score,skill_gap,ai_adoption,remote_score,edu_level,salary_change,skill_growth,wage_volatility,disruption_intensity]],columns=['job_role','industry','ai_replacement_score','skill_gap_index','ai_adoption_level','remote_feasibility_score','education_requirement_level','salary_change_percent','skill_demand_growth_percent','wage_volatility_index','ai_disruption_intensity'])
    transformed_data=transformer.transform(input_data)
    prediction=model.predict(transformed_data)
    st.success(f"The Risk Category is: {prediction[0]}")
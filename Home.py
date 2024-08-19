import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Load the trained model
model_filename = 'random_forest_model_aids.pkl'
model = joblib.load(model_filename)

# Load data to get the raw feature names
file_path = 'AIDS_Classification_50000.csv'
data = pd.read_csv(file_path)

# Split data into features and target
X = data.drop('infected', axis=1)
y = data['infected']

# Calculate the mean of each column
column_means = X.mean()

# Mapping of feature names to more understandable labels with descriptions
feature_labels = {
    'age': 'Age (years)',
    'wtkg': 'Weight (kg)',
    'cd4': 'CD4 Count (immune system health)',
    'hemo': 'Hemoglobin Level (g/dL)',
    'visits': 'Number of Visits (doctor visits)',
    'symptom_count': 'Number of Symptoms (reported symptoms)',
    'gender': 'Gender (M/F)',
    'transmission': 'Transmission Mode (e.g., sexual contact, needle)',
    'viral_load': 'Viral Load (copies/mL)',
    'education': 'Education Level (years of education)',
    'marital_status': 'Marital Status (single/married)',
    'employment_status': 'Employment Status (employed/unemployed)',
    'alcohol_use': 'Alcohol Use (yes/no)',
    'drug_use': 'Drug Use (yes/no)',
    'adherence': 'Medication Adherence (%)',
    'duration_infected': 'Duration of Infection (years)',
    'smoking_status': 'Smoking Status (current/former/never)',
    'insurance_status': 'Insurance Status (insured/uninsured)',
    'income_level': 'Income Level (annual income)',
    'race': 'Race/Ethnicity',
    'residence': 'Residence (urban/rural)',
    'cd8': 'CD8 Count (immune system health)',
    'body_mass_index': 'Body Mass Index (BMI)',
    'blood_pressure': 'Blood Pressure (systolic/diastolic)',
    'cholesterol_level': 'Cholesterol Level (mg/dL)',
    'triglycerides': 'Triglycerides Level (mg/dL)',
    'glucose_level': 'Glucose Level (mg/dL)',
    'diabetes_status': 'Diabetes Status (yes/no)',
    'hypertension_status': 'Hypertension Status (yes/no)',
    'heart_disease_history': 'Heart Disease History (yes/no)',
    'mental_health_status': 'Mental Health Status (good/poor)',
    'housing_status': 'Housing Status (stable/unstable)',
    'physical_activity_level': 'Physical Activity Level (low/medium/high)',
}

# Streamlit app title and description
st.title("AIDS Prediction Application")
st.write("This app predicts the likelihood of AIDS infection based on provided features.")

# Create six columns
cols = st.columns(6)

# Split input fields across the six columns
def user_input_features():
    input_data = {}
    features = list(X.columns)
    for i, feature in enumerate(features):
        label = feature_labels.get(feature, feature)
        default_value = column_means[feature]
        
        col = cols[i % 6]  # Assign each feature to one of the six columns
        with col:
            if X[feature].dtype == 'float64' or X[feature].dtype == 'int64':
                input_data[feature] = st.number_input(f"{label}:", value=float(default_value))
            else:
                input_data[feature] = st.text_input(f"{label}:", value=str(default_value))

    return pd.DataFrame([input_data])

# Display input fields
input_df = user_input_features()

# Add a predict button
if st.button('Predict'):
    # Apply the same preprocessing techniques used during training

    # Feature scaling
    scaler = StandardScaler()
    scaler.fit(X)
    input_scaled = scaler.transform(input_df)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=10)
    selector.fit(X, y)
    input_selected = selector.transform(input_scaled)

    # Prediction
    prediction = model.predict(input_selected)
    prediction_proba = model.predict_proba(input_selected)

    # Display the prediction result
    st.subheader('Prediction')
    st.write('Infected' if prediction[0] else 'Not Infected')

    # Display the prediction probabilities
    st.subheader('Prediction Probability')
    st.write(f'Not Infected: {prediction_proba[0][0]:.2f}')
    st.write(f'Infected: {prediction_proba[0][1]:.2f}')

    # Provide feature importance information
    st.subheader('Selected Features for Model')
    selected_feature_names = [feature_labels.get(feature, feature) for feature in X.columns[selector.get_support()]]
    st.write(f"The model was trained using the following features: {selected_feature_names}")

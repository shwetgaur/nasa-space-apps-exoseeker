# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="ExoSeeker AI",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model and Data ---
@st.cache_resource
def load_model_and_data():
    """Load the pre-trained model and cleaned data."""
    model = joblib.load('exoplanet_model.joblib')
    df_clean = pd.read_csv('data/cleaned_koi_data.csv')
    # We need the feature names for the input form
    features = df_clean.drop('is_exoplanet', axis=1).columns.tolist()
    return model, df_clean, features

model, df_clean, features = load_model_and_data()

# --- App UI ---
st.title("ExoSeeker AI 🪐")
st.markdown("""
    Welcome to **ExoSeeker AI**! This tool uses a machine learning model to classify potential exoplanets 
    from NASA's Kepler mission data. Use the sidebar to input a candidate's data and see the prediction.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Input Candidate Data")
input_data = {}
for feature in features:
    # Create a number input for each feature
    input_data[feature] = st.sidebar.number_input(
        label=f"{feature}", 
        value=df_clean[feature].mean(), # Default to the mean value
        step=0.01
    )

# --- Prediction Logic ---
if st.sidebar.button("Classify Candidate"):
    # Convert the input dictionary to a DataFrame for the model
    input_df = pd.DataFrame([input_data])

    # Make a prediction
    prediction_proba = model.predict_proba(input_df)[0]
    prediction = model.predict(input_df)[0]

    # Display the results
    st.subheader("Classification Result")
    if prediction == 1:
        st.success(f"**Likely an Exoplanet** with a probability of {prediction_proba[1]:.2%}")
    else:
        st.error(f"**Likely a False Positive** with a probability of {prediction_proba[0]:.2%}")

# --- Data Visualization Section ---
st.markdown("---")
st.header("Explore the Kepler Dataset")
st.dataframe(df_clean.head())

# Example Plot: Relationship between Planetary Radius and Equilibrium Temperature
st.subheader("Planetary Radius vs. Equilibrium Temperature")
fig = px.scatter(
    df_clean, x='koi_teq', y='koi_prad', 
    color='is_exoplanet', 
    labels={'koi_teq': 'Equilibrium Temperature (K)', 'koi_prad': 'Planetary Radius (Earth Radii)', 'is_exoplanet': 'Is Exoplanet?'},
    title="Exoplanet Candidates Distribution",
    hover_name=df_clean.index
)
st.plotly_chart(fig, use_container_width=True)
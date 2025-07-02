import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, classification_report
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# 🔐 Set your OpenWeatherMap API key here
API_KEY = "46eab8d8bb29b175b36ecde37ce6c3f7"  # Replace this!

# 🌦️ Weather Fetch Function
def get_weather(city_name, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    temp = data['main']['temp']
    humidity = data['main']['humidity']
    rainfall = data.get('rain', {}).get('1h', 0.0)  # mm
    return temp, humidity, rainfall

# 📊 Simple 2x2 Confusion Matrix
def show_binary_confusion_matrix(y_true, y_pred, predicted_crop, model_name):
    """Show a simple 2x2 confusion matrix for the predicted crop vs all others"""
    
    # Convert to binary: predicted crop vs others
    y_true_binary = (y_true == y_pred[0]).astype(int)  # 1 if matches prediction, 0 otherwise
    y_pred_binary = np.ones_like(y_pred)  # All predictions are "positive" for the predicted crop
    y_pred_binary[y_pred != y_pred[0]] = 0  # Set non-matching predictions to 0
    
    # Calculate TP, TN, FP, FN
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    # Create 2x2 confusion matrix
    cm_2x2 = np.array([[tn, fp], [fn, tp]])
    
    st.markdown(f"### 📊 2x2 Confusion Matrix for {model_name}")
    st.markdown(f"**Predicted Crop: {predicted_crop}** vs All Others")
    
    # Display TP, TN, FP, FN values
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("True Positive (TP)", tp)
    with col2:
        st.metric("True Negative (TN)", tn)
    with col3:
        st.metric("False Positive (FP)", fp)
    with col4:
        st.metric("False Negative (FN)", fn)
    
    # Create simple 2x2 heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Labels for 2x2 matrix
    labels = [['TN', 'FP'], ['FN', 'TP']]
    annotations = [[f'{labels[i][j]}\n{cm_2x2[i][j]}' for j in range(2)] for i in range(2)]
    
    sns.heatmap(cm_2x2, annot=annotations, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Others', predicted_crop], 
                yticklabels=['Others', predicted_crop])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'2x2 Confusion Matrix\n{predicted_crop} vs Others')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
   

# Metrics calculation helper function with simple 2x2 confusion matrix
def get_metrics(model, X_test, y_test, model_name, label_encoder):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    st.markdown(f"### 📊 Overall Metrics for {model_name}:")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{acc:.3f}")
    with col2:
        st.metric("Precision", f"{precision:.3f}")
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    with col4:
        st.metric("F1 Score", f"{f1:.3f}")
    
    # Show simple 2x2 confusion matrix for most predicted crop
    most_predicted_crop_idx = np.bincount(y_pred).argmax()
    most_predicted_crop = label_encoder.inverse_transform([most_predicted_crop_idx])[0]
    
    show_binary_confusion_matrix(y_test, y_pred, most_predicted_crop, model_name)

# 🧠 Load and prepare data with preprocessing
@st.cache_data
def load_and_preprocess_data():
    # Load data
    data = pd.read_csv('crop_data.csv')
    
    # Separate features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Remove outliers using IQR method
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
    y = y[X.index]
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    
    return X, X_scaled, X_pca, y_encoded, le, scaler, pca

# 🌱 Streamlit UI
st.set_page_config(page_title="Enhanced Crop Recommender", layout="wide", page_icon="🌿")
st.title("🌱 YieldYoda - Enhanced Crop Recommendation System")
st.markdown("**Advanced ML with PCA preprocessing and detailed performance analysis**")

# Load and preprocess data
try:
    X_original, X_scaled, X_pca, y, le, scaler, pca = load_and_preprocess_data()
    
    # Display preprocessing info
    st.sidebar.markdown("### 🔧 Preprocessing Info")
    st.sidebar.write(f"Original features: {X_original.shape[1]}")
    st.sidebar.write(f"PCA components: {X_pca.shape[1]}")
    st.sidebar.write(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Model selection
    use_pca = st.sidebar.checkbox("Use PCA-transformed features", value=True)
    X_to_use = X_pca if use_pca else X_scaled
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_to_use, y, test_size=0.2, random_state=42)
    
    # Initialize models with better parameters
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(probability=True, random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[('Random Forest', rf), ('KNN', knn), ('SVM', svm)],
        voting='soft'
    )
    
    # Fit models on training data
    with st.spinner("Training models..."):
        ensemble_model.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        svm.fit(X_train, y_train)
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🧪 Input Parameters")
        
        # 🧪 Soil Nutrients Inputs
        N = st.number_input("Nitrogen level (N)", min_value=0.0, max_value=200.0, value=50.0)
        P = st.number_input("Phosphorus level (P)", min_value=0.0, max_value=150.0, value=50.0)
        K = st.number_input("Potassium level (K)", min_value=0.0, max_value=200.0, value=50.0)
        
        # ☁️ Weather Data Option
        city = st.text_input("🌍 Enter your city (for weather)")
        use_weather = st.checkbox("Use real-time weather data")
        
        # Set default values
        temperature, humidity, ph, rainfall = 25.0, 60.0, 7.0, 100.0
        
        if use_weather and city:
            weather = get_weather(city, API_KEY)
            if weather:
                temperature, humidity, rainfall = weather
                st.success(f"📡 Weather in {city.title()}: {temperature}°C, {humidity}% humidity, {rainfall} mm rain")
            else:
                st.error("❌ Could not fetch weather. Check city name or try again.")
        
        # Manual fallback inputs
        if not use_weather or not city:
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=temperature)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=humidity)
            ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=ph)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=rainfall)
    
    with col2:
        # 🌾 Predict and show results
        if st.button("🌾 Recommend Crop", type="primary"):
            # Prepare input data
            input_raw = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            input_scaled = scaler.transform(input_raw)
            
            if use_pca:
                input_processed = pca.transform(input_scaled)
            else:
                input_processed = input_scaled
            
            # Make predictions
            encoded_pred = ensemble_model.predict(input_processed)[0]
            final_crop = le.inverse_transform([encoded_pred])[0]
            
            # Get prediction probabilities
            proba = ensemble_model.predict_proba(input_processed)[0]
            top_3_idx = np.argsort(proba)[-3:][::-1]
            
            st.success(f"✅ **Recommended Crop: {final_crop}**")
            
            # Show top 3 predictions with confidence
            # st.markdown("### 🎯 Top 3 Recommendations:")
            # for i, idx in enumerate(top_3_idx):
            #     crop_name = le.inverse_transform([idx])[0]
            #     confidence = proba[idx] * 100
            #     st.write(f"{i+1}. **{crop_name}** - {confidence:.1f}% confidence")
    
            # Show individual model predictions
            rf_pred = le.inverse_transform(rf.predict(input_processed))[0]
            knn_pred = le.inverse_transform(knn.predict(input_processed))[0]
            svm_pred = le.inverse_transform(svm.predict(input_processed))[0]
            
            st.markdown("### 🧠 Individual Model Predictions:")
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            with pred_col1:
                st.write(f"🌲 **Random Forest**\n{rf_pred}")
            with pred_col2:
                st.write(f"🤖 **KNN**\n{knn_pred}")
            with pred_col3:
                st.write(f"🔍 **SVM**\n{svm_pred}")
    
    # Model Performance Analysis
    st.markdown("---")
    st.markdown("## 📈 Model Performance Analysis")
    
    # Model selection for analysis
    model_choice = st.selectbox(
        "Select model for detailed analysis:",
        ["Ensemble Voting Classifier", "Random Forest", "K-Nearest Neighbors", "Support Vector Machine"]
    )
    
    model_map = {
        "Ensemble Voting Classifier": ensemble_model,
        "Random Forest": rf,
        "K-Nearest Neighbors": knn,
        "Support Vector Machine": svm
    }
    
    selected_model = model_map[model_choice]
    get_metrics(selected_model, X_test, y_test, model_choice, le)
    
    

except FileNotFoundError:
    st.error("❌ crop_data.csv file not found. Please ensure the dataset is available.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.info("Please check your dataset format and try again.")
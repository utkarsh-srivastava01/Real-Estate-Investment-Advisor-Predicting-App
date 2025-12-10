import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🏠 Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════
# PROFESSIONAL 3D CSS STYLING WITH DARK THEME
# ════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─────────────────────── GENERAL ─────────────────────── */
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
    background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 50%, #1a1f2e 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #e0e0e0;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
}

/* ─────────────────────── HEADER ─────────────────────── */
.header-main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
    padding: 50px 40px;
    border-radius: 25px;
    margin-bottom: 30px;
    box-shadow: 
        0 20px 60px rgba(102, 126, 234, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.2),
        0 -5px 20px rgba(0, 0, 0, 0.3);
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.header-main::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.05) 50%, transparent 70%);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.header-main > * {
    position: relative;
    z-index: 1;
}

.header-title {
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -1px;
    margin-bottom: 10px;
    text-shadow: 0 4px 15px rgba(0,0,0,0.4);
}
.header-subtitle {
    font-size: 1.1rem;
    opacity: 0.98;
    font-weight: 300;
    letter-spacing: 0.5px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* ─────────────────────── SIDEBAR ─────────────────────── */
.sidebar-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 
        0 12px 40px rgba(102, 126, 234, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
}

.sidebar-header::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, transparent 50%);
    border-radius: 15px;
    pointer-events: none;
}

.sidebar-header h2 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 2;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.sidebar-header p {
    margin-top: 10px;
    opacity: 0.95;
    font-size: 0.9rem;
    position: relative;
    z-index: 2;
}

/* ─────────────────────── SECTION DIVIDER ─────────────────────── */
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #64b5f6;
    margin-top: 25px;
    margin-bottom: 15px;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(100, 181, 246, 0.6);
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

/* ─────────────────────── RESULTS CARD ─────────────────────── */
.result-card {
    border-radius: 18px;
    padding: 30px;
    box-shadow: 
        0 15px 50px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
    border-top: 5px solid #667eea;
    background: linear-gradient(135deg, #2a3f5f 0%, #1e2d42 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
}

.result-good {
    background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(39, 157, 63, 0.1) 100%);
    border-top: 5px solid #28a745;
    border: 1px solid rgba(40, 167, 69, 0.3);
}

.result-risky {
    background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 183, 0, 0.1) 100%);
    border-top: 5px solid #ffc107;
    border: 1px solid rgba(255, 193, 7, 0.3);
}

.result-label {
    font-size: 0.9rem;
    color: #64b5f6;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

.result-value {
    font-size: 2.8rem;
    font-weight: 900;
    margin: 15px 0;
    color: #64b5f6;
    text-shadow: 0 3px 10px rgba(100, 181, 246, 0.3);
}

.result-value-good { 
    color: #4caf50;
    text-shadow: 0 3px 10px rgba(76, 175, 80, 0.3);
}

.result-value-risky { 
    color: #ffc107;
    text-shadow: 0 3px 10px rgba(255, 193, 7, 0.3);
}

.result-confidence {
    font-size: 1.6rem;
    font-weight: 800;
    margin-top: 10px;
    color: #90caf9;
    text-shadow: 0 2px 8px rgba(144, 202, 249, 0.3);
}

/* ─────────────────────── METRIC CARD ─────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #2a3f5f 0%, #1e2d42 100%);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 
        0 10px 30px rgba(0,0,0,0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin: 10px;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 
        0 20px 50px rgba(102, 126, 234, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    border-color: rgba(102, 126, 234, 0.4);
}

.metric-label {
    font-size: 0.75rem;
    color: #90caf9;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

.metric-value {
    font-size: 2rem;
    font-weight: 900;
    color: #64b5f6;
    text-shadow: 0 2px 8px rgba(100, 181, 246, 0.3);
}

/* ─────────────────────── INFO BOX ─────────────────────── */
.info-box {
    background: linear-gradient(135deg, rgba(25, 118, 210, 0.15) 0%, rgba(21, 101, 192, 0.1) 100%);
    border-left: 5px solid #1976d2;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid rgba(25, 118, 210, 0.2);
    color: #90caf9;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.success-box {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(67, 160, 71, 0.1) 100%);
    border-left: 5px solid #4caf50;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid rgba(76, 175, 80, 0.2);
    color: #81c784;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.warning-box {
    background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 179, 0, 0.1) 100%);
    border-left: 5px solid #ffc107;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid rgba(255, 193, 7, 0.2);
    color: #ffb74d;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.info-box strong, .success-box strong, .warning-box strong {
    display: block;
    margin-bottom: 10px;
    font-size: 1.05rem;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

.info-box ul, .info-box ol,
.success-box ul, .success-box ol,
.warning-box ul, .warning-box ol {
    margin: 10px 0 0 20px;
    line-height: 2;
}

.info-box li, .success-box li, .warning-box li {
    margin-bottom: 8px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

/* ─────────────────────── BUTTON ─────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 15px 40px;
    border-radius: 12px;
    border: 2px solid rgba(255, 255, 255, 0.2);
    box-shadow: 
        0 12px 35px rgba(102, 126, 234, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    width: 100%;
    transition: all 0.3s ease;
    text-shadow: 0 2px 5px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.2) 0%, transparent 50%);
    pointer-events: none;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 
        0 18px 50px rgba(102, 126, 234, 0.45),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.stButton > button:active {
    transform: translateY(-1px);
}

/* ─────────────────────── DIVIDER ─────────────────────── */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(to right, #667eea, #764ba2, #667eea);
    margin: 30px 0;
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
}

/* ─────────────────────── FOOTER ─────────────────────── */
.footer {
    text-align: center;
    color: #90caf9;
    font-size: 0.9rem;
    margin-top: 50px;
    padding: 30px;
    border-top: 2px solid rgba(102, 126, 234, 0.3);
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    border-radius: 12px;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

.footer strong {
    color: #64b5f6;
    display: block;
    margin-bottom: 8px;
}

.footer small {
    display: block;
    margin-top: 5px;
    opacity: 0.8;
}

/* ─────────────────────── TEXT CONTRAST FIX ─────────────────────── */
p, span, div, label {
    color: #e0e0e0 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #64b5f6 !important;
    text-shadow: 0 2px 5px rgba(0,0,0,0.3);
}

strong {
    color: #90caf9 !important;
}

/* Streamlit input elements */
.stTextInput input,
.stNumberInput input,
.stSlider,
.stSelectbox select {
    background: linear-gradient(135deg, #2a3f5f 0%, #1e2d42 100%) !important;
    color: #e0e0e0 !important;
    border: 1px solid rgba(100, 181, 246, 0.3) !important;
    border-radius: 8px !important;
}

.stTextInput input::placeholder {
    color: rgba(224, 224, 224, 0.5) !important;
}

/* Success/Error messages */
.stSuccess, .stError, .stWarning, .stInfo {
    background: rgba(255, 255, 255, 0.08) !important;
    border-left: 4px solid;
    padding: 15px 20px !important;
    border-radius: 8px !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    """Load both trained models"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'models')
        
        with open(os.path.join(models_dir, 'best_classification_model.pkl'), 'rb') as f:
            class_model = pickle.load(f)
        
        with open(os.path.join(models_dir, 'best_regression_model.pkl'), 'rb') as f:
            reg_model = pickle.load(f)
        
        return class_model, reg_model
        
    except FileNotFoundError:
        st.error("""
        ❌ **ERROR: Models Not Found!**

        **Create this folder structure:**
        ```
        your_project_folder/
        ├── app.py (this file)
        └── models/
            ├── best_classification_model.pkl
            └── best_regression_model.pkl
        ```
        """)
        st.stop()

try:
    class_model, reg_model = load_models()
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

# ════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION & ENCODING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════
def get_model_features(model):
    """Extract feature names from trained model"""
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    elif hasattr(model, 'feature_names'):
        return list(model.feature_names)
    return []

def encode_categorical_features(data_dict):
    """
    Convert user-friendly text inputs to model-expected encoded values
    All encoding happens INTERNALLY - user never sees numbers!
    """
    encodings = {
        'Property_Type': {
            'Apartment': 0, 'Villa': 1, 'Independent House': 2, 
            'Townhouse': 3, 'Duplex': 4, 'Bungalow': 5
        },
        'Furnished_Status': {
            'Unfurnished': 0, 'Semi-Furnished': 1, 'Fully Furnished': 2
        },
        'Owner_Type': {
            'Individual': 0, 'Builder': 1, 'Agent': 2
        },
        'Facing': {
            'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Northeast': 4, 'Northwest': 5,
            'Southeast': 6, 'Southwest': 7
        },
        'Public_Transport_Accessibility': {
            'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3
        },
        'Parking_Space': {
            'No': 0, 'Limited': 1, 'Available': 2
        },
        'Security': {
            'None': 0, 'Basic': 1, 'Good': 2, 'Excellent': 3
        },
        'Availability_Status': {
            'Ready to Move': 0, 'Under Construction': 1, 'Resale': 2
        }
    }
    
    encoded_data = data_dict.copy()
    
    for feature_name, mapping in encodings.items():
        for key in encoded_data:
            if key.lower().replace('_', '') == feature_name.lower().replace('_', ''):
                if isinstance(encoded_data[key], str) and encoded_data[key] in mapping:
                    encoded_data[key] = mapping[encoded_data[key]]
    
    return encoded_data

def prepare_input_dataframe(model_features, user_inputs):
    """
    Create DataFrame with EXACT features model expects
    Handles missing/extra features intelligently
    """
    df = pd.DataFrame()
    
    for feature in model_features:
        found = False
        
        if feature in user_inputs:
            df[feature] = [user_inputs[feature]]
            found = True
        else:
            for key in user_inputs:
                if key.lower().replace('_', '') == feature.lower().replace('_', ''):
                    df[feature] = [user_inputs[key]]
                    found = True
                    break
        
        if not found:
            if any(x in feature.lower() for x in ['price', 'cost', 'value']):
                df[feature] = [25.0]
            elif any(x in feature.lower() for x in ['age', 'year']):
                df[feature] = [10]
            elif any(x in feature.lower() for x in ['size', 'sqft', 'area']):
                df[feature] = [1500]
            elif any(x in feature.lower() for x in ['bhk', 'bedroom']):
                df[feature] = [2]
            elif any(x in feature.lower() for x in ['floor', 'parking', 'school', 'hospital']):
                df[feature] = [3]
            else:
                df[feature] = [0]
    
    return df

# ════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════
def main():
    # HEADER
    st.markdown("""
    <div class="header-main">
        <div class="header-title">🏠 Real Estate Investment Advisor</div>
        <div class="header-subtitle">
            AI-Powered Property Profitability & Future Value Prediction System
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2>📋 Property Details</h2>
        <p>Fill all fields for accurate predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ════════════════════════════════════════════════════════════════════
    # PROPERTY INFORMATION SECTION
    # ════════════════════════════════════════════════════════════════════
    st.sidebar.markdown('<div class="section-header">🏢 Property Information</div>', unsafe_allow_html=True)
    
    property_type = st.sidebar.selectbox(
        "Property Type",
        ['Apartment', 'Villa', 'Independent House', 'Townhouse', 'Duplex', 'Bungalow']
    )
    
    # CHANGED: Slider -> Number input (decimals allowed)
    bhk = st.sidebar.number_input(
        "Number of Bedrooms (BHK)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.5,
        format="%.1f"
    )
    
    size_sqft = st.sidebar.number_input(
        "Property Size (Sq.Ft.)",
        min_value=0.0,
        max_value=100000.0,
        value=1500.0,
        step=10.0,
        format="%.2f"
    )
    
    # ════════════════════════════════════════════════════════════════════
    # PRICE & FINANCIAL SECTION
    # ════════════════════════════════════════════════════════════════════
    st.sidebar.markdown('<div class="section-header">💰 Price & Financial</div>', unsafe_allow_html=True)
    
    price_lakhs = st.sidebar.number_input(
        "Property Price (₹ Lakhs)",
        min_value=0.0,
        max_value=10000.0,
        value=50.0,
        step=0.5,
        format="%.2f"
    )
    
    price_per_sqft = (price_lakhs * 100000) / size_sqft if size_sqft > 0 else 0
    st.sidebar.metric("Calculated Price/SqFt", f"₹{price_per_sqft:,.2f}")
    
    # ════════════════════════════════════════════════════════════════════
    # PROPERTY AGE & CONSTRUCTION
    # ════════════════════════════════════════════════════════════════════
    st.sidebar.markdown('<div class="section-header">📅 Age & Construction</div>', unsafe_allow_html=True)
    
    year_built = st.sidebar.number_input(
        "Year Built",
        min_value=1900.0,
        max_value=2100.0,
        value=2015.0,
        step=0.5,
        format="%.1f"
    )
    age_property = 2025 - year_built
    st.sidebar.metric("Property Age", f"{age_property:.1f} years")
    
    floor_no = st.sidebar.number_input(
        "Floor Number",
        min_value=0.0,
        max_value=200.0,
        value=5.0,
        step=1.0,
        format="%.1f"
    )
    total_floors = st.sidebar.number_input(
        "Total Floors in Building",
        min_value=1.0,
        max_value=200.0,
        value=15.0,
        step=1.0,
        format="%.1f"
    )
    
    # ════════════════════════════════════════════════════════════════════
    # LOCATION & AMENITIES
    # ════════════════════════════════════════════════════════════════════
    st.sidebar.markdown('<div class="section-header">📍 Location & Amenities</div>', unsafe_allow_html=True)
    
    nearby_schools = st.sidebar.number_input(
        "Nearby Schools (Rating 0-10)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        format="%.1f"
    )
    nearby_hospitals = st.sidebar.number_input(
        "Nearby Hospitals (Rating 0-10)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.1,
        format="%.1f"
    )
    public_transport = st.sidebar.selectbox(
        "Public Transport Accessibility",
        ['Poor', 'Average', 'Good', 'Excellent']
    )
    
    # ════════════════════════════════════════════════════════════════════
    # INFRASTRUCTURE & FEATURES
    # ════════════════════════════════════════════════════════════════════
    st.sidebar.markdown('<div class="section-header">🏗️ Infrastructure & Features</div>', unsafe_allow_html=True)
    
    parking = st.sidebar.selectbox(
        "Parking Space",
        ['No', 'Limited', 'Available']
    )
    
    security = st.sidebar.selectbox(
        "Security Level",
        ['None', 'Basic', 'Good', 'Excellent']
    )
    
    facing = st.sidebar.selectbox(
        "Property Facing",
        ['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest']
    )
    
    # ════════════════════════════════════════════════════════════════════
    # AMENITIES & STATUS
    # ════════════════════════════════════════════════════════════════════
    st.sidebar.markdown('<div class="section-header">✨ Amenities & Status</div>', unsafe_allow_html=True)
    
    furnished = st.sidebar.selectbox(
        "Furnished Status",
        ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
    )
    
    owner_type = st.sidebar.selectbox(
        "Owner Type",
        ['Individual', 'Builder', 'Agent']
    )
    
    availability = st.sidebar.selectbox(
        "Availability Status",
        ['Ready to Move', 'Under Construction', 'Resale']
    )
    
    amenities = (nearby_schools + nearby_hospitals) / 2.0
    
    st.sidebar.markdown("---")
    
    # ════════════════════════════════════════════════════════════════════
    # PREPARE DATA INTERNALLY
    # ════════════════════════════════════════════════════════════════════
    user_inputs = {
        'BHK': float(bhk),
        'Size_in_SqFt': float(size_sqft),
        'Price_in_Lakhs': float(price_lakhs),
        'Price_per_SqFt': float(price_per_sqft),
        'Year_Built': float(year_built),
        'Age_of_Property': float(age_property),
        'Floor_No': float(floor_no),
        'Total_Floors': float(total_floors),
        'Nearby_Schools': float(nearby_schools),
        'Nearby_Hospitals': float(nearby_hospitals),
        'Amenities': float(amenities),
        'Public_Transport_Accessibility': public_transport,
        'Parking_Space': parking,
        'Security': security,
        'Facing': facing,
        'Property_Type': property_type,
        'Furnished_Status': furnished,
        'Owner_Type': owner_type,
        'Availability_Status': availability,
    }
    
    encoded_inputs = encode_categorical_features(user_inputs)
    
    class_features = get_model_features(class_model)
    reg_features = get_model_features(reg_model)
    
    X_class = prepare_input_dataframe(class_features, encoded_inputs)
    X_reg = prepare_input_dataframe(reg_features, encoded_inputs)
    
    # ════════════════════════════════════════════════════════════════════
    # PREDICT BUTTON
    # ════════════════════════════════════════════════════════════════════
    if st.sidebar.button("🚀 PREDICT & ANALYZE", use_container_width=True):
        try:
            class_pred = class_model.predict(X_class)[0]
            class_prob = class_model.predict_proba(X_class)[0]
            reg_pred = reg_model.predict(X_reg)[0]
            
            current_price = price_lakhs
            future_price = reg_pred
            appreciation = future_price - current_price
            roi = (appreciation / current_price * 100) if current_price > 0 else 0
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if class_pred == 1:
                    st.markdown(f"""
                    <div class="result-card result-card-good">
                        <div class="result-label">Investment Recommendation</div>
                        <div class="result-value result-value-good">✅ GOOD INVESTMENT</div>
                        <div class="result-confidence">Confidence: {class_prob[1]*100:.1f}%</div>
                        <p style="margin-top: 15px; color: #81c784; line-height: 1.6; font-size: 0.95rem;">
                            This property shows strong investment potential with favorable terms and market conditions.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-card-risky">
                        <div class="result-label">Investment Recommendation</div>
                        <div class="result-value result-value-risky">⚠️ MODERATE/RISKY</div>
                        <div class="result-confidence">Risk Level: {class_prob[0]*100:.1f}%</div>
                        <p style="margin-top: 15px; color: #ffb74d; line-height: 1.6; font-size: 0.95rem;">
                            This property has certain risk factors. Recommend further analysis before investment.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Estimated Future Value</div>
                    <div class="result-value">₹{future_price:,.2f} L</div>
                    <p style="margin-top: 15px; color: #90caf9; font-size: 0.9rem;">
                        <strong>Expected Value after 5 years</strong><br>
                        Based on market trends and property features
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("<h3 style='color: #64b5f6; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.3);'>📊 Financial Analysis</h3>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">₹{current_price:,.0f}L</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Future Price (5Y)</div>
                    <div class="metric-value">₹{future_price:,.0f}L</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Expected Appreciation</div>
                    <div class="metric-value" style="color: #4caf50; text-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);">₹{appreciation:,.0f}L</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">5-Year ROI</div>
                    <div class="metric-value">{roi:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("<h3 style='color: #64b5f6; text-shadow: 0 2px 8px rgba(0,0,0,0.3);'>📋 Property Summary</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <strong style="color: #64b5f6; text-shadow: 0 1px 3px rgba(0,0,0,0.3);">Physical Details</strong>
                    <hr style="border: none; height: 1px; background: rgba(100, 181, 246, 0.3); margin: 10px 0;">
                    <p style="margin: 0; line-height: 2; font-size: 0.95rem; color: #e0e0e0;">
                        <strong style="color: #90caf9;">Type:</strong> {property_type}<br>
                        <strong style="color: #90caf9;">BHK:</strong> {bhk} Bedrooms<br>
                        <strong style="color: #90caf9;">Size:</strong> {size_sqft:,} Sq.Ft.<br>
                        <strong style="color: #90caf9;">Age:</strong> {age_property:.1f} years<br>
                        <strong style="color: #90caf9;">Floor:</strong> {floor_no}/{total_floors}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <strong style="color: #64b5f6; text-shadow: 0 1px 3px rgba(0,0,0,0.3);">Location & Access</strong>
                    <hr style="border: none; height: 1px; background: rgba(100, 181, 246, 0.3); margin: 10px 0;">
                    <p style="margin: 0; line-height: 2; font-size: 0.95rem; color: #e0e0e0;">
                        <strong style="color: #90caf9;">Schools:</strong> {nearby_schools}/10<br>
                        <strong style="color: #90caf9;">Hospitals:</strong> {nearby_hospitals}/10<br>
                        <strong style="color: #90caf9;">Transport:</strong> {public_transport}<br>
                        <strong style="color: #90caf9;">Facing:</strong> {facing}<br>
                        <strong style="color: #90caf9;">Amenities:</strong> {amenities:.1f}/10
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="result-card">
                    <strong style="color: #64b5f6; text-shadow: 0 1px 3px rgba(0,0,0,0.3);">Infrastructure & Status</strong>
                    <hr style="border: none; height: 1px; background: rgba(100, 181, 246, 0.3); margin: 10px 0;">
                    <p style="margin: 0; line-height: 2; font-size: 0.95rem; color: #e0e0e0;">
                        <strong style="color: #90caf9;">Parking:</strong> {parking}<br>
                        <strong style="color: #90caf9;">Security:</strong> {security}<br>
                        <strong style="color: #90caf9;">Furnished:</strong> {furnished}<br>
                        <strong style="color: #90caf9;">Owner:</strong> {owner_type}<br>
                        <strong style="color: #90caf9;">Status:</strong> {availability}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("<h3 style='color: #64b5f6; text-shadow: 0 2px 8px rgba(0,0,0,0.3);'>💡 Smart Investment Recommendations</h3>", unsafe_allow_html=True)
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                if class_pred == 1:
                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Investment Strengths</strong>
                        <ul style="margin: 10px 0 0 20px; line-height: 2;">
                            <li>Favorable pricing and market conditions</li>
                            <li>Strong appreciation potential</li>
                            <li>Good location with amenities</li>
                            <li>Reasonable property age</li>
                            <li>Multi-bedroom configuration</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Investment Considerations</strong>
                        <ul style="margin: 10px 0 0 20px; line-height: 2;">
                            <li>Higher price per unit</li>
                            <li>Older property - maintenance risk</li>
                            <li>Limited appreciation potential</li>
                            <li>Additional analysis recommended</li>
                            <li>Compare with similar properties</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown("""
                <div class="info-box">
                    <strong>📋 Next Steps for Investment</strong>
                    <ol style="margin: 10px 0 0 20px; line-height: 2;">
                        <li>Verify property documents & legal status</li>
                        <li>Get professional property inspection done</li>
                        <li>Review neighborhood trends & growth</li>
                        <li>Check RERA compliance & authenticity</li>
                        <li>Consult with real estate advisor</li>
                        <li>Review market comparison data</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            
            st.success("✅ **Prediction completed successfully!** Use this analysis as a supporting tool alongside professional real estate advice.")
        
        except Exception as e:
            st.error(f"❌ **Prediction Error:** {str(e)}")
            st.info("Please ensure all models are properly loaded. If error persists, check your data format.")
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <strong>🏠 Real Estate Investment Advisor v4.1</strong><br>
        <small>AI-Powered Property Profitability & Future Value Prediction System</small><br>
        <small style="opacity: 0.7;">Production Ready | Data Science Powered | Enhanced 3D Visuals | 2025</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from utils.data_preprocessing import DataPreprocessor
from utils.model_builder import AlzheimerModel
import pandas as pd
from datetime import datetime
import json
import os
# Set page config
st.set_page_config(
    page_title="Alzheimer's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Initialize session state for storing results and form data
if 'results_history' not in st.session_state:
    # Load existing history if available
    if os.path.exists('results/history.json'):
        with open('results/history.json', 'r') as f:
            st.session_state.results_history = json.load(f)
    else:
        st.session_state.results_history = []
if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        "patient_id": "",
        "age": 0,
        "gender": "Male",
        "medical_history": ""
    }

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #f5f7fa;
        padding: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1a365d;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #2c5282;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #2b6cb0;
        font-size: 1.4rem;
        margin-top: 1.5rem;
    }
    
    /* Upload area */
    .upload-area {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #4299e1;
        text-align: center;
    }
    
    /* Prediction box */
    .prediction-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(66, 153, 225, 0.1);
        border-left: 4px solid #4299e1;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4299e1;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #3182ce;
        transform: translateY(-2px);
    }
    
    /* Footer */
    .footer {
        background-color: #ebf8ff;
        padding: 1rem;
        text-align: center;
        margin-top: 2rem;
        border-top: 1px solid #bee3f8;
    }
    
    /* Confidence indicator */
    .confidence {
        font-size: 1.2rem;
        color: #2b6cb0;
        font-weight: 600;
    }
    
    /* Confidence explanation */
    .confidence-explanation {
        background-color: #ebf8ff;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #4a5568;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize model and preprocessor
@st.cache_resource
def load_model():
    model = AlzheimerModel()
    model.load_model('models/final_model.h5')
    return model

@st.cache_resource
def load_preprocessor():
    return DataPreprocessor()

def get_confidence_level(confidence):
    if confidence >= 90:
        return "Very High"
    elif confidence >= 70:
        return "High"
    elif confidence >= 50:
        return "Moderate"
    else:
        return "Low"

def clear_form():
    """Clear the patient information form"""
    st.session_state.form_data = {
        "patient_id": "",
        "age": 0,
        "gender": "Male",
        "medical_history": ""
    }
    st.session_state.patient_info = None

def save_results(patient_info, prediction_results):
    """Save results to a JSON file"""
    result = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'patient_info': patient_info,
        'prediction': prediction_results
    }
    st.session_state.results_history.append(result)
    
    # Save to file
    os.makedirs('results', exist_ok=True)
    with open('results/history.json', 'w') as f:
        json.dump(st.session_state.results_history, f)

def export_to_csv():
    """Export results history to CSV"""
    if not st.session_state.results_history:
        return None
    
    df = pd.DataFrame([
        {
            'Date': r['timestamp'],
            'Patient ID': r['patient_info'].get('patient_id', '') if r.get('patient_info') is not None else '',
            'Age': r['patient_info'].get('age', '') if r.get('patient_info') is not None else '',
            'Gender': r['patient_info'].get('gender', '') if r.get('patient_info') is not None else '',
            'Diagnosis': r['prediction']['pred_class'],
            'Confidence': f"{r['prediction']['confidence']:.2f}%"
        }
        for r in st.session_state.results_history
    ])
    
    return df.to_csv(index=False)

def display_results_history(preprocessor):
    """Display the complete results history with filtering options"""
    st.markdown("<h2>Analysis History</h2>", unsafe_allow_html=True)
    
    # Add filtering options
    col1, col2, col3 = st.columns(3)
    with col1:
        search_id = st.text_input("Search by Patient ID")
    with col2:
        date_range = st.date_input(
            "Filter by Date Range",
            value=(datetime.now().date(), datetime.now().date()),
            max_value=datetime.now().date()
        )
    with col3:
        diagnosis_filter = st.selectbox(
            "Filter by Diagnosis",
            options=["All"] + list(set(r['prediction']['pred_class'] for r in st.session_state.results_history))
        )
    
    # Filter results
    filtered_results = st.session_state.results_history
    if search_id:
        filtered_results = [r for r in filtered_results if r.get('patient_info') is not None and search_id.lower() in r['patient_info'].get('patient_id', '').lower()]
    if date_range:
        start_date, end_date = date_range
        filtered_results = [
            r for r in filtered_results 
            if start_date <= datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S").date() <= end_date
        ]
    if diagnosis_filter != "All":
        filtered_results = [r for r in filtered_results if r['prediction']['pred_class'] == diagnosis_filter]
    
    # Display results
    if not filtered_results:
        st.info("No results found matching the selected filters.")
        return
    
    # Sort results by date (newest first)
    filtered_results.sort(key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S"), reverse=True)
    
    # Display results in a table format
    results_data = []
    for result in filtered_results:
        # Safely access patient info, defaulting to empty dict if None
        patient_info = result.get('patient_info') if result.get('patient_info') is not None else {}
        
        # Parse timestamp into date and time
        timestamp_dt = datetime.strptime(result['timestamp'], "%Y-%m-%d %H:%M:%S")
        analysis_date = timestamp_dt.strftime("%Y-%m-%d")
        analysis_time = timestamp_dt.strftime("%H:%M:%S")
        
        results_data.append({
            'Date': analysis_date,
            'Time': analysis_time,
            'Patient ID': patient_info.get('patient_id', 'N/A'),
            'Age': patient_info.get('age', 'N/A'),
            'Gender': patient_info.get('gender', 'N/A'),
            'Diagnosis': result['prediction']['pred_class'],
            'Confidence': f"{result['prediction']['confidence']:.2f}%"
        })
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)
    
    # Add detailed view for each result
    st.markdown("<h3>Detailed Results</h3>", unsafe_allow_html=True)
    for result in filtered_results:
        # Safely access patient info for detailed view
        patient_info = result.get('patient_info') if result.get('patient_info') is not None else {}
        with st.expander(f"Result from {result['timestamp']} - Patient ID: {patient_info.get('patient_id', 'N/A')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Patient Information**")
                st.write(f"Patient ID: {patient_info.get('patient_id', 'N/A')}")
                st.write(f"Age: {patient_info.get('age', 'N/A')}")
                st.write(f"Gender: {patient_info.get('gender', 'N/A')}")
                st.write(f"Medical History: {patient_info.get('medical_history', 'N/A')}")
            
            with col2:
                st.markdown("**Analysis Results**")
                st.write(f"Diagnosis: {result['prediction']['pred_class']}")
                st.write(f"Confidence: {result['prediction']['confidence']:.2f}%")
                
                # Display probability breakdown
                st.markdown("**Probability Breakdown**")
                for class_name, prob in zip(preprocessor.class_names, result['prediction']['pred_probs']):
                    st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        ## About
        This application uses advanced deep learning to analyze MRI scans 
        of the hippocampus for early detection of Alzheimer's Disease.
        
        ### How to Use
        1. Enter patient information
        2. Upload an MRI scan
        3. Click 'Analyze Image'
        4. Review results
        5. Clear form for next patient
        
        ### Model Information
        - Architecture: EfficientNetV2
        - Accuracy: ~88%
        - Classes: 4 stages of Alzheimer's
        """)
        
        # Add export functionality in sidebar
        if st.session_state.results_history:
            st.markdown("### Export Results")
            csv = export_to_csv()
            if csv:
                st.download_button(
                    label="Download Complete History (CSV)",
                    data=csv,
                    file_name="alzheimer_results.csv",
                    mime="text/csv"
                )
    
    # Main content
    st.markdown("<h1>🧠 Alzheimer's Disease Detection System</h1>", unsafe_allow_html=True)
    
    # Patient Information Form
    st.markdown("<h2>Patient Information</h2>", unsafe_allow_html=True)
    with st.form("patient_info_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID", value=st.session_state.form_data["patient_id"])
            age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.form_data["age"])
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.form_data["gender"]))
            medical_history = st.text_area("Medical History", value=st.session_state.form_data["medical_history"])
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Save Patient Information")
        with col2:
            clear_button = st.form_submit_button("Clear Form")
        
        if submitted:
            st.session_state.patient_info = {
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "medical_history": medical_history
            }
            st.session_state.form_data = {
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "medical_history": medical_history
            }
            st.success("Patient information saved!")
        
        if clear_button:
            clear_form()
            st.success("Form cleared!")
            st.experimental_rerun()
    
    # Load model and preprocessor
    model = load_model()
    preprocessor = load_preprocessor()
    
    # Create main content area
    st.markdown("<h2>Upload MRI Scan</h2>", unsafe_allow_html=True)
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display the uploaded image
        # Use the uploaded_file object directly without saving to a temporary file initially
        image = Image.open(uploaded_file)
        
        # Resize the image for display (optional, based on previous discussion)
        original_width, original_height = image.size
        target_width = 600 # Choose a target width for display
        # Calculate new height maintaining aspect ratio
        target_height = int(original_height * (target_width / original_width))
        resized_image_display = image.resize((target_width, target_height))
        
        st.image(resized_image_display, caption="Uploaded MRI Scan", use_column_width=False)
        
        # Add analyze button
        if st.button("Analyze Image", key="analyze_button"):
            if not hasattr(st.session_state, 'patient_info'):
                st.error("Please save patient information before analyzing the image!")
                return
                
            # Save the uploaded file temporarily just for preprocessing
            temp_path = "temp_image.jpg"
            
            # Convert image to RGB if it's in RGBA mode before saving as JPEG
            if image.mode == 'RGBA':
                image = image.convert('RGB')
                
            image.save(temp_path)
            
            # Preprocess the image using the temporary file path
            processed_image = preprocessor.preprocess_single_image(temp_path)

            # Make prediction
            pred_probs = model.model.predict(processed_image)[0]
            pred_class_idx = np.argmax(pred_probs)
            pred_class = preprocessor.class_names[pred_class_idx]
            confidence = pred_probs[pred_class_idx] * 100
            
            # Remove temporary file immediately after preprocessing
            import os
            os.remove(temp_path)
            
            # Store results in session state
            results = {
                'pred_class': pred_class,
                'confidence': confidence,
                'pred_probs': pred_probs.tolist()
            }
            st.session_state['results'] = results
            
            # Store image info in session state, including the temporary path (will be used for thumbnail if needed before deletion)
            # Remove or comment out this section if not needed elsewhere
            # st.session_state['last_analyzed_image_info'] = {
            #     'name': uploaded_file.name,
            #     'size': uploaded_file.size,
            #     'type': uploaded_file.type,
            #     'width': original_width,
            #     'height': original_height,
            #     'temp_path': temp_path # Store temp path for thumbnail display if needed immediately
            # }

            # We don't need to store the thumbnail image object anymore since we'll load from the temp file
            # del st.session_state['last_analyzed_image_thumbnail']
            
            # Save results if patient info exists
            if hasattr(st.session_state, 'patient_info'):
                save_results(st.session_state.patient_info, results)
                # Clear the form after successful analysis
                clear_form()
                st.success("Analysis complete! Form cleared for next patient.")
                st.experimental_rerun()
            
            st.markdown("<h2>Analysis Results</h2>", unsafe_allow_html=True)
            
            # Prediction box
            confidence_level = get_confidence_level(confidence)
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Primary Diagnosis</h3>
                <p style='font-size: 1.4rem; margin: 1rem 0;'><strong>{pred_class}</strong></p>
                <p class="confidence">Confidence: {confidence:.2f}% ({confidence_level})</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add detailed probability breakdown
            st.markdown("<h3>Detailed Analysis</h3>", unsafe_allow_html=True)
            for i, (class_name, prob) in enumerate(zip(preprocessor.class_names, pred_probs)):
                st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")
            
            brain_context = ""  
            if confidence_level == "Very High":
                brain_context = f"<p style='color: #444;'>Based on the very high confidence, the scan shows strong indicators often associated with <strong>{pred_class}</strong>, potentially including significant atrophy in regions like the hippocampus.</p>"
            elif confidence_level == "High":
                brain_context = f"<p style='color: #444;'>With high confidence, the AI identified patterns consistent with <strong>{pred_class}</strong>, which may involve noticeable changes in brain structure.</p>"
            elif confidence_level == "Moderate":
                brain_context = f"<p style='color: #444;'>The AI has moderate confidence in predicting <strong>{pred_class}</strong>. Features may be subtle, and further medical review is recommended.</p>"
            else: # Low confidence
                brain_context = f"<p style='color: #444;'>The AI's low confidence suggests the features in the scan are not clearly indicative of a specific stage, or the image quality may be suboptimal. A medical expert's assessment is essential.</p>"

            st.markdown(f"""
            <div style='background-color: #e9ecef; padding: 1rem; border-radius: 8px; margin-top: 1rem; font-size: 0.9rem; color: #555;'>
                <h4>Brain Imaging Context</h4>
                {brain_context}
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence explanation
            st.markdown("""
            <div class="confidence-explanation">
                <h4>About Confidence Score</h4>
                <p>The confidence score indicates how certain the AI model is about its prediction:</p>
                <ul>
                    <li><strong>Very High (90-100%)</strong>: The model is very confident in its prediction</li>
                    <li><strong>High (70-89%)</strong>: The model is confident in its prediction</li>
                    <li><strong>Moderate (50-69%)</strong>: The model is somewhat confident in its prediction</li>
                    <li><strong>Low (0-49%)</strong>: The model is less confident in its prediction</li>
                </ul>
                <p>Note: A higher confidence score doesn't necessarily mean the prediction is more accurate. 
                It only indicates how certain the model is about its prediction.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display the most recent analysis summary card
    if 'results' in st.session_state and st.session_state.results:
        st.markdown("<h2>Recent Analysis Summary</h2>", unsafe_allow_html=True)
        results = st.session_state.results
        confidence_level = get_confidence_level(results['confidence'])
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Diagnosis: {results['pred_class']}</h3>
            <p class="confidence">Confidence: {results['confidence']:.2f}% ({confidence_level})</p>
        </div>
        """, unsafe_allow_html=True)

    # Overview about Alzheimer's Disease
    # Display only if patient information has been saved
    if hasattr(st.session_state, 'patient_info') and st.session_state.patient_info:
        patient_name = st.session_state.patient_info.get('patient_id', 'N/A Patient')
        st.markdown(f"<h2>Overview for Patient {patient_name}</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color: #e9ecef; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; font-size: 0.9rem; color: #555;'>
            <h4>Understanding Alzheimer's Disease</h4>
            <p>Alzheimer's disease is a progressive neurological disorder that causes the brain to shrink and brain cells to die. It is the most common cause of dementia—a gradual decline in memory, thinking, behavior, and social skills that affects a person's ability to function independently.</p>
            <p>The stages of Alzheimer's often follow a pattern, although the speed of progression can vary:</p>
            <ul>
                <li><strong>Non Demented (Healthy)</strong>: No cognitive impairment.</li>
                <li><strong>Very Mild Impairment</strong>: Slight memory problems or changes in thinking that do not yet affect daily life.</li>
                <li><strong>Mild Impairment</strong>: More noticeable memory and thinking issues; may get lost, struggle with money or complex tasks.</li>
                <li><strong>Moderate Impairment</strong>: Significant challenges with daily activities, communication, and recognizing familiar people.</li>
            </ul>
            <p>Early and accurate detection is crucial for timely intervention and management.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display complete results history
    if st.session_state.results_history:
        display_results_history(preprocessor)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style='color: #666;'>This tool is for research purposes only. Please consult a medical professional for diagnosis.</p>
        <p style='color: #999; font-size: 0.8rem;'>© 2024 Alzheimer's Detection System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
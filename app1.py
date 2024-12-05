import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import ResNet50V2, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import os

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MarineConservationML:
    def __init__(self):
        """
        Initialize the Marine Conservation Machine Learning Application
        with pre-trained models and configuration
        """
        # Plastic Waste Detection Model
        self.plastic_classes = [
            'Plastic Bottle', 'Plastic Bag', 'Plastic Container', 
            'Plastic Wrapper', 'Plastic Packaging', 'Other Plastic Waste'
        ]
        
        # Pre-trained model paths (create these if not exists)
        self.plastic_model_path = 'plastic_waste_model.h5'
        self.coral_model_path = 'coral_health_model.h5'
        
        # Initialize or load models
        self.plastic_waste_model = self.load_plastic_waste_model()
        self.coral_health_model = self.load_coral_health_model()
        self.hab_model = self.train_hab_prediction_model()

    def create_base_model(self, num_classes):
        """
        Create a base transfer learning model
        """
        # Use ResNet50V2 for transfer learning
        base_model = ResNet50V2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        return model

    def load_plastic_waste_model(self):
        """
        # Load or create plastic waste detection model
        """
        try:
            # Try to load existing model
            if os.path.exists(self.plastic_model_path):
                return load_model(self.plastic_model_path)
            
            # Create and save new model if not exists
            model = self.create_base_model(len(self.plastic_classes))
            model.save(self.plastic_model_path)
            return model
        except Exception as e:
            st.error(f"Error loading plastic waste model: {e}")
            return self.create_base_model(len(self.plastic_classes))

    def load_coral_health_model(self):
        """
        Load or create coral health monitoring model
        """
        try:
            # Health status categories
            health_classes = ["Healthy", "Bleached", "Damaged"]
            
            # Try to load existing model
            if os.path.exists(self.coral_model_path):
                return load_model(self.coral_model_path)
            
            # Create and save new model if not exists
            model = self.create_base_model(len(health_classes))
            model.save(self.coral_model_path)
            return model
        except Exception as e:
            st.error(f"Error loading coral health model: {e}")
            return self.create_base_model(3)  # 3 health classes

    def train_hab_prediction_model(self):
        """
        Train a Random Forest Regressor for HAB prediction
        """
        # More comprehensive environmental data
        X = np.array([
            [20, 10, 2, 7.5, 35],    # Low risk scenario
            [22, 12, 3, 8.0, 38],
            [25, 15, 1, 8.5, 40],     # Moderate risk
            [28, 20, 4, 9.0, 42],     # High risk scenario
            [30, 25, 5, 9.5, 45]
        ])
        y = np.array([0.2, 0.3, 0.6, 0.8, 0.9])  # HAB probability

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        return model, scaler

    def preprocess_image(self, uploaded_file):
        """
        Preprocess uploaded image for model input
        """
        try:
            # Open image
            img = Image.open(uploaded_file)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize and normalize
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            return np.expand_dims(img_array, axis=0)
        except Exception as e:
            st.error(f"Image preprocessing error: {e}")
            return None

    def plastic_waste_detection(self, uploaded_file):
        """
        Detect and classify plastic waste from an uploaded image
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(uploaded_file)
            
            if img_array is None:
                return "Detection Failed", np.zeros(len(self.plastic_classes))

            # Predict
            prediction = self.plastic_waste_model.predict(img_array)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            
            return (
                self.plastic_classes[predicted_class_index], 
                prediction[0]
            )
        except Exception as e:
            st.error(f"Plastic waste detection error: {e}")
            return "Detection Failed", np.zeros(len(self.plastic_classes))

    def coral_reef_health_analysis(self, uploaded_file):
        """
        Comprehensive coral reef health monitoring with detailed insights
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(uploaded_file)
            
            if img_array is None:
                return {"status": "Analysis Failed", "confidence": 0, "recommendation": ""}

            # Predict
            prediction = self.coral_health_model.predict(img_array)
            health_status = ["Healthy", "Bleached", "Damaged"]
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][predicted_class]

            # Detailed health recommendations
            recommendations = {
                "Healthy": "Coral looks vibrant and shows no signs of stress. Continue current conservation efforts.",
                "Bleached": "Coral is experiencing stress. Reduce water temperature and monitor nutrient levels closely.",
                "Damaged": "Severe coral damage detected. Immediate intervention required. Consider coral restoration techniques."
            }

            return {
                "status": health_status[predicted_class],
                "confidence": float(confidence),
                "recommendation": recommendations[health_status[predicted_class]]
            }
        except Exception as e:
            st.error(f"Coral reef health analysis error: {e}")
            return {"status": "Analysis Failed", "confidence": 0, "recommendation": ""}

    def hab_prediction(self, temperature, chlorophyll, nutrients, ph, salinity):
        """
        Predict Harmful Algal Bloom with detailed risk analysis
        """
        try:
            # Prepare input data
            input_data = np.array([[temperature, chlorophyll, nutrients, ph, salinity]])
            scaler = self.hab_model[1]
            model = self.hab_model[0]

            # Scale and predict
            input_scaled = scaler.transform(input_data)
            hab_probability = model.predict(input_scaled)[0]

            # Generate visualization
            plt.figure(figsize=(10, 6))
            plt.title("Harmful Algal Bloom Risk Analysis")
            
            # Risk level visualization
            risk_levels = ['Low', 'Moderate', 'High']
            risk_colors = ['green', 'yellow', 'red']
            
            plt.bar(risk_levels, 
                    [max(0, 1-hab_probability), 
                     abs(hab_probability - 0.5), 
                     hab_probability], 
                    color=risk_colors)
            plt.ylabel("Risk Probability")
            plt.ylim(0, 1)

            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)

            return {
                "probability": hab_probability,
                "risk_level": "Low" if hab_probability < 0.3 else "Moderate" if hab_probability < 0.7 else "High",
                "risk_plot": buf
            }
        except Exception as e:
            st.error(f"HAB prediction error: {e}")
            return {"probability": 0, "risk_level": "Error", "risk_plot": None}

def main():
    # Page configuration
    st.set_page_config(page_title="Marine Conservation ML", layout="wide")
    st.sidebar.title("Marine Conservation Projects")
    
    # Initialize the ML application
    marine_ml = MarineConservationML()

    # Project selection
    project_choice = st.sidebar.selectbox(
        "Select a Project",
        ("Plastic Waste Detection", "Coral Reef Health Monitoring", "Harmful Algal Bloom Prediction")
    )

    # Plastic Waste Detection
    if project_choice == "Plastic Waste Detection":
        st.title("Plastic Waste Detection")
        uploaded_file = st.file_uploader("Upload Marine Waste Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Detect plastic waste
            result, confidence = marine_ml.plastic_waste_detection(uploaded_file)
            
            st.write(f"Prediction: {result}")
            
            # Confidence visualization
            plt.figure(figsize=(10, 6))
            plt.bar(marine_ml.plastic_classes, confidence)
            plt.title("Plastic Waste Classification Confidence")
            plt.xlabel("Waste Categories")
            plt.ylabel("Confidence")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(plt)

    # Coral Reef Health Monitoring
    elif project_choice == "Coral Reef Health Monitoring":
        st.title("Coral Reef Health Monitoring")
        uploaded_file = st.file_uploader("Upload Coral Reef Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Analyze coral health
            health_result = marine_ml.coral_reef_health_analysis(uploaded_file)
            
            st.write(f"Health Status: {health_result['status']}")
            st.write(f"Confidence: {health_result.get('confidence', 0):.2%}")
            st.write(f"Recommendation: {health_result['recommendation']}")

    # Harmful Algal Bloom Prediction
    elif project_choice == "Harmful Algal Bloom Prediction":
        st.title("Harmful Algal Bloom Prediction")
        
        # Input columns
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.number_input("Water Temperature (°C)", min_value=0.0, max_value=40.0, value=25.0)
            chlorophyll = st.number_input("Chlorophyll Concentration (µg/L)", min_value=0.0, max_value=100.0, value=5.0)
        
        with col2:
            nutrients = st.number_input("Nutrient Levels (mg/L)", min_value=0.0, max_value=10.0, value=2.0)
            ph = st.number_input("Water pH", min_value=6.0, max_value=9.0, value=8.0)
        
        salinity = st.number_input("Salinity (ppt)", min_value=0.0, max_value=40.0, value=35.0)
        
        if st.button("Predict HAB Likelihood"):
            hab_result = marine_ml.hab_prediction(temperature, chlorophyll, nutrients, ph, salinity)
            
            st.write(f"HAB Probability: {hab_result['probability']:.2%}")
            st.write(f"Risk Level: {hab_result['risk_level']}")
            
            # Display risk visualization
            if hab_result['risk_plot']:
                st.image(hab_result['risk_plot'], caption="Harmful Algal Bloom Risk Analysis")

    # Footer
    st.sidebar.text("Marine Ecosystem Conservation")

if __name__ == "__main__":
    main()
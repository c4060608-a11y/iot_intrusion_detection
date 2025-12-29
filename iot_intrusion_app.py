import streamlit as st
import joblib
import torch
import torch.nn as nn
import numpy as np

# --- 1. Model Architecture (Must match your training script) ---
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# --- 2. Configuration & Asset Loading ---
st.set_page_config(page_title="IoT Attack Detector", layout="wide")

@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load("iot_scaled.pkl")
        label_encoder = joblib.load("iot_label_encoder.pkl")
        
        input_dim = 46 
        num_classes = len(label_encoder.classes_)
        
        model = MLP(input_dim, num_classes)
        model.load_state_dict(torch.load("iot_mlp_model_weights.pth", map_location=torch.device('cpu')))
        model.eval()
        
        return scaler, label_encoder, model
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

SCALER, LABEL_ENCODER, MODEL = load_assets()

# --- 3. UI Layout ---
st.title("🛡️ IoT Network Intrusion Detection System")
st.markdown("Enter network flow parameters below to predict if the traffic is benign or an attack.")

if SCALER:
    # We group features logically to make the UI cleaner
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Flow Metrics")
            flow_duration = st.number_input("Flow Duration", value=0.0)
            header_length = st.number_input("Header Length", value=54.0)
            protocol_type = st.number_input("Protocol Type", value=6.0)
            duration = st.number_input("Duration", value=64.0)
            rate = st.number_input("Rate", value=0.33)
            srate = st.number_input("Srate", value=0.33)
            drate = st.number_input("Drate", value=0.0)
            iat = st.number_input("IAT (Inter-Arrival Time)", value=83343831.0)

        with col2:
            st.subheader("Flags & Counts")
            f1, f2 = st.columns(2)
            fin_flag = f1.selectbox("FIN Flag", [0, 1], index=0)
            syn_flag = f2.selectbox("SYN Flag", [0, 1], index=0)
            rst_flag = f1.selectbox("RST Flag", [0, 1], index=0)
            psh_flag = f2.selectbox("PSH Flag", [0, 1], index=0)
            ack_flag = f1.selectbox("ACK Flag", [0, 1], index=0)
            ece_flag = f2.selectbox("ECE Flag", [0, 1], index=0)
            cwr_flag = f1.selectbox("CWR Flag", [0, 1], index=0)
            
            ack_count = st.number_input("ACK Count", value=0.0)
            syn_count = st.number_input("SYN Count", value=0.0)
            fin_count = st.number_input("FIN Count", value=0.0)
            urg_count = st.number_input("URG Count", value=0.0)
            rst_count = st.number_input("RST Count", value=0.0)

        with col3:
            st.subheader("Protocols & Stats")
            # Simplified protocol selection (mapped to binary 0/1)
            http = st.selectbox("HTTP", [0, 1], index=0)
            https = st.selectbox("HTTPS", [0, 1], index=0)
            tcp = st.selectbox("TCP", [0, 1], index=0)
            udp = st.selectbox("UDP", [0, 1], index=0)
            
            # The rest of the stats
            tot_sum = st.number_input("Total Sum", value=0.0)
            min_val = st.number_input("Min", value=0.0)
            max_val = st.number_input("Max", value=0.0)
            avg_val = st.number_input("Average", value=54.0)
            tot_size = st.number_input("Total Size", value=54.0)
            weight = st.number_input("Weight", value=0.0)

        # Remaining fields (Hidden or default values to match 46 features)
        # Note: You can add these to the UI if users need to tune them
        remaining_features = [
            0, 0, 0, 0, 0, 0, # DNS, Telnet, SMTP, SSH, IRC
            0, 0, 0, # DHCP, ARP, ICMP
            1, 1, # IPv, LLC
            0.0, # Std
            9.5, # Number
            10.39, # Magnitude
            0.0, 0.0, 0.0 # Radius, Covariance, Variance
        ]

        submit = st.form_submit_button("Analyze Traffic", use_container_width=True)

    # --- 4. Prediction Logic ---
    if submit:
        # Construct the 46-feature list in the exact order required by the model
        input_data = [
            flow_duration, header_length, protocol_type, duration, rate, srate, drate,
            fin_flag, syn_flag, rst_flag, psh_flag, ack_flag, ece_flag, cwr_flag,
            ack_count, syn_count, fin_count, urg_count, rst_count,
            http, https, 0, 0, 0, 0, 0, tcp, udp, 0, 0, 0, 1, 1, # Including static placeholders
            tot_sum, min_val, max_val, avg_val, 0.0, tot_size, iat, 
            9.5, 10.39, 0.0, 0.0, 0.0, weight
        ]
        
        # Ensure we have exactly 46 features
        if len(input_data) == 46:
            input_array = np.array([input_data], dtype=np.float32)
            scaled_input = SCALER.transform(input_array)
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

            with torch.no_grad():
                output = MODEL(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                conf, pred_idx = torch.max(probabilities, 1)
                
                label = LABEL_ENCODER.inverse_transform([pred_idx.item()])[0]
                confidence = conf.item() * 100

            # --- 5. Display Results ---
            st.divider()
            if "Benign" in label:
                st.success(f"### Prediction: {label}")
            else:
                st.error(f"### ALERT: {label} Detected!")
            
            st.metric("Confidence Score", f"{confidence:.2f}%")
        else:
            st.error(f"Feature mismatch! Expected 46, got {len(input_data)}")
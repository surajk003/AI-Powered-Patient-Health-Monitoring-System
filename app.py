import tkinter as tk
from tkinter import messagebox, scrolledtext
import pandas as pd
import joblib
import google.generativeai as genai

# ===================================
# CONFIGURE GEMINI API
# ===================================
API_KEY = ""  # <-- Replace with your Gemini API key
genai.configure(api_key=API_KEY)

# ===================================
# LOAD TRAINED MODEL
# ===================================
try:
    model_data = joblib.load("patient_monitoring_rf_model.pkl")
    model = model_data["model_pipeline"]
    label_encoder = model_data["label_encoder"]
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model: {e}")
    raise SystemExit

# ===================================
# MAIN APP WINDOW
# ===================================
root = tk.Tk()
root.title("AI-Powered Patient Health Monitoring System")
root.geometry("720x780")
root.config(bg="#f4f6f7")

# ===================================
# HEADER
# ===================================
header = tk.Label(
    root,
    text="🩺 Patient Health Monitoring System",
    font=("Helvetica", 22, "bold"),
    bg="#2e7d32",
    fg="white",
    pady=12
)
header.pack(fill="x")

# ===================================
# INPUT FRAME
# ===================================
form_frame = tk.Frame(root, bg="#f4f6f7")
form_frame.pack(pady=25)

fields = [
    "Patient Number",
    "Heart Rate (bpm)",
    "SpO2 Level (%)",
    "Systolic BP (mmHg)",
    "Diastolic BP (mmHg)",
    "Body Temperature (°C)",
    "Fall Detection (Yes/No)"
]

entries = {}
for field in fields:
    sub_frame = tk.Frame(form_frame, bg="#f4f6f7")
    sub_frame.pack(fill="x", pady=6)
    lbl = tk.Label(sub_frame, text=field + ":", font=("Arial", 12, "bold"), bg="#f4f6f7", width=25, anchor="w")
    lbl.pack(side="left", padx=10)
    ent = tk.Entry(sub_frame, width=25, font=("Arial", 12))
    ent.pack(side="left", padx=5)
    entries[field] = ent

# ===================================
# RESULT FRAME
# ===================================
result_frame = tk.Frame(root, bg="#e8f5e9", padx=15, pady=15, highlightbackground="#81c784", highlightthickness=2)
result_frame.pack(pady=15, fill="x", padx=30)

result_label = tk.Label(
    result_frame,
    text="Prediction result will appear here.",
    font=("Arial", 14, "bold"),
    bg="#e8f5e9",
    fg="#1b5e20",
    wraplength=600,
    justify="center"
)
result_label.pack(pady=5)

# ===================================
# GEMINI OUTPUT AREA
# ===================================
ai_label = tk.Label(
    root,
    text="💡 Gemini AI Health Insights:",
    font=("Arial", 14, "bold"),
    bg="#f4f6f7",
    fg="#1b5e20"
)
ai_label.pack(pady=(10, 5))

ai_text = scrolledtext.ScrolledText(
    root,
    height=10,
    width=80,
    font=("Arial", 11),
    wrap="word",
    bg="#f1f8e9",
    relief="solid",
    bd=1
)
ai_text.pack(pady=5, padx=20)

# ===================================
# PREDICTION FUNCTION
# ===================================
def predict_disease():
    try:
        patient_number = int(entries["Patient Number"].get() or 1)
        hr = float(entries["Heart Rate (bpm)"].get())
        spo2 = float(entries["SpO2 Level (%)"].get())
        sys_bp = float(entries["Systolic BP (mmHg)"].get())
        dia_bp = float(entries["Diastolic BP (mmHg)"].get())
        temp = float(entries["Body Temperature (°C)"].get())
        fall = entries["Fall Detection (Yes/No)"].get().strip().lower()

        if fall not in ["yes", "no"]:
            messagebox.showwarning("Input Error", "Fall Detection must be 'Yes' or 'No'")
            return

        # Create input DataFrame
        input_df = pd.DataFrame([{
            "Patient Number": patient_number,
            "Heart Rate (bpm)": hr,
            "SpO2 Level (%)": spo2,
            "Systolic Blood Pressure (mmHg)": sys_bp,
            "Diastolic Blood Pressure (mmHg)": dia_bp,
            "Body Temperature (°C)": temp,
            "Fall Detection": fall,
            "Data Accuracy (%)": 100,
            "Heart Rate Alert": "NORMAL" if 60 <= hr <= 100 else "ABNORMAL",
            "SpO2 Level Alert": "NORMAL" if spo2 >= 94 else "ABNORMAL",
            "Blood Pressure Alert": "NORMAL" if (90 <= sys_bp <= 140) and (60 <= dia_bp <= 90) else "ABNORMAL",
            "Temperature Alert": "NORMAL" if 36.0 <= temp <= 37.5 else "ABNORMAL",
        }])

        # Predict and decode label
        y_pred = model.predict(input_df)
        prediction_label = label_encoder.inverse_transform(y_pred)[0]

        # Update label color based on healthiness
        color = "#1b5e20" if str(prediction_label).lower() in ["normal", "healthy"] else "red"
        result_label.config(text=f"🧾 Predicted Disease: {prediction_label}", fg=color)

        # ===============================
        # GEMINI AI HEALTH INSIGHT
        # ===============================
        ai_text.delete(1.0, tk.END)
        prompt = f"""
        You are a healthcare assistant AI.
        Patient vitals:
        - Heart Rate: {hr} bpm
        - SpO2: {spo2}%
        - Systolic BP: {sys_bp} mmHg
        - Diastolic BP: {dia_bp} mmHg
        - Temperature: {temp}°C
        - Fall Detection: {fall}
        The ML model predicted: {prediction_label}.
        Provide a brief, safe, medically general explanation and advice. Avoid diagnoses or prescriptions.
        """

        try:
            response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
            advice = response.text.strip()
        except Exception as e:
            advice = f"(AI advice unavailable)\nError: {e}"

        ai_text.insert(tk.END, advice)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for vitals.")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{e}")

# ===================================
# PREDICT BUTTON
# ===================================
predict_btn = tk.Button(
    root,
    text="🔍 Predict Disease & Get AI Advice",
    font=("Arial", 14, "bold"),
    bg="#388e3c",
    fg="white",
    relief="raised",
    bd=3,
    padx=15,
    pady=8,
    cursor="hand2",
    activebackground="#2e7d32",
    command=predict_disease
)
predict_btn.pack(pady=20)

# ===================================
# FOOTER
# ===================================
footer = tk.Label(
    root,
    text="© 2025 HealthAI - Powered by Gemini & Random Forest",
    font=("Arial", 10),
    bg="#f4f6f7",
    fg="#616161"
)
footer.pack(side="bottom", pady=10)

# ===================================
# RUN APP
# ===================================
root.mainloop()

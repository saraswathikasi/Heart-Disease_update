import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import numpy as np
# ----------------------------
# Load saved models and scaler
# ----------------------------
with open("stack_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["🏠 Home", "🧑‍⚕️ Prediction", "📈 Model Insights", "ℹ️ Feature Info"],
        icons=["house", "activity", "bar-chart", "info-circle"],
        menu_icon="heart",
        default_index=0,
    )

# ----------------------------
# Home Page
# ----------------------------
if selected == "🏠 Home":
    st.title("❤️ Heart Disease Prediction App")
    st.image("heart.png", use_container_width=True)
    st.markdown(
        """
        This app predicts the **risk of heart disease** using a stacked machine learning model.  
        Navigate to **Prediction** to test with inputs,  
        explore **Model Insights** for explanations, and  
        learn about each **Feature Info**.
        """
    )

# ----------------------------
# Prediction Page
# ----------------------------
elif selected == "🧑‍⚕️ Prediction":
    st.title("🧑‍⚕️ Heart Disease Risk Prediction")

    with st.form("prediction_form"):
        # --- all inputs ---
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
        restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 250, 150)
        exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
        oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels Colored (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
              exang, oldpeak, slope, ca, thal]],
            columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                     "restecg", "thalach", "exang", "oldpeak", "slope",
                     "ca", "thal"]
        )

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        st.subheader("✅ Prediction Result:")
        st.write("High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease")
        st.progress(int(probability * 100))

        # Save inputs for Model Insights
        st.session_state["last_input"] = input_data

# -------------------------
# Model Insights Page
# -------------------------
# Model Insights Page
# ----------------------------
elif selected == "📈 Model Insights":
    st.title("🔍 Model Insights and Explainability")

    # Load models (removed Naive Bayes)
    models = {}
    model_files = {
        "Logistic Regression": "lr_model.pkl",
        "SVM": "svm_model.pkl",
        "Stacked Model": "stack_model.pkl",
        "Voting Classifier": "voting_model.pkl",
    }

    for name, file in model_files.items():
        try:
            with open(file, "rb") as f:
                models[name] = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Could not load {name}: {e}")

    X_test = pd.read_csv("cleveland_heart_disease_cleaned.csv")
    X = X_test.drop("target", axis=1)
    y = X_test["target"]

    st.markdown("### Global Model Explanation")
    st.write("These plots show how important each feature is for each model.")

    from sklearn.inspection import permutation_importance

    # --- Display plots in 2 per row ---
    model_items = list(models.items())
    for row_start in range(0, len(model_items), 2):
        cols = st.columns(2)
        for col, (name, model) in zip(cols, model_items[row_start:row_start+2]):
            with col:
                st.subheader(name)
                try:
                    if name == "Logistic Regression":
                        explainer = shap.Explainer(model, X)
                        shap_values = explainer(X[:50])
                        fig, ax = plt.subplots(figsize=(2, 1.5))
                        shap.summary_plot(shap_values, X[:50], plot_type="bar", show=False)
                        st.pyplot(fig)
                        # Top 3 important features (SHAP)
                        top_features = pd.DataFrame({
                            "Feature": X.columns,
                            "SHAP_Importance": np.abs(shap_values.values).mean(0)
                        }).sort_values(by="SHAP_Importance", ascending=False).head(3)

                        st.markdown("**📝 Explanation:**")
                        for f in top_features["Feature"]:
                            st.write(f"- {f} strongly influences the model’s predictions.")

                        plt.close(fig)

                    else:
                        # Fallback → Permutation Importance
                        result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
                        importance_df = pd.DataFrame({
                            "Feature": X.columns,
                            "Importance": result.importances_mean
                        }).sort_values(by="Importance", ascending=False).head(5)

                        fig, ax = plt.subplots(figsize=(2, 1.5))
                        importance_df.plot(
                            kind="barh", x="Feature", y="Importance",
                            ax=ax, legend=False,
                            color="#4b8bbe"
                        )
                        ax.set_title("Top Features", fontsize=7)
                        ax.tick_params(axis="both", labelsize=6)
                        fig.tight_layout()
                        st.pyplot(fig)
                        # Top 3 important features (Permutation Importance)
                        top_features = importance_df.head(3)

                        st.markdown("**📝 Explanation:**")
                        for f in top_features["Feature"]:
                            st.write(f"- {f} is a key driver of this model’s predictions.")

                        plt.close(fig)

                except Exception as e:
                    st.warning(f"⚠️ Could not explain {name}. Reason: {e}")

    # ----------------------------
    # Patient-Specific Explanation
    # ----------------------------
    st.markdown("### Patient-Specific Explanation")
    patient_id = st.number_input("Enter Patient ID:", min_value=1, max_value=len(X), step=1)

    if st.button("Explain Patient"):
        X_patient = X.iloc[[patient_id - 1]]

        model_items = list(models.items())
        for row_start in range(0, len(model_items), 2):
            cols = st.columns(2)
            for col, (name, model) in zip(cols, model_items[row_start:row_start + 2]):
                with col:
                    st.markdown(f"**{name}**")
                    try:
                        # Use SHAP explainer for all models
                        if name == "Logistic Regression":
                            explainer = shap.Explainer(model, X)
                        else:
                            # KernelExplainer as a fallback for unsupported models
                            explainer = shap.KernelExplainer(model.predict_proba, X)

                        shap_values = explainer(X_patient)
                        # Take the contribution for the positive class
                        if shap_values.values.ndim == 3:
                            shap_contribs = shap_values.values[0, :, 1]
                        else:
                            shap_contribs = shap_values.values[0]

                        feature_contributions = pd.DataFrame({
                            "Feature": X.columns,
                            "Contribution": shap_contribs
                        }).sort_values(by="Contribution", key=abs, ascending=False).head(5)

                        # Plot bar chart
                        fig, ax = plt.subplots(figsize=(2, 1.5))
                        feature_contributions.plot(
                            kind="barh", x="Feature", y="Contribution",
                            legend=False, ax=ax,
                            color=["#ff4b4b" if v > 0 else "#4caf50" for v in feature_contributions["Contribution"]]
                        )
                        ax.set_title("Top Features", fontsize=7)
                        ax.tick_params(axis="both", labelsize=6)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        # Top 3 contributing features description
                        top_features = feature_contributions.head(3)
                        st.markdown("**📝 Explanation for this patient:**")
                        for _, row in top_features.iterrows():
                            direction = "increases risk" if row["Contribution"] > 0 else "reduces risk"
                            st.write(f"- {row['Feature']} {direction} the prediction.")

                    except Exception as e:
                        st.warning(f"⚠️ Could not explain {name}. Reason: {e}")
# ----------------------------
# Feature Info Page
# ----------------------------
elif selected == "ℹ️ Feature Info":
    st.title("ℹ️ Feature Information")

    feature_info = {
        "age": "Age of the patient. Older patients are at higher risk of heart disease.",
        "sex": "Biological sex (1 = Male, 0 = Female). Males generally have higher risk.",
        "cp": "Chest pain type (0-3). Higher values indicate more severe chest pain symptoms.",
        "trestbps": "Resting blood pressure. Higher pressure increases heart risk.",
        "chol": "Serum cholesterol level in mg/dl. High cholesterol is a major risk factor.",
        "fbs": "Fasting blood sugar > 120 mg/dl (1 = True, 0 = False). High sugar indicates diabetes risk.",
        "restecg": "Resting electrocardiographic results (0-2). Detects heart abnormalities.",
        "thalach": "Maximum heart rate achieved. Lower rates may indicate weaker heart health.",
        "exang": "Exercise induced angina (1 = Yes, 0 = No). Angina during exercise is a risk indicator.",
        "oldpeak": "ST depression induced by exercise. Higher values show abnormal heart response.",
        "slope": "Slope of the peak exercise ST segment (0-2). Certain slopes indicate higher risk.",
        "ca": "Number of major vessels colored by fluoroscopy (0-3). Higher numbers show blockages.",
        "thal": "Thalassemia test result (0 = Normal, 1 = Fixed defect, 2 = Reversible defect)."
    }

    feature_df = pd.DataFrame(list(feature_info.items()), columns=["Feature", "Description"])
    st.table(feature_df)

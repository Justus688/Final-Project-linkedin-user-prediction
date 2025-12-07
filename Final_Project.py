# Import necessary packages
import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from pathlib import Path

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score
)


### 1: Load and clean the data
BASE_DIR = Path(__file__).resolve().parent
s = pd.read_csv(BASE_DIR / "social_media_usage.csv")

# check s dimensions
print(s.info())

# define a function to clean the data
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

# Create a toy dataframe with three rows and two columns and test your function
toy_df = pd.DataFrame({
    'col1': [1, 2, 1],
    'col2': [0, 1, 2]
})
cleaned_toy_df = toy_df.apply(clean_sm)
print(cleaned_toy_df)

# create dataframe ss with selected columns and clean the data
ss = s[["web1h", "income", "educ2", "age", "par", "marital", "gender"]]
ss = ss.rename(columns={"web1h": "sm_li", "income": "income", "educ2": "educ", "age": "age", "par": "parent", "marital": "married", "gender": "female"})
ss["sm_li"] = ss["sm_li"].apply(clean_sm)
ss["income"] = ss["income"].where(ss["income"] <= 9, np.nan)
ss["educ"] = ss["educ"].where(ss["educ"] <= 8, np.nan)
ss["age"] = pd.to_numeric(ss["age"].where(ss["age"] <= 98, np.nan), errors="coerce")
ss["parent"] = ss["parent"].apply(clean_sm)
ss["married"] = ss["married"].apply(clean_sm)
ss["female"] = ss["female"].apply(clean_sm)
ss = ss.dropna()
ss["income"] = ss["income"].astype(int)
ss["educ"] = ss["educ"].astype(int)
print(ss.info())
ss.head()


### 2: load ss into streamlit
#st.dataframe(ss)

### 3: exploratory data analysis
# summary statistics
#st.write(ss.describe())
# plot pairplot in seaborn
#st.pyplot(sns.pairplot(ss, hue="sm_li"))


### 4-5: Split the data into training and testing sets stratifed by sm_li ***(go back and explain this step more in depth)***
feature_cols = ["income", "educ", "age", "parent", "married", "female"]
X = ss[feature_cols]
y = ss["sm_li"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=385, stratify=y)

# initialize the logistic regression model and fit and set class weight to balanced
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000)
log_reg.fit(X_train, y_train)



### 6: Model evaluation

# make predictions on the test set
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# generate model accuracy
accuracy = log_reg.score(X_test, y_test)

# confusion matrix & put into dataframe for better visualization
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

# precision and recall and f1 score using confusion matrix outputs
precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
f1_score = 2 * (precision * recall) / (precision + recall)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# plot ROC curve using matplotlib
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Model")
plt.legend(loc="upper left")

# Plot Average Marginal Effects for each feature
X2 = sm.add_constant(X)
logit_model = sm.Logit(y, X2)
result = logit_model.fit()
marginal_effects = result.get_margeff()
dy_dx = (marginal_effects.margeff)*100  # convert to percentage
me_df = pd.DataFrame({
    "Feature": feature_cols,
    "Average Marginal Effect": [f"{effect:.2f}%" for effect in dy_dx]
})


### 7 Build interactive prediction tool

## *** Streamlit Title and description ***
BASE_DIR = Path(__file__).resolve().parent
col1, col2 = st.columns([3, 1])
with col1:
    st.title("LinkedIn User Prediction App")
with col2:
    st.image(BASE_DIR / "linkedin_logo.png", width=120)


# mapping user inputs to model features
income_options = {
    "less than $10,000": 1,
    "$10,000 to under $20,000": 2,
    "$20,000 to under $30,000": 3,
    "$30,000 to under $40,000": 4,
    "$40,000 to under $50,000": 5,
    "$50,000 to under $75,000": 6,
    "$75,000 to under $100,000": 7,
    "$100,000 to under $150,000": 8,
    "$150,000 or more": 9}
educ_options = {
    "Less than high school": 1,
    "High school incomplete": 2,
    "High school graduate": 3,
    "Some college, no degree": 4,
    "Associate degree": 5,
    "Bachelor's degree": 6,
    "Master's degree": 7,
    "Doctorate or professional degree": 8}
parent_options = {
    "No": 0,
    "Yes": 1}
married_options = {
    "No": 0,
    "Yes": 1}
gender_options = {
    "Male": 0,
    "Female": 1}

## *** Streamlit Sidebar for user inputs ***
with st.sidebar:
    st.header("Input Features")
    income_label = st.selectbox("Income (Household)",['Select Income'] + list(income_options.keys()))
    educ_label = st.selectbox("Education Level",['Select Education'] + list(educ_options.keys()))
    age_str = st.text_input("Age (18-97)", value="")
    parent_label = st.selectbox("Parent", ['Select Yes or No'] + list(parent_options.keys()))
    married_label = st.selectbox("Married",['Select Yes or No'] + list(married_options.keys()))
    gender_label = st.selectbox("Gender",['Select Gender'] + list(gender_options.keys()))

age_valid = age_str.isdigit() and (18 <= int(age_str) <= 97)
inputs_complete = (
    income_label not in ['Select Income'] and
    educ_label not in ['Select Education'] and
    parent_label not in ['Select Yes or No'] and
    married_label not in ['Select Yes or No'] and
    gender_label not in ['Select Gender'] and
    age_valid
)

## *** Streamlit average marginal effects in sidebar ***
st.sidebar.caption("Expand the section below to see the Average Marginal Effects of each feature on the likelihood of being a LinkedIn user.")
with st.sidebar.expander("Average Marginal Effects of Features"):
    st.write(me_df)

# Warn user if inputs are incomplete
if not inputs_complete:
    st.warning("Please complete all input features with valid selections in the sidebar to get a prediction.", icon="⚠️")
    st.stop()

# Prepare input data for prediction
if inputs_complete:
    income = income_options[income_label]
    educ = educ_options[educ_label]
    age = int(age_str)
    parent = parent_options[parent_label]
    married = married_options[married_label]
    gender = gender_options[gender_label]

    input_data = pd.DataFrame({
    "income": [income],
    "educ": [educ],
    "age": [age],
    "parent": [parent],
    "married": [married],
    "female": [gender]
})
else:
    input_data = None


# Display prediction button when inputs are complete
run_prediction = st.button("Run Prediction")

# Warn user if inputs are incomplete or invalid when button is pressed
if run_prediction and not inputs_complete:
    st.error("Please complete all input features with valid selections in the sidebar to get a prediction.", icon="❌")

# Generate and display prediction when button is pressed and inputs are complete
if run_prediction and inputs_complete:

    # make prediction
    prediction = log_reg.predict(input_data)
    prediction_proba = log_reg.predict_proba(input_data)[:, 1]

    proba = float(prediction_proba[0])
    not_proba = 1 - proba

    prediction_label = "LinkedIn user" if prediction[0] == 1 else "Not a LinkedIn user"
    prediction_proba = proba if prediction[0] == 1 else not_proba

    # Status box
    if prediction[0] == 1:
        st.success(f'The Model predicts this individual **IS a LinkedIn user**.', icon="✅")
    else:
        st.error(f'The Model predicts this individual **IS NOT a LinkedIn user**.', icon="❌")

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Prediction", value=prediction_label)
    with col2:
        st.metric(label="Probability", value=f"{prediction_proba:.1%}")
    # Probability bar chart
    proba_df = pd.DataFrame({
        "Status": ["LinkedIn User", "Not LinkedIn User"],
        "Probability": [proba, not_proba]
    })
    bar_chart = alt.Chart(proba_df).mark_bar().encode(
        x=alt.X("Status", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
        tooltip=["Status", alt.Tooltip("Probability", format=".1%")]
    ).properties(
        title="Prediction Probability",
        width=400,
        height=300)
    st.altair_chart(bar_chart)

    # explanation
    st.caption(
        f"The model estimates a {proba:.1%} chance this individual **IS** a LinkedIn user"
        f" and a {not_proba:.1%} chance this individual **IS NOT** a LinkedIn user.")
    st.caption(
        f" See below for model evaluation metrics and a log of all predictions made during this session. (note: click in top right of charts to download as csv file)"
    )

### Display model evaluation metrics with buttons to show/hide
    with st.expander("Show Model Evaluation Metrics"):
        st.markdown("### Model Evaluation Metrics")
        st.write("Confusion Matrix:")
        st.dataframe(cm_df)
        st.write("Evaluation Scores:")
        st.dataframe({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            "Score": [f"{accuracy:.3f}", f"{precision:.3f}", f"{recall:.3f}", f"{f1_score:.3f}", f"{roc_auc:.3f}"]
        })
        st.pyplot(plt)


    # app stores the users input in a dataframe and adds to it as more predictions are made
    with st.expander("View and Download Predictions Log"):
        if "predictions_log" not in st.session_state:
            st.session_state["predictions_log"] = pd.DataFrame(columns=[
                "income", "educ", "age", "parent", "married", "female", "probability", "prediction"
            ])
        new_entry = {
            "income": income,
            "educ": educ,
            "age": age,
            "parent": parent,
            "married": married,
            "female": gender,
            "probability": float(prediction_proba),
            "prediction": int(prediction[0]),
        }
        st.session_state["predictions_log"] = pd.concat([st.session_state["predictions_log"], pd.DataFrame([new_entry])], ignore_index=True)
        st.dataframe(st.session_state["predictions_log"])
### App end

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- MODEL TRAINING ----------------

data = pd.DataFrame({
    "Study Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [15, 25, 35, 45, 55, 65, 72, 80, 90, 95]
})

X = data[["Study Hours"]]
y = data["Marks"]

model = LinearRegression()
model.fit(X, y)

# ---------------- APP TITLE ----------------

st.title("🎓 Student Marks Prediction")
st.write(
    "Enter student details, record subject-wise marks, "
    "and predict exam marks based on daily study hours."
)

st.info(
    "Demo project: predictions use sample data and study hours. "
    "Subject-wise marks are recorded separately."
)

# ---------------- SESSION STORAGE ----------------

if "students" not in st.session_state:
    st.session_state.students = []

# ---------------- STUDENT INPUT FORM ----------------

st.header("📝 Enter Student Details")

with st.form("student_form", clear_on_submit=True):

    name = st.text_input("Student Name")

    study_hours = st.number_input(
        "Daily Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    st.subheader("Subject-wise Marks")

    col1, col2, col3 = st.columns(3)

    with col1:
        maths = st.number_input(
            "Maths Marks",
            min_value=0,
            max_value=100,
            value=0
        )

    with col2:
        science = st.number_input(
            "Science Marks",
            min_value=0,
            max_value=100,
            value=0
        )

    with col3:
        english = st.number_input(
            "English Marks",
            min_value=0,
            max_value=100,
            value=0
        )

    submitted = st.form_submit_button(
        "➕ Add Student and Predict"
    )

    if submitted:
        if not name.strip():
            st.error("Please enter the student name.")
        else:
            prediction = model.predict(
                pd.DataFrame({"Study Hours": [study_hours]})
            )[0]

            prediction = max(0, min(100, prediction))

            average = (maths + science + english) / 3

            student = {
                "Student Name": name.strip(),
                "Study Hours": study_hours,
                "Maths": maths,
                "Science": science,
                "English": english,
                "Subject Average": round(average, 2),
                "Predicted Exam Marks": round(prediction, 2)
            }

            st.session_state.students.append(student)

            st.success(
                f"Added {name.strip()} successfully!"
            )

# ---------------- DISPLAY RESULTS ----------------

st.header("📊 Student Results")

if st.session_state.students:

    results = pd.DataFrame(st.session_state.students)

    st.dataframe(results, use_container_width=True)

    st.subheader("🏆 Predicted Marks")

    chart_data = results.set_index("Student Name")[
        "Predicted Exam Marks"
    ]

    st.bar_chart(chart_data)

    st.subheader("📚 Subject-wise Comparison")

    subject_columns = ["Maths", "Science", "English"]

    subject_chart = results.set_index("Student Name")[
        subject_columns
    ]

    st.bar_chart(subject_chart)

    st.subheader("⬇️ Download Results")

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Student Results as CSV",
        data=csv_data,
        file_name="student_marks_results.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear All Students"):
        st.session_state.students = []
        st.rerun()

else:
    st.write(
        "No students added yet. Enter student details above "
        "and click Add Student and Predict."
    )

# ---------------- MODEL EVALUATION ----------------

st.divider()

st.header("🤖 Model Evaluation")

predictions = model.predict(X)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Mean Absolute Error",
        f"{mean_absolute_error(y, predictions):.2f}"
    )

with col2:
    st.metric(
        "R² Score",
        f"{r2_score(y, predictions):.3f}"
    )

st.caption(
    "Evaluation uses the same small sample dataset used to train "
    "the model. It does not measure performance on unseen students."
)

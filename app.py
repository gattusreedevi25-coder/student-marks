
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

# Train the Machine Learning model
data = pd.DataFrame({
    "Study Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [15, 25, 35, 45, 55, 65, 72, 80, 90, 95]
})

X = data[["Study Hours"]]
y = data["Marks"]

model = LinearRegression()
model.fit(X, y)

# App heading
st.title("🎓 Student Marks Prediction Dashboard")
st.caption("Developed by Sreedevi | Python • Streamlit • Machine Learning")

if "students" not in st.session_state:
    st.session_state.students = []

# Input form and live prediction preview
left, right = st.columns([1, 1])

with left:
    st.subheader("📝 Student Details")

    name = st.text_input("Student Name")

    hours = st.number_input(
        "Daily Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        maths = st.number_input(
            "Maths", 0, 100, 80
        )

    with c2:
        science = st.number_input(
            "Science", 0, 100, 85
        )

    with c3:
        english = st.number_input(
            "English", 0, 100, 90
        )

    average = (maths + science + english) / 3

    if st.button("➕ Add Student and Predict", use_container_width=True):
        if not name.strip():
            st.error("Please enter the student name.")
        else:
            prediction = model.predict(
                pd.DataFrame({"Study Hours": [hours]})
            )[0]

            prediction = max(0, min(100, prediction))

            student = {
                "Student Name": name.strip(),
                "Study Hours": hours,
                "Maths": maths,
                "Science": science,
                "English": english,
                "Subject Average": round(average, 2),
                "Predicted Exam Marks": round(prediction, 2)
            }

            # Update existing student rather than duplicate the name
            st.session_state.students = [
                s for s in st.session_state.students
                if s["Student Name"].casefold() != name.strip().casefold()
            ]
            st.session_state.students.append(student)
            st.success(f"Results added for {name.strip()}!")

with right:
    st.subheader("🔮 Prediction Preview")

    prediction = model.predict(
        pd.DataFrame({"Study Hours": [hours]})
    )[0]
    prediction = max(0, min(100, prediction))

    st.metric("Predicted Exam Marks", f"{prediction:.2f} / 100")
    st.metric("Subject Average", f"{average:.2f} / 100")
    st.metric("Daily Study Hours", f"{hours:g} hours")

    st.info(
        "The prediction is estimated from study hours using sample data. "
        "Subject marks are recorded separately."
    )

st.divider()

# Results table
st.subheader("📊 Student Results")

if st.session_state.students:
    results = pd.DataFrame(st.session_state.students)
    st.dataframe(results, use_container_width=True, hide_index=True)

    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results as CSV",
        data=csv_data,
        file_name="student_marks_results.csv",
        mime="text/csv"
    )

    # Display both charts side by side
    chart1, chart2 = st.columns(2)

    with chart1:
        st.subheader("🏆 Predicted Marks")
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.bar(
            results["Student Name"],
            results["Predicted Exam Marks"]
        )
        ax1.set_ylabel("Predicted Marks")
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis="x", rotation=45)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with chart2:
        st.subheader("📚 Subject-wise Marks")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        results.set_index("Student Name")[
            ["Maths", "Science", "English"]
        ].plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Marks")
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis="x", rotation=45)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    if st.button("🗑️ Clear All Students"):
        st.session_state.students = []
        st.rerun()

else:
    st.info("Add a student above to display the results table and charts.")

# Model evaluation
st.divider()
st.subheader("🤖 Model Evaluation")

predictions = model.predict(X)
m1, m2 = st.columns(2)

m1.metric(
    "Mean Absolute Error",
    f"{mean_absolute_error(y, predictions):.2f}"
)
m2.metric(
    "R² Score",
    f"{r2_score(y, predictions):.3f}"
)

st.caption(
    "Evaluation uses the small sample training dataset; "
    "it does not measure accuracy on unseen students."
)

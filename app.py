
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Marks Prediction", page_icon="🎓")

st.title("🎓 Student Marks Prediction")
st.write("Predict exam marks based on daily study hours.")

data = pd.DataFrame({
    "Study Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [15, 25, 35, 45, 55, 65, 72, 80, 90, 95]
})

X = data[["Study Hours"]]
y = data["Marks"]

model = LinearRegression()
model.fit(X, y)

hours = st.slider(
    "Daily study hours",
    min_value=0.0,
    max_value=12.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Marks"):
    prediction = model.predict(
        pd.DataFrame({"Study Hours": [hours]})
    )[0]

    prediction = max(0, min(100, prediction))
    st.success(f"Predicted Marks: {prediction:.2f} / 100")

st.subheader("Study Hours vs Marks")

fig, ax = plt.subplots()
ax.scatter(X["Study Hours"], y, label="Actual sample data")
ax.plot(X["Study Hours"], model.predict(X), label="Regression line")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Marks")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)

st.subheader("Model Evaluation")
st.write(f"Mean Absolute Error: {mean_absolute_error(y, model.predict(X)):.2f}")
st.write(f"R² Score: {r2_score(y, model.predict(X)):.3f}")

st.caption("Demo project using sample data. Predictions are estimates.")

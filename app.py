import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
import joblib
import numpy as np
from io import StringIO

st.title("Прогноз оценки студента")

# Данные
data_csv = """Hours,Scores
2.5,21
5.1,47
3.2,27
8.5,75
3.5,30
1.5,20
9.2,88
5.5,60
8.3,81
2.7,25
7.7,85
5.9,62
4.5,41
3.3,42"""
df = pd.read_csv(StringIO(data_csv))

X = df[["Hours"]]
y = df["Scores"]

# Файл модели
model_file = "student_model.pkl"

loaded = joblib.load(model_file)
if isinstance(loaded, tuple):
    model, poly = loaded
else:
    # старый файл без poly
    model = loaded
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model.fit(X_poly, y)
    joblib.dump((model, poly), model_file)

# Слайдер для часов обучения
st.sidebar.header("Введите данные:")
hours = st.sidebar.slider(
    "Часы обучения",
    float(df["Hours"].min()),
    float(df["Hours"].max()),
    float(df["Hours"].mean())
)
input_data = pd.DataFrame({"Hours": [hours]})
input_poly = poly.transform(input_data)
prediction = model.predict(input_poly)[0]

# Вывод прогноза
st.subheader("Прогнозируемая оценка:")
st.write(f"{prediction:.2f}")

# Точность модели
X_poly = poly.transform(X)
r2 = model.score(X_poly, y)
st.subheader("Точность модели (R²):")
st.write(f"{r2:.2f}")

# Вывод всей базы данных
st.subheader("База данных:")
st.dataframe(df)

# График
x_range = np.linspace(df["Hours"].min(), df["Hours"].max(), 100)
x_range_poly = poly.transform(x_range.reshape(-1, 1))
y_pred_line = model.predict(x_range_poly)

fig, ax = plt.subplots()
ax.scatter(df["Hours"], df["Scores"], label="Фактические данные", color="royalblue")
ax.plot(x_range, y_pred_line, label="Линия регрессии", color="red", linewidth=2)
ax.scatter(hours, prediction, color="lime", s=100, label="Прогноз")

ax.set_xlabel("Часы обучения")
ax.set_ylabel("Оценка")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

X = df[["Hours"]]   
y = df["Scores"]   

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_file = "student_model.pkl"

try:
    model = joblib.load(model_file)
    print("Модель загружена из файла")
except:
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    print("Модель обучена и сохранена")

y_pred = model.predict(X_test)

print(f"Точность (R²): {model.score(X_test, y_test):.4f}")

plt.figure(figsize=(8, 6))

plt.scatter(X_test, y_test, alpha=0.6, label="Реальные данные")
plt.plot(X_test, y_pred, linestyle="--", label="Линия регрессии")

plt.xlabel("Часы обучения")
plt.ylabel("Оценка")
plt.title("Зависимость оценки от времени обучения")
plt.legend()
plt.grid(True)
plt.show()

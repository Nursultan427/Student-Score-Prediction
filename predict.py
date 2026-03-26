import pandas as pd
import joblib
from sklearn.metrics import r2_score

data = pd.read_csv("student_scores.csv") 
print("Первые строки данных:")
print(data.head())

X = data[['Hours']]
y = data['Scores']

model = joblib.load("student_model.pkl")
print("\nМодель загружена из файла")

y_pred = model.predict(X)
print(f"\nТочность модели (R²): {r2_score(y, y_pred):.3f}")

def get_float(prompt, min_val=0, max_val=24):
    while True:
        try:
            value = float(input(prompt))
            if value < min_val or value > max_val:
                print(f"Введите число от {min_val} до {max_val}")
                continue
            return value
        except ValueError:
            print("Ошибка! Введите число.")

hours = get_float("\nВведите количество часов обучения: ", 0, 24)

predicted_score = model.predict([[hours]])[0]

print(f"\nОжидаемый результат: {predicted_score:.2f} баллов при {hours} часах обучения.")

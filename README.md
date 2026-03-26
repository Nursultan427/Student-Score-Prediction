# Установка библиотек
pip install streamlit
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install joblib
pip install numpy

# Файлы проекта
app.py
student_scores.csv

# Пример содержимого student_scores.csv
Hours,Scores
2.5,21
5.1,47
3.2,27
8.5,75
3.5,30
1.5,20
9.2,88
5.5,60

# Запуск приложения
streamlit run app.py

# Что создаётся автоматически
student_model.pkl  # сохранённая модель после обучения

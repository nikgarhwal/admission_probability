# admission_predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load data
admsn = pd.read_csv("https://github.com/ybifoundation/Dataset/raw/main/Admission%20Chance.csv")

# Define features and target
y = admsn['Chance of Admit ']
x = admsn[['Serial No', 'GRE Score', 'TOEFL Score', 'University Rating', ' SOP',
           'LOR ', 'CGPA', 'Research']]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=2529)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
print("\n📊 Model Evaluation:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# Predict for custom input
def predict_admission():
    print("\n🎓 Predict Your Admission Probability")
    try:
        serial_no = int(input("Enter Serial No: "))
        gre = float(input("Enter GRE Score (e.g. 320): "))
        toefl = float(input("Enter TOEFL Score (e.g. 110): "))
        rating = float(input("Enter University Rating (1-5): "))
        sop = float(input("Enter SOP strength (1-5): "))
        lor = float(input("Enter LOR strength (1-5): "))
        cgpa = float(input("Enter CGPA (e.g. 8.5): "))
        research = int(input("Research experience? (1 = Yes, 0 = No): "))

        features = [[serial_no, gre, toefl, rating, sop, lor, cgpa, research]]
        prediction = model.predict(features)
        print(f"\n🧾 Estimated Chance of Admission: {prediction[0]*100:.2f}%")

    except Exception as e:
        print("⚠️ Invalid input:", e)

# Call the function
if __name__ == "__main__":
    predict_admission()

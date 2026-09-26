import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
    
st.title("Dự đoán giá ô tô dựa trên các đặc điểm của ô tô")

uploaded_file = st.file_uploader("Tải lên tệp CSV", type="csv")

if uploaded_file is None:
    st.warning("Vui lòng tải lên tệp CSV để tiếp tục.")
    st.stop()

st.subheader("Dữ liệu ô tô đã tải lên")
df = pd.read_csv(uploaded_file)
st.data_editor(df)

df_encoded = pd.get_dummies(df, columns=["Brand"], drop_first=True)

X = df_encoded.drop(columns=["Price"])
y = df_encoded["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("So sánh giá đoán của mô hình với giá thực tế")

y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
st.pyplot(plt)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.subheader("Đánh giá hiệu suất mô hình")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R2): {r2:.2f}")

st.subheader("Giá bán trung bình của các hãng xe")
df[["Brand", "Price"]].groupby("Brand").mean().sort_values(by="Price", ascending=False).plot(kind="bar", figsize=(10, 6))
plt.xlabel("Brand")
plt.ylabel("Average Price")
plt.title("Average Price by Brand")
st.pyplot(plt)

st.subheader("Phân bố giá bán theo hãng xe")
plt.figure(figsize=(10, 6))
sns.boxplot(x="Brand", y="Price", data=df)
plt.xticks(rotation=45)
plt.title("Price Distribution by Brand")
st.pyplot(plt)

st.subheader("Đánh giá độ ảnh hưởng của các đặc điểm ô tô đến giá bán")

st.subheader("Đánh giá mức độ ảnh hưởng của dung tích động cơ")
plt.figure(figsize=(8, 6))
plt.scatter(df["EngineSize"], df["Price"])
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.title("Engine Size vs Price")
st.pyplot(plt)

st.subheader("Đánh giá mức độ ảnh hưởng của mã lực")
plt.figure(figsize=(8, 6))
X_line_hp = np.linspace(df["Horsepower"].min(), df["Horsepower"].max(), 100)
y_line_hp = model.coef_[2] * X_line_hp + model.intercept_
plt.scatter(df["Horsepower"], df["Price"])
plt.plot(X_line_hp, y_line_hp, color="red", linewidth=2)
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Horsepower vs Price")
st.pyplot(plt)

st.subheader("Đánh giá mức độ ảnh hưởng của tuổi xe")
plt.figure(figsize=(8, 6))
plt.scatter(df["Car_Age"], df["Price"])
plt.xlabel("Car Age")
plt.ylabel("Price")
plt.title("Car Age vs Price")
st.pyplot(plt)

st.subheader("Đánh giá mức độ ảnh hưởng của số dặm đã đi")
plt.figure(figsize=(8, 6))
plt.scatter(df["Mileage"], df["Price"])
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Mileage vs Price")
st.pyplot(plt)

st.subheader("Đánh giá độ ảnh hưởng của các đặc điểm ô tô đến giá bán")
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_
}).sort_values(by="Importance", ascending=False)
st.dataframe(feature_importance)

with st.form("predict car price"):
    st.write("Dự đoán giá xe dựa trên các đặc điểm:")

    Brand = st.selectbox("Brand", df["Brand"].unique())
    Car_Age = st.number_input("Age")
    EngineSize = st.number_input("Engine Size")
    Horsepower = st.number_input("Horsepower")
    Mileage = st.number_input("Mileage")

    submitted = st.form_submit_button("Dự đoán giá bán")

if submitted:
    input_data = pd.DataFrame({
        "Car_Age": [Car_Age],
        "EngineSize": [EngineSize],
        "Horsepower": [Horsepower],
        "Mileage": [Mileage]
    })

    for brand in df_encoded.columns:
        if brand.startswith("Brand_"):
            input_data[brand] = 1 if brand == f"Brand_{Brand}" else 0

    predicted_price = model.predict(input_data)[0]
    st.success(f"Giá bán dự đoán: {predicted_price:.2f}")
# 🛒 BigMart Sales Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

<div align="center">
  <img src="https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=800&h=400&fit=crop" alt="Retail Analytics" width="800" height="400"/>
</div>

📊 Project Overview

This project predicts the sales of BigMart products using Machine Learning.
It analyzes product attributes, outlet characteristics, and historical sales data to forecast store performance with high accuracy.

Developed as part of a Data Science portfolio project, this system demonstrates data preprocessing, model training, feature engineering, and evaluation techniques.

🎯 Key Features

✅ Handles missing values and feature engineering automatically

🧠 Trains multiple ML models — Linear, Lasso, Decision Tree, Random Forest, and Extra Trees

📈 Compares model performance with metrics (RMSE, MAE, R²)

🎨 Generates visualizations of data distribution and feature importance

🗂️ Clean and modular structure — easy to extend or deploy

🚀 Quick Start
1️⃣ Clone the Repository
git clone https://github.com/GowsalyaKN/BigMart-Sales-Prediction.git
cd BigMart-Sales-Prediction

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Run the Project
python src/main.py

📁 Project Structure
BigMart-Sales-Prediction/
├── data/
│   └── Train.csv
├── results/
│   ├── best_model.pkl
│   ├── feature_distributions.png
│   ├── model_performance.csv
│   └── model_performance.png
├── src/
│   └── main.py
├── README.md
├── requirements.txt
└── LICENSE

🧩 Data Description

Product Features:

Item_Weight — Weight of the product

Item_Fat_Content — Fat level (Low Fat, Regular, etc.)

Item_Visibility — Percentage visibility of the item in the store

Item_Type — Type/category of product

Item_MRP — Maximum retail price

Outlet Features:

Outlet_Identifier — Unique store code

Outlet_Size — Store size (Small/Medium/Large)

Outlet_Location_Type — Tier of the store location

Outlet_Type — Type of outlet (Supermarket/Grocery)

Outlet_Establishment_Year — Year of establishment

Target Variable:

Item_Outlet_Sales — Sales value to be predicted

🧠 Model Training Summary
Model	RMSE	MAE	R² Score
Linear Regression	0.5328	0.4149	0.7300
Lasso Regression	0.5486	0.4294	0.7137
Decision Tree	0.6007	0.4525	0.6568
Random Forest	0.5438	0.4241	0.7187
Extra Trees	0.5566	0.4353	0.7053

📊 Best Model: Linear Regression
🏆 R² Score: 0.73

🎨 Visualizations

Output images will be saved automatically inside the results/ folder:

feature_distributions.png — Shows data distribution

model_performance.png — Model RMSE comparison

feature_importance.png — Key influential features

🧰 Technologies Used

Python 3.8+

Pandas, NumPy — Data handling

Matplotlib, Seaborn — Visualization

Scikit-learn — Machine learning models

Joblib — Model serialization

🧼 Data Preprocessing Steps

Impute missing values (Item_Weight, Outlet_Size)

Replace zero visibility values with mean

Encode categorical variables (LabelEncoder + OneHotEncoder)

Create new features (New_Item_Type, Outlet_Years)

Log-transform the target variable (Item_Outlet_Sales)

🧑‍💻 Author
👩‍💻 Gowsalya K N
📧 Email: gowsalyakn889@gmail.com

💼 GitHub: @GowsalyaKN

<div align="center">

⭐ Star this repository if you found it helpful!
Made with ❤️ by Gowsalya K N for Data Science learning and exploration.

</div>

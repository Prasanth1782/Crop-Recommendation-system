# 🌾 Resilient Farming with Climate-Based Crop Guidance

This project develops a **Machine Learning-powered crop recommendation system** to assist farmers in making **data-driven decisions** about crop selection based on environmental and soil conditions.  
The system leverages **soil nutrients (NPK), pH, temperature, rainfall, and humidity** to predict the most suitable crop for cultivation.  
It is built to improve **agricultural productivity, sustainability, and farmer profitability**.

---

## 🚀 Features
- **Data Preprocessing**
  - Missing value handling
  - Label encoding
  - Min-Max normalization
  - SMOTE oversampling for class imbalance
- **Machine Learning Models Implemented**
  - Naive Bayes (Best Accuracy: **99.55%**)
  - Random Forest
  - Bagging Classifier
  - Gradient Boosting
  - K-Nearest Neighbors
  - Support Vector Machine
  - Decision Tree
  - Logistic Regression
  - Extra Trees
- **Visualization**
  - Label encoding visualization
  - Correlation heatmaps
  - Class distribution (donut chart)
  - Feature-to-label relationships (boxplot, violin plot, swarmplot)
  - Accuracy comparison bar charts
  - Confusion matrix heatmaps
- **Evaluation Metrics**
  - Accuracy
  - Precision
  - Recall
  - F1-Score

---
```
## 📂 Project Structure
├── data/
│   └── Crop_recommendation.csv # Dataset
├── code/
│   └── crop_recommendation.ipynb # Main code & analysis
├── visuals/
│   ├── label_encoding.png
│   ├── class_distribution.png
│   └── model_comparisions.png
├── docs/
│   ├── Climate_Crop_Recommendation_Report.docx
│   ├── Climate_Crop_Recommendation_Presentation.pptx # Presentation
│   └── Climate_Crop_Recommendation_Research_Paper.pdf # Research paper
├── README.md # Project documentation
└── requirements.txt # Python dependencies

```



## 🧾 Dataset
The dataset contains:
| Feature | Description |
|---------|-------------|
| N | Nitrogen content in soil |
| P | Phosphorous content in soil |
| K | Potassium content in soil |
| Temperature (°C) | Atmospheric temperature |
| Humidity (%) | Air moisture |
| pH | Soil acidity/alkalinity |
| Rainfall (mm) | Rainfall amount |
| Label | Recommended crop type |

Source: [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

---

## 🔬 Methodology
1. Load dataset and preprocess (remove duplicates, handle nulls, encode labels, normalize).
2. Perform Exploratory Data Analysis (EDA) with Seaborn and Matplotlib.
3. Apply **SMOTE** to balance classes.
4. Train multiple ML models for classification.
5. Evaluate and compare models using Accuracy, Precision, Recall, and F1-Score.
6. Visualize model performance.

---

## 🏆 Results
| Model                | Accuracy |
|----------------------|----------|
| Naive Bayes          | 99.55%   |
| Random Forest        | 99.32%   |
| Bagging Classifier   | 98.64%   |
| Gradient Boosting    | 98.18%   |
| K-Nearest Neighbors  | 96.82%   |
| Support Vector Machine| 96.82%  |
| Logistic Regression  | 91.82%   |

- **Naive Bayes** emerged as the top-performing model with the highest accuracy and balanced performance metrics.

---

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**:
  - `pandas`, `numpy`, `scikit-learn`, `imblearn`
  - `matplotlib`, `seaborn`
- **Environment**: Jupyter Notebook, Google Colab

---


## 📊 Visualizations

### 🔹 Class Distribution
![Class Distribution](Visuals/class_distribution.png)

### 🔹 Label Encoding
![Label Encoding](Visuals/label_encoding.png)

### 🔹 Model Comparison
![Model Comparison](Visuals/Model_Comparision.png)


---


## 📖 Documentation
- [📄 Full Project Report (DOCX)](docs/Climate_Crop_Recommendation_Report.docx)
- [📊 Presentation Slides (PPTX)](docs/Climate_Crop_Recommendation_Presentation.pptx)
- [📕 Research Paper (PDF)](docs/Climate_Crop_Recommendation_Research_Paper.pdf)
```

🌱 Future Scope
Integrate real-time IoT sensor data for predictions.

Add features like soil type, pest detection, and market price trends.

Build a farmer-friendly mobile/web app for recommendations.

Use Explainable AI (XAI) for better interpretability.

---

👨‍💻 Contributors
Abhirama Raju Nadimpalli

Yaswanth Krishna Kumar Pothuri

Venkata Naga Sai Kiran Kothuru

Prasanth Tuta

Prince Kumar

Under the guidance of Ms. Bhargavi Maridu
Department of CSE, Vignan’s Foundation for Science, Technology & Research




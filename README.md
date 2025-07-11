# Framingham Ischemic Heart Disease Death Prediction

This project uses the **Framingham Heart Study dataset** to predict the **probability of death due to ischemic heart disease** using multiple machine learning models, including **Random Forest**, **XGBoost**, and a **Deep Learning model**.

## 🔬 Dataset
The dataset is based on the **Framingham Heart Study**, a long-term, ongoing cardiovascular cohort study on residents of Framingham, Massachusetts. It includes features like:

- Demographics (age, gender)
- Clinical measurements (blood pressure, cholesterol, BMI)
- Lifestyle (smoking, diabetes)
- Medical history (stroke, hypertension)


## 📊 Objective
To build models that output `predict_proba()` values indicating the **probability of death due to ischemic heart disease**, rather than a simple classification.

---

## 🧹 Data Preprocessing
The following preprocessing steps were applied:

- Dropped rows with missing or invalid values (e.g., `NaN`)
- Does'nt enclude categorical variables for one-hot encoding
- Normalize data using StandardScaler
- split data as 0.8 for train and 0.2 for test with train_test_split

---

## 🤖 Models Used

### 1. Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,max_depth=8, random_state=42)
rf.fit(x_train, y_train)
probabilities = rf.predict_proba(x_test)

for i, prob in enumerate(probabilities):
    print(f"Sample {i+1}: Class 0 Probability = {prob[0]:.3f}, Class 1 Probability = {prob[1]:.3f}")
```

### 2. XGBoost Classifier
```python
import xgboost as xgb
from sklearn.metrics import accuracy_score
xgb_cls = xgb.XGBClassifier()
xgb_cls.fit(x_train,y_train)
probabilities = xgb_cls.predict_proba(x_test)

for i, prob in enumerate(probabilities):
    print(f"{i+1}: Class 0 Probability = {prob[0]:.3f}, Class 1 Probability = {prob[1]:.3f}")
```

### 3. Deep Learning (Keras)
```python
import tensorflow as tf
from tensorflow import keras

dl_model = keras.Sequential()
dl_model.add(keras.layers.Input([15]))
dl_model.add(keras.layers.Dense(units=128,activation='relu'))
dl_model.add(keras.layers.BatchNormalization())
dl_model.add(keras.layers.Dropout(0.4))
dl_model.add(keras.layers.Dense(units=64,activation='relu'))
dl_model.add(keras.layers.BatchNormalization())
dl_model.add(keras.layers.Dropout(0.3))
dl_model.add(keras.layers.Dense(units=32,activation='relu'))
dl_model.add(keras.layers.BatchNormalization())
dl_model.add(keras.layers.Dropout(0.2))

dl_model.add(keras.layers.Dense(units=1, activation='sigmoid'))

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss = keras.losses.BinaryCrossentropy
dl_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
dl_model.fit(x_train,y_train,epochs=60,batch_size=64,validation_data=(x_test,y_test))

y_pred_prob = dl_model.predict(x_test).flatten()

proba_df = pd.DataFrame({
    'P(class_0)': 1 - y_pred_prob,
    'P(class_1)': y_pred_prob
})

proba_df['predicted_label'] = (y_pred_prob > 0.5).astype(int)

print("Probabilities for first 10 samples:")
print(proba_df.head(12))
```

---

## 📈 Evaluation Metrics
- ROC AUC Score
- Accuracy

---

## 📌 Requirements
```text
scikit-learn
xgboost
pandas
numpy
tensorflow
keras
```

---

## 🧠 To improve acc
- Hyperparameter tuning
- Feature selection
- Add interpretability tools like SHAP or LIME

---

## 🙌 Acknowledgements
- Framingham Heart Study
- UCI / Kaggle public health repositories

---

## 🚀 Author
**[Yousef]** — _Electrical Engineering Student & ML Researcher_

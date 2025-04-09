# ⚙️ Scikit-Learn Pipeline for Income Classification

A compact and powerful machine learning project using Scikit-Learn Pipelines to classify income levels from the UCI Adult dataset.

---

## 🚀 Features

- 🧹 Full preprocessing pipeline (numeric & categorical features)
- 🔄 Imputation, scaling, and one-hot encoding
- 🧠 Logistic Regression model (easily switchable)
- 📊 Cross-validation & test evaluation
- 📈 Accuracy visualization using Matplotlib
- 🛠️ Designed for easy model & hyperparameter customization

---

## 🗂️ Project Structure

```
📁 project_root/
├── data/
│   └── adult_income.csv
├── config/
│   └── config.json
│   └── model.pkl
├── main.py
└── README.md
```

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 📚 Dataset

- **Source**: [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Goal**: Predict whether income > $50K based on census data.

---

## 🧠 How It Works

### 1. Load & Clean Data
Missing values (`"?"`) are replaced with `NaN` for proper imputation.

### 2. Build Pipeline
```python
Pipeline([
  ('preprocessor', ColumnTransformer([...]),  # numeric & categorical pipelines
  ('classifier', LogisticRegression())        # replace with any model
])
```

### 3. Train & Evaluate
- Model is trained on training data
- Evaluated using cross-validation and test set
- Accuracy results are visualized in a bar chart

---

## 🖼️ Output Example

```
Test Accuracy: 0.8532
```

A bar chart is displayed comparing cross-validation and test accuracy.

---

## 🔁 Customization

### ✅ Switch Models Easily

Replace the classifier in `build_pipeline()`:

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
```

### ⚙️ Tune Hyperparameters

```python
LogisticRegression(max_iter=1000, C=0.5)
```

You can also integrate `GridSearchCV` for automatic tuning.

---

## 🧪 Example Output

![Pipeline Accuracy Bar Plot](https://via.placeholder.com/500x250?text=Pipeline+Accuracy+Plot)

---

## ✅ To Do

- [x] Add cross-validation
- [x] Include visualization
- [x] Add model selection via config file
- [x] Export pipeline using `joblib`

---

## 📄 License

MIT License. Use freely. ✌️

---

## 🤝 Contributing

Pull requests are welcome! Open an issue or suggest improvements anytime.

---

## 🙌 Acknowledgements

Thanks to the UCI Machine Learning Repository for the dataset, and the Scikit-Learn devs for making ML awesome and accessible.
```

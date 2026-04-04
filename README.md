# End-to-End Machine Learning Project: Student Performance Prediction

## Overview

This is an end-to-end machine learning project that predicts student performance (math scores) based on various demographic and educational factors. The project implements a complete ML pipeline from data ingestion to model training, but deployment is not done.

## Problem Statement

The project aims to understand how student performance (test scores) is affected by various factors such as:
- Gender
- Race/Ethnicity
- Parental level of education
- Lunch type (standard/free-reduced)
- Test preparation course completion
- Reading and writing scores

## Dataset

- **Source**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Size**: 1000 rows × 8 columns
- **Features**:
  - `gender`: Male/Female
  - `race_ethnicity`: Groups A, B, C, D, E
  - `parental_level_of_education`: Bachelor's degree, Some college, Master's degree, Associate's degree, High school
  - `lunch`: Standard or Free/Reduced
  - `test_preparation_course`: Completed or Not completed
  - `math_score`: Target variable (0-100)
  - `reading_score`: 0-100
  - `writing_score`: 0-100

## Tech Stack

- **Programming Language**: Python 3.x
- **Web Framework**: Flask
- **Machine Learning Libraries**:
  - scikit-learn
  - CatBoost
  - XGBoost
  - pandas
  - numpy
- **Visualization**: matplotlib, seaborn
- **Model Evaluation**: R² Score, other regression metrics

## Machine Learning Models

The project evaluates multiple regression models and selects the best performing one:
- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Linear Regression
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

## Project Structure

```
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── setup.py                       # Package setup
├── README.md                      # Project documentation
├── artifacts/                     # Model artifacts and data
│   ├── model.pkl                  # Trained model
│   ├── preprocessor.pkl           # Data preprocessor
│   ├── data.csv                   # Processed data
│   ├── train.csv                  # Training data
│   └── test.csv                   # Test data
├── notebook/                      # Jupyter notebooks
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb    # Exploratory Data Analysis
│   └── 2. MODEL TRAINING.ipynb             # Model training and evaluation
├── src/                           # Source code
│   ├── __init__.py
│   ├── exception.py               # Custom exceptions
│   ├── logger.py                  # Logging configuration
│   ├── utils.py                   # Utility functions
│   ├── components/                # ML pipeline components
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py # Feature engineering and preprocessing
│   │   └── model_trainer.py       # Model training and evaluation
│   └── pipeline/                  # Prediction pipeline
│       ├── predict_pipeline.py    # Prediction logic
│       └── train_pipeline.py      # Training pipeline
└── templates/                     # Flask HTML templates
    ├── index.html                 # Home page
    └── home.html                  # Prediction form
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mlproject
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package** (optional):
   ```bash
   pip install -e .
   ```

## Usage

### Training the Model

Run the training pipeline:
```python
from src.pipeline.train_pipeline import TrainPipeline

train_pipeline = TrainPipeline()
train_pipeline.run_pipeline()
```

### Running the Web Application

1. Start the Flask app:
   ```bash
   python app.py
   ```

2. Open your browser and go to `http://localhost:5000`

3. Fill in the student details in the form and click "Predict" to get the math score prediction.

### Making Predictions Programmatically

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create custom data object
data = CustomData(
    gender="female",
    race_ethnicity="group A",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=85.0,
    writing_score=88.0
)

# Get data as DataFrame
pred_df = data.get_data_as_data_frame()

# Make prediction
predict_pipeline = PredictPipeline()
result = predict_pipeline.predict(pred_df)
print(f"Predicted Math Score: {result[0]}")
```

## Model Performance

The model is evaluated using R² score and other regression metrics. The best performing model is saved as `model.pkl` in the artifacts directory.

## Data Preprocessing

- **Feature Engineering**: Categorical variables are encoded using appropriate techniques
- **Scaling**: Numerical features are standardized using StandardScaler
- **Handling Missing Values**: No missing values in the dataset
- **Outlier Treatment**: Not required for this dataset

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Shubham Gupta**
- Email: shubhamgupta262005@gmail.com

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Inspired by various ML tutorials and best practices

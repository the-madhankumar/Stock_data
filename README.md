

# Stock Price Prediction Using Linear Regression

## Overview

This project aims to predict stock prices using linear regression. The primary objective is to forecast the **Adjusted Close** prices of a stock based on historical data. The analysis involves calculating the **percentage change** in stock prices, applying **Exponential Moving Averages (EMA)** for trend smoothing, and evaluating the model performance using various metrics. The project also includes the visualization of actual vs predicted stock prices and residuals to analyze the model's performance.

---

## Files and Directories

- `stock_data.csv`: The dataset containing historical stock data (Date, Adjusted Close, Open, Close, High, Low, Volume).
- `stock_prediction.ipynb`: The Jupyter notebook containing the complete analysis, from data preprocessing to model evaluation.
- `requirements.txt`: A file listing the Python libraries required to run the project.

---

## Requirements

### Python Libraries:
To run this project, you will need to install the following libraries:

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **matplotlib**: Visualization library for plotting graphs.
- **seaborn**: Statistical data visualization.
- **scikit-learn**: Machine learning library for regression modeling.

You can install the required libraries by running the following command:

```bash
pip install -r requirements.txt
```

---

## Getting Started

1. Clone the repository or download the files:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook `stock_prediction.ipynb` in Jupyter Lab or Jupyter Notebook.

4. Follow the instructions within the notebook to load the dataset, preprocess the data, train the model, and visualize the results.

---

## Steps Performed in the Analysis

### 1. **Data Preprocessing**

The dataset contains columns such as `Price`, `Adj Close`, `Close`, `High`, `Low`, `Open`, and `Volume`. We performed the following preprocessing steps:

- **Set the index**: The Date was set as the index for time-series analysis.
- **Selected relevant columns**: The `Adj Close` column was chosen as the target for stock price prediction.
- **Handle missing data**: Missing values (if any) are handled appropriately before training the model.

### 2. **Feature Engineering**

- **Percentage Change**: We calculated the percentage change in the `Adj Close` to understand the relative increase/decrease in stock price over time.
- **Exponential Moving Average (EMA)**: A 10-day EMA was computed to smooth out short-term fluctuations and highlight longer-term trends.

### 3. **Linear Regression Model**

- We used **Linear Regression** from `scikit-learn` to model the relationship between the stock's adjusted closing prices and its historical features.
- The model was trained on the data, and the coefficients, Mean Absolute Error (MAE), and R² (coefficient of determination) were calculated to evaluate the model's performance.

### 4. **Model Evaluation**

The model was evaluated using:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors between predicted and actual values.
- **Coefficient of Determination (R²)**: Indicates how well the model explains the variance in the target variable.

### 5. **Visualization**

- **Actual vs Predicted**: A plot showing both the actual and predicted adjusted close prices to visualize how well the model fits the data.
- **Residuals**: The residuals (the difference between actual and predicted values) were plotted to check for patterns that may suggest model improvements.
- **Residuals Histogram**: A histogram of the residuals was plotted to assess if they follow a normal distribution.

---

## Example Output

Upon running the model, you should get the following outputs:

- **Linear Regression Model Coefficients**: 
   - Coefficients of the model, showing the relationship between the input features and the target variable.
   
- **Evaluation Metrics**: 
   - Mean Absolute Error: `2.95`
   - R² (Coefficient of Determination): `0.98`

- **Visualizations**:
   - **Actual vs Predicted Prices**: A time-series plot showing the adjusted closing prices against the model’s predictions.
   - **Residual Plot**: A plot showing the residuals, which should ideally have no clear pattern.
   - **Residuals Histogram**: A histogram of residuals to check for a normal distribution.

---

## Model Performance

The linear regression model achieved a high **R² score (0.98)**, indicating that the model explains 98% of the variance in the adjusted close price data. The **Mean Absolute Error (MAE)** was calculated to be approximately 2.96, which suggests that on average, the model's predictions are off by around 2.96 units of price.

While the model performs well, it’s essential to remember that stock price prediction is inherently uncertain due to external factors. The model can be further enhanced with additional features and more complex algorithms.

---

## Conclusion

This project demonstrates the application of **Linear Regression** for stock price prediction using historical data. By leveraging techniques like **EMA** and **percentage change**, the model is able to identify trends and make accurate predictions. This project serves as a foundation for more complex models and real-world applications in stock market analysis.

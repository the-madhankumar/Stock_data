
# Stock Price Prediction Using Linear Regression and Principal Component Analysis (PCA)

## Overview
This project aims to predict stock prices using Linear Regression, augmented by **Principal Component Analysis (PCA)** for dimensionality reduction. The objective is to forecast the Adjusted Close prices of a stock based on historical data, while PCA helps simplify the model by reducing the feature set, retaining the essential information. The analysis involves calculating the percentage change in stock prices, applying Exponential Moving Averages (EMA) for trend smoothing, and evaluating model performance using various metrics. The project also includes visualization of actual vs. predicted stock prices and residuals for model performance analysis.

## Files and Directories
- `stock_data.csv`: The dataset containing historical stock data (Date, Adjusted Close, Open, Close, High, Low, Volume).
- `main.ipynb`: The Jupyter notebook with the complete analysis, from data preprocessing to model evaluation.
- `requirements.txt`: A file listing the Python libraries required to run the project.

## Requirements
### Python Libraries
To run this project, you will need the following libraries:

- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical operations.
- `matplotlib`: Visualization library for plotting graphs.
- `seaborn`: Statistical data visualization.
- `scikit-learn`: Machine learning library for regression modeling and PCA.

Install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Getting Started
1. **Clone the repository** or download the files:

    ```bash
    git clone https://github.com/the-madhankumar/Stock_data.git
    cd Stock_data
    ```

2. **Install the required libraries**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Open the Jupyter notebook** `main.ipynb` in Jupyter Lab or Jupyter Notebook.

4. **Follow the instructions** in the notebook to load the dataset, preprocess the data, train the model, apply PCA, and visualize the results.

## Steps Performed in the Analysis
### 1. Data Preprocessing
   - **Set the index**: The Date column was set as the index for time-series analysis.
   - **Select relevant columns**: The `Adj Close` column was chosen as the target for stock price prediction.
   - **Handle missing data**: Missing values (if any) were handled before training the model.

### 2. Feature Engineering
   - **Percentage Change**: Calculated to understand the relative increase/decrease in stock price over time.
   - **Exponential Moving Average (EMA)**: A 10-day EMA was computed to smooth short-term fluctuations.

### 3. Dimensionality Reduction with PCA
   - **PCA Application**: Principal Component Analysis was applied to the feature set to reduce dimensionality, retaining key information while minimizing complexity.
   - **Component Selection**: The number of components for PCA was selected based on variance retention and analysis needs (e.g., `n_components=7`).
   - **Reduced Feature Set**: PCA-transformed features were used to train the model, simplifying the feature space and improving model interpretability.

### 4. Linear Regression Model
   - **Model Training**: Linear Regression was used to model the relationship between the PCA-transformed stock features and the adjusted close prices.
   - **Evaluation Metrics**: Calculated metrics included coefficients, Mean Squared Error (MSE), and R² (coefficient of determination).

### 5. Model Evaluation
   - **Mean Squared Error (MSE)**: Measures the average magnitude of the errors between predicted and actual values.
   - **Coefficient of Determination (R²)**: Indicates how well the model explains variance in the target variable.
   
### 6. Visualization
   - **Actual vs Predicted**: A plot showing both actual and predicted adjusted close prices to assess model fit.
   - **Residuals Plot**: The residuals (differences between actual and predicted values) were plotted to check for any patterns.
   - **Residuals Histogram**: A histogram of the residuals was plotted to assess if they follow a normal distribution.

## Example Output
Upon running the model, you should see:

1. **Linear Regression Model Coefficients**:
   - Coefficients representing relationships between the PCA-transformed input features and the target variable.

2. **Evaluation Metrics**:
   - Mean Squared Error: ~0.00075 (or similar, based on model tuning).
   - R² (Coefficient of Determination): ~0.999, indicating high explanatory power.

3. **Visualizations**:
   - **Actual vs Predicted Prices**: Time-series plot showing adjusted closing prices and model predictions.
   - **Residual Plot**: A plot to verify if residuals show any patterns.
   - **Residuals Histogram**: A histogram to assess the normal distribution of residuals.

## Model Performance
The model's high **R² score (0.999)** suggests it explains nearly all variance in the adjusted close price data, with an **MSE close to zero**. The use of PCA effectively reduced dimensionality, allowing the model to capture key data patterns while simplifying the dataset.

### Note
While the model performs well, remember that stock price prediction is inherently uncertain due to external factors. Adding more features or applying complex models like neural networks could improve performance in a more robust prediction system.

## Conclusion
This project demonstrates the application of Linear Regression for stock price prediction, enhanced by PCA for dimensionality reduction. By leveraging techniques like EMA, percentage change, and PCA, the model identifies trends and provides accurate predictions, serving as a foundation for further work in stock market analysis.

# About Reservoir Computing Project

## Project Overview

This project implements and compares three different reservoir computing approaches for stock price time series forecasting:

- **Echo State Networks (ESN)** - Traditional recurrent neural networks with fixed random reservoir
- **Liquid State Machines (LSM)** - Spiking neural reservoirs with rate-to-spike conversion  
- **Physical Reservoir Simulation** - Software simulation of physical nonlinear dynamical systems

## Project Structure

```
reservoir_computing/
├── README.md                 # Quick start guide
├── ABOUT.md                  # Detailed project documentation (this file)
├── pyproject.toml            # Python dependencies and project metadata
├── .gitignore                # Git ignore patterns
├── LICENSE                   # MIT License
├── 
├── Core Modules/
│   ├── ticker.py             # Stock data downloading and scraping
│   ├── data_utils.py          # Data loading and preprocessing utilities
│   └── logger.py             # Custom logging system
├── Simulation Notebooks/
│   ├── esn_reservoir.ipynb   # ESN experimentation notebook
│   ├── lsm_reservoir.ipynb   # LSM experimentation notebook
│   └── physical_reservoir.ipynb # Physical reservoir notebook
├── Data/
│   └── tickers/              # Downloaded stock CSV files (git-ignored)
├── Models/
│   └── models/                # Trained model artefacts (git-ignored)
└── Logs/
    └── logs/                  # Application logs (git-ignored)
```

## Reservoir Computing Approaches

### 1. Echo State Networks (ESN)

- **Architecture:** Fixed random recurrent weights with trained linear readout
- **Key Features:** Spectral radius control, input scaling, leaky integration
- **Performance:** Best performing approach with positive R² (~0.65) and baseline improvement
- **Use Case:** Traditional time series forecasting with stable dynamics

### 2. Liquid State Machines (LSM)  

- **Architecture:** Spiking neurons with leaky integrate-and-fire dynamics
- **Key Features:** Rate-to-spike conversion, spike count features, configurable connectivity
- **Performance:** Underperforming with negative R², needs hyperparameter tuning
- **Use Case:** Neuromorphic computing applications, event-driven processing

### 3. Physical Reservoir Simulation

- **Architecture:** Nonlinear delay-based dynamical system
- **Key Features:** Virtual node masking, feedback control, configurable nonlinearity
- **Performance:** Moderate results with negative R², potential for optimization
- **Use Case:** Physical reservoir computing, analogue computing simulation

## Data Pipeline

1. **Data Acquisition:** `ticker.py` scrapes financial news and downloads stock data via yfinance
2. **Data Processing:** `data_utils.py` loads CSVs, aligns time series, and creates windows
3. **Model Training:** Each reservoir implementation trains Ridge regression readout
4. **Evaluation:** Comprehensive metrics including baseline comparisons and directional accuracy
5. **Visualisation:** Enhanced plots with confidence bands and residual analysis

## Performance Evaluation Framework

All models implement standardised evaluation:

### Core Metrics

- **MSE/MAE:** Standard regression error measures
- **R-squared:** Variance explained (positive = good, negative = worse than mean)
- **Baseline Comparison:** Performance vs naive "tomorrow = today" prediction
- **Directional Accuracy:** Percentage of correct up/down predictions

### Advanced Analysis

- **Residual Analysis:** Error pattern detection and randomness verification
- **Confidence Bands:** Moving average uncertainty visualization
- **Comprehensive Logging:** Detailed execution tracking via custom logger

## Key Findings

- **ESN significantly outperforms** both LSM and Physical approaches
- **ESN achieves positive R² (0.65)** indicating meaningful pattern capture
- **LSM and Physical reservoirs show negative R²**, suggesting architectural mismatch
- **Directional accuracy varies** 36-53% across approaches (50% = random chance)

## Technical Implementation

### Dependencies

- **Core:** NumPy, Pandas, scikit-learn, Matplotlib
- **Data:** yfinance for stock data, requests/lxml for scraping
- **Utilities:** joblib for model persistence, custom logging system

### Design Patterns

- **Consistent API:** All reservoirs follow similar interface patterns
- **Modular Architecture:** Separate data, model, and evaluation components
- **Comprehensive Testing:** Standardized metrics across all approaches
- **Reproducible Results:** Fixed random seeds and persistent logging

## Usage Recommendations

1. **Start with ESN:** Best performing approach for stock forecasting
2. **Use ticker.py first:** Download data before running reservoirs
3. **Monitor logs:** Check `logs/` directory for detailed execution info
4. **Compare models:** All three approaches can be run for direct comparison
5. **Experiment freely:** Notebooks allow easy parameter tuning and testing

## Future Enhancements

- **Hyperparameter Optimisation:** Automated tuning for LSM and Physical reservoirs
- **Feature Engineering:** Technical indicators, volume, sentiment analysis
- **Multi-step Prediction:** Beyond single-step forecasting
- **Real-time Integration:** Live data feeds and online learning
- **Advanced Architectures:** Deep reservoirs, coupled reservoir systems

## Research Context

This project serves as a comprehensive foundation for reservoir computing research in financial time series forecasting, providing both traditional (ESN) and alternative (LSM, Physical) approaches with standardised evaluation for fair comparison and further development.

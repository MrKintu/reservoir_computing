# Reservoir Computing for Stock Time Series

A comprehensive implementation and comparison of reservoir computing approaches for financial time series forecasting.

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Download stock data
uv run ticker.py

# 3. Train reservoir models
uv run esn_reservoir.py      # Best performing approach
uv run lsm_reservoir.py       # Spiking neural approach  
uv run physical_reservoir.py   # Physical simulation approach

# 4. Experiment with notebooks
jupyter esn_reservoir.ipynb
jupyter lsm_reservoir.ipynb
jupyter physical_reservoir.ipynb
```

## 📁 Project Structure

```
reservoir_computing/
├── 📊 Data Pipeline/
│   ├── ticker.py              # Stock data downloader & scraper
│   ├── data_utils.py           # Data loading & preprocessing
│   └── logger.py              # Custom logging system
├── 📓 Notebooks/
│   ├── esn_reservoir.ipynb     # ESN experimentation (⭐ BEST)
│   ├── lsm_reservoir.ipynb     # LSM experimentation  
│   └── physical_reservoir.ipynb # Physical reservoir exp
├── 💾 Data/
│   ├── tickers/               # Stock CSV files (auto-generated)
│   ├── models/                # Trained models (auto-generated)
│   └── logs/                 # Execution logs (auto-generated)
└── 📚 Documentation/
    ├── README.md               # This file
    ├── ABOUT.md                # Detailed project documentation
    └── pyproject.toml          # Dependencies
```

## 🎯 What This Project Does

### Three Reservoir Computing Approaches

1. **Echo State Networks (ESN)** ⭐ *Recommended*
   - Traditional recurrent reservoir with fixed random weights
   - **Best performance:** R² = 0.65, beats naive baseline
   - Use for: Standard time series forecasting

2. **Liquid State Machines (LSM)**
   - Spiking neurons with rate-to-spike conversion
   - **Experimental:** Needs hyperparameter tuning
   - Use for: Neuromorphic computing research

3. **Physical Reservoir Simulation**
   - Software simulation of physical nonlinear dynamics
   - **Research focus:** Analog computing concepts
   - Use for: Physical reservoir exploration

### Performance Summary

| Approach | MSE | MAE | R² | vs Naive | Directional Acc |
|-----------|------|-----|-----|-----------|----------------|
| ESN       | 0.48 | 0.38 | 0.65 | +11.98% | 36.84% |
| LSM       | 4.40 | 1.96 | -2.23 | -938.67% | 52.63% |
| Physical  | 1.67 | 0.90 | -0.22 | -293.20% | 42.11% |

## 🛠️ How to Use

### For Stock Prediction (Recommended)

```bash
# Download data first
uv run ticker.py

# Train best model (ESN)
jupyter esn_reservoir.ipynb

# Check results in logs/
cat logs/esn_reservoir_YYYYMMDD.log
```

### For Research & Experimentation

```bash
# Compare all approaches
uv run ticker.py
jupyter esn_reservoir.ipynb
jupyter lsm_reservoir.ipynb  
jupyter physical_reservoir.ipynb

# Analyse performance differences
grep -r "Test MSE" logs/
```

### For Development

```bash
# Open notebooks for interactive experimentation, jupyter esn_reservoir.ipynb

# Modify hyperparameters in .py files
# Key parameters: n_reservoir, spectral_radius, connectivity, etc.
```

## 📊 Understanding the Output

### Log Files (logs/)

- **INFO:** Progress tracking and model performance
- **DEBUG:** Detailed technical information
- **WARNING:** Non-critical issues (missing data, etc.)
- **ERROR:** Critical problems

### Model Files (models/)

- `esn_model.pkl` - Trained ESN with readout
- `lsm_model.pkl` - Trained LSM with readout  
- `physical_reservoir_model.pkl` - Trained Physical reservoir

### Key Metrics

- **MSE/MAE:** Lower is better
- **R²:** Positive = good, negative = worse than predicting mean
- **vs Naive:** Positive improvement = beats simple baseline
- **Directional Acc:** >50% = better than random chance

## ⚙️ Configuration

### Environment Setup

```bash
# Python 3.11+ required
python --version

# Install with uv (recommended)
pip install uv
uv sync

# Or traditional pip
pip install -r pyproject.toml
```

### Data Sources

- **Primary:** yfinance API for historical OHLCV data
- **Secondary:** PR Newswire scraping for ticker discovery
- **Format:** CSV files with Date, Open, High, Low, Close, Volume

## 🔧 Customization

### Hyperparameter Tuning

```python
# In esn_reservoir.py
esn = ESN(
    n_reservoir=400,      # Reservoir size (try 200-1000)
    spectral_radius=0.9,    # Dynamics stability (0.5-0.99)
    input_scaling=0.5,     # Input influence (0.1-2.0)
    leak_rate=0.3          # Memory retention (0.1-1.0)
)

# In lsm_reservoir.py  
lsm = SimpleLSM(
    n_neurons=400,          # Neuron count
    connectivity=0.05,       # Sparsity (0.01-0.2)
    tau=8.0,               # Time constant (1.0-20.0)
)

# In physical_reservoir.py
dr = DelayReservoir(
    n_neurons=300,          # Virtual nodes
    input_gain=0.6,         # Input scaling
    feedback=0.95,           # Feedback strength
)
```

## 📈 Next Steps

1. **Production:** Deploy the ESN model for real-time prediction
2. **Enhancement:** Add technical indicators as features
3. **Optimization:** Automated hyperparameter tuning
4. **Expansion:** Multi-step prediction capabilities
5. **Research:** Improve LSM and Physical reservoir performance

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request

---

**💡 Tip:** Start with `esn_reservoir.ipynb` - it's the proven performer. Use the notebooks for interactive experimentation before modifying the core Python files.

# Wind Speed Forecasting with Neural Networks 🌬️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.13.1-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.21.2-blue.svg)](https://scikit-learn.org/)

A machine learning project that implements a Multilayer Perceptron (MLP) neural network for accurate wind speed forecasting. This project demonstrates the application of deep learning techniques in time series forecasting and environmental data analysis.

## 🎯 Features

- **Advanced Neural Network Architecture**: Implements a 4-layer MLP with optimized neuron configuration
- **High-Performance Metrics**: Achieves impressive results with:
  - MAE: 0.77821
  - MSE: 0.99975
  - Pearson Correlation Coefficient: 0.93546
  - R² Score: 0.99474
- **Data Preprocessing**: Includes comprehensive data normalization and scaling
- **Visualization Tools**: Built-in plotting capabilities for model performance analysis
- **Model Persistence**: Save and load trained models for future predictions

## 🏗️ Architecture

The neural network architecture consists of:
- Input Layer: 9 neurons
- First Hidden Layer: 9 neurons
- Second Hidden Layer: 6 neurons
- Output Layer: 1 neuron

All hidden layers use the tanh activation function, while the output layer uses a linear activation function for regression tasks.

## 🚀 Getting Started

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/patricksferraz/wind-speed-forecast.git
cd wind-speed-forecast
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Train the model:
```bash
python rna.py --dataset path/to/your/data.txt --model output/model.h5 --plot output/plot.png --norm output/scaler
```

2. Make predictions:
```bash
python predict_rna.py --model output/model.h5 --input your_input_data.txt
```

## 📊 Results

The model demonstrates excellent performance in wind speed forecasting:
- Minimum Error: 0.00396
- Maximum Error: 3.05602
- High correlation coefficient (r = 0.93546) indicates strong predictive power
- R² score of 0.99474 shows exceptional model fit

## 🛠️ Technologies

- TensorFlow 1.13.1
- scikit-learn 0.21.2
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📫 Contact

Patrick Ferraz - [GitHub Profile](https://github.com/patricksferraz)

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for their invaluable tools and libraries

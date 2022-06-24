![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

# Project: COVID19 Daily Cases Forecaster

# Description
The project is about developing a deep learning model using LSTM neural networks to forecast/predict the daily cases of COVID19. This is done by training a LSTM deep learning model on the COVID19 daily cases data obtained from [Malaysia Ministry of Health (MOH) COVID19 GitHub Repository](https://github.com/MoH-Malaysia/covid19-public). The main layer of the LSTM deep learning model is the LSTM layer that is inputted with the aforementioned daily cases data, containing nearly 700 daily records.

# How to Install and Run the Project
To run and train the model on your own device, clone the whole repository first. Then, proceed to the directory containing the cloned repository. In this particular directory, locate the `covid_case_predictor.py` file and run this file in your terminal or any of your favorite IDEs. This will generate all the relevant plots and results especially the trained LSTM deep learning model.

# Results
## Neural Network Model Summary & Plot
![LSTM summary](statics/model_summary.png)
![LSTM plot](statics/model_plot.png)

## Model Training MSE Loss and MAPE Metric
### Matplotlib Plot
![loss metric mpl](statics/model_loss_metric_matplotlib_plot.png)
### Smoothed Tensorboard Plot
![loss metric tensorboard](statics/base_performance_loss_metric_tensorboard.png)

## Model Key Performance Metrics
![key metrics](statics/model_metrics.png)

## Model Predictions
![predictions](statics/model_prediction.png)

# Credits
- [Malaysia Ministry of Health (MOH) GitHub Repository](https://github.com/MoH-Malaysia/covid19-public)
- [Markdown badges source 1](https://github.com/Ileriayo/markdown-badges)
- [Markdown badges source 2](https://github.com/alexandresanlim/Badges4-README.md-Profile)

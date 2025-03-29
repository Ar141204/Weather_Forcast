from django.shortcuts import render
from pathlib import Path
import requests  # Library for fetching data from APIs
import pandas as pd  # For handling and analyzing data
import numpy as np  # For numerical operations
import pytz
import os
from django.conf import settings
from django.views.decorators.csrf import ensure_csrf_cookie

# Get the BASE_DIR
BASE_DIR = Path(__file__).resolve().parent.parent

# Machine Learning utilities
from sklearn.model_selection import train_test_split  # Splitting data into training and testing sets
from sklearn.preprocessing import LabelEncoder  # Converting categorical data into numerical values

# Models for classification and regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Metric for evaluating model performance
from sklearn.metrics import mean_squared_error  

# Handling date and time
from datetime import datetime, timedelta  

from django.http import HttpResponse

# Remove hardcoded API keys and use settings
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

#1.Fetch Current Weather Data

def get_current_weather(city):
    try:
        url = f"{BASE_URL}weather?q={city}&appid={settings.WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 404:
            return {'error': 'City not found. Please check the spelling and try again.'}
        elif response.status_code != 200:
            return {'error': 'Weather service is temporarily unavailable. Please try again later.'}
        
        if 'main' not in data or 'weather' not in data:
            return {'error': 'Invalid response from weather service. Please try again.'}

        return {
            'city': data.get('name', 'Unknown City'),
            'current_temp': round(data.get('main', {}).get('temp', 0)),
            'feels_like': round(data.get('main', {}).get('feels_like', 0)),
            'temp_min': round(data.get('main', {}).get('temp_min', 0)),
            'temp_max': round(data.get('main', {}).get('temp_max', 0)),
            'humidity': round(data.get('main', {}).get('humidity', 0)),
            'description': data.get('weather', [{}])[0].get('description', 'No description'),
            'country': data.get('sys', {}).get('country', 'Unknown'),
            'wind_gust_dir': data.get('wind', {}).get('deg', 0),
            'pressure': data.get('main', {}).get('pressure', 0),
            'Wind_Gust_Speed': data.get('wind', {}).get('speed', 0),
            'clouds': data.get('clouds', {}).get('all', 0),
            'visibility': data.get('visibility', 10000) / 1000,  # Convert to kilometers
            'timezone': data.get('timezone', 0),
        }
    except requests.RequestException:
        return {'error': 'Failed to connect to weather service. Please check your internet connection.'}
    except Exception as e:
        return {'error': f'An unexpected error occurred: {str(e)}'}



#2.Read Historical Data

def read_historical_data(filename):
    df=pd.read_csv(filename) #load csv file into dataframe
    df=df.drop_duplicates()
    return df

#3.Prepare Data for Training

def prepare_data(data):
    if 'WindGustDir' not in data.columns or 'RainTomorrow' not in data.columns:
        raise ValueError("Historical data is missing required columns: 'WindGustDir' or 'RainTomorrow'")

    le = LabelEncoder()  # Create a LabelEncoder instance
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    # Define the feature variables and target variables
    x = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]  # Feature variables
    y = data['RainTomorrow']  # Target variable

    return x, y, le  # Return feature variables, target variable, and the LabelEncoder

#4.Train Rain Prediction Model

def train_rain_model(x,y):
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    model=RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(x_train,y_train) #train the model

    y_pred=model.predict(x_test) #to make predictions on test set

    print("Mean Squared Error for Rain Model")

    print(mean_squared_error(y_test,y_pred))

    return model

#5.Prepare Regression Data

def prepare_regression_data(data, feature):
    if feature not in data.columns:
        raise ValueError(f"Feature '{feature}' not found in historical data")

    x, y = [], []  # Initialize lists for feature and target values

    for i in range(len(data) - 1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])

    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    return x, y  # Return correctly after loop


     
#6.Train Regression Model

def train_regression_model(x,y):
    model=RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x,y)

    return model

#7.Predict Future
def predict_future(model, current_value):
    predictions = [current_value]

    for _ in range(5):  # Predict 5 future values
        next_value = model.predict(np.array([[predictions[-1]]]))[0]
        predictions.append(next_value)

    return predictions[1:]  # Return all predictions after the initial current value

# 8

def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city', '').strip()
        if not city:
            return render(request, 'weather.html', {'error': 'Please enter a city name.'})
    else:
        city = 'London'  # default city

    current_weather = get_current_weather(city)
    
    if 'error' in current_weather:
        return render(request, 'weather.html', {'error': current_weather['error'], 'city': city})

    try:
        # Prepare time for future predictions
        now = datetime.now()
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        # Simple temperature prediction (gradual decrease then increase)
        current_temp = float(current_weather['current_temp'])
        temp_variations = [-0.4, -3.0, -3.2, -5.1, -2.1]  # Temperature variations
        future_temps = [current_temp + variation for variation in temp_variations]

        # Simple humidity prediction (gradual change)
        current_humidity = float(current_weather['humidity'])
        humidity_variations = [-5.2, -13.8, -24.8, -23.7, -21.8]  # Humidity variations
        future_humidity = [max(min(current_humidity + variation, 100), 0) for variation in humidity_variations]

        context = {
            'location': city,
            'current_temp': current_weather['current_temp'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'city': current_weather['city'],
            'country': current_weather['country'],
            'date': datetime.now().strftime("%B %d, %Y"),
            'wind': current_weather['Wind_Gust_Speed'],
            'pressure': current_weather['pressure'],
            'visibility': current_weather['visibility'],
            'time1': future_times[0],
            'time2': future_times[1],
            'time3': future_times[2],
            'time4': future_times[3],
            'time5': future_times[4],
            'temp1': f"{round(future_temps[0], 1)}",
            'temp2': f"{round(future_temps[1], 1)}",
            'temp3': f"{round(future_temps[2], 1)}",
            'temp4': f"{round(future_temps[3], 1)}",
            'temp5': f"{round(future_temps[4], 1)}",
            'hum1': f"{round(future_humidity[0], 1)}",
            'hum2': f"{round(future_humidity[1], 1)}",
            'hum3': f"{round(future_humidity[2], 1)}",
            'hum4': f"{round(future_humidity[3], 1)}",
            'hum5': f"{round(future_humidity[4], 1)}",
        }

        return render(request, 'weather.html', context)

    except Exception as e:
        return render(request, 'weather.html', {
            'error': f"An error occurred while processing weather data: {str(e)}",
            'city': city
        })

from django.shortcuts import render  # Ensure this is imported

def my_view(request):  # Ensure the request parameter exists
    if request.method == "POST":
        print(request.POST)  # Now it will work correctly
    return render(request, "some_template.html")

@ensure_csrf_cookie  # Add this decorator
def index(request):
    return render(request, 'index.html')


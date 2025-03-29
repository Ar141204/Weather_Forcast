from django.shortcuts import render
from pathlib import Path
import requests  # Library for fetching data from APIs
import pandas as pd  # For handling and analyzing data
import numpy as np  # For numerical operations
import pytz
import os
from django.conf import settings
from django.views.decorators.csrf import ensure_csrf_cookie
from django.http import JsonResponse

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

@ensure_csrf_cookie
def index(request):
    return render(request, 'weather.html')

def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        api_key = os.getenv('WEATHER_API_KEY')
        
        # Get current weather
        current_url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
        # Get forecast
        forecast_url = f'http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric'
        
        try:
            current_response = requests.get(current_url)
            forecast_response = requests.get(forecast_url)
            
            if current_response.status_code == 200 and forecast_response.status_code == 200:
                current_data = current_response.json()
                forecast_data = forecast_response.json()
                
                # Process forecast data for the next 5 time periods
                forecast_times = []
                forecast_temps = []
                forecast_hums = []
                
                for i, item in enumerate(forecast_data['list'][:5]):
                    time = datetime.fromtimestamp(item['dt']).strftime('%H:%M')
                    forecast_times.append(time)
                    forecast_temps.append(round(item['main']['temp']))
                    forecast_hums.append(item['main']['humidity'])

                context = {
                    'city': city,
                    'country': current_data['sys']['country'],
                    'date': datetime.now().strftime('%B %d, %Y'),
                    'description': current_data['weather'][0]['description'],
                    'current_temp': round(current_data['main']['temp']),
                    'feels_like': round(current_data['main']['feels_like']),
                    'humidity': current_data['main']['humidity'],
                    'pressure': current_data['main']['pressure'],
                    'wind': current_data['wind']['speed'],
                    'clouds': current_data['clouds']['all'],
                    'visibility': current_data.get('visibility', 'N/A'),
                    'MaxTemp': round(current_data['main']['temp_max']),
                    'MinTemp': round(current_data['main']['temp_min']),
                    
                    # Forecast data
                    'time1': forecast_times[0],
                    'time2': forecast_times[1],
                    'time3': forecast_times[2],
                    'time4': forecast_times[3],
                    'time5': forecast_times[4],
                    'temp1': forecast_temps[0],
                    'temp2': forecast_temps[1],
                    'temp3': forecast_temps[2],
                    'temp4': forecast_temps[3],
                    'temp5': forecast_temps[4],
                    'hum1': forecast_hums[0],
                    'hum2': forecast_hums[1],
                    'hum3': forecast_hums[2],
                    'hum4': forecast_hums[3],
                    'hum5': forecast_hums[4],
                }
                return render(request, 'weather.html', context)
            else:
                return render(request, 'weather.html', {'error': 'City not found'})
                
        except Exception as e:
            return render(request, 'weather.html', {'error': str(e)})
    
    return render(request, 'weather.html')

from django.shortcuts import render  # Ensure this is imported

def my_view(request):  # Ensure the request parameter exists
    if request.method == "POST":
        print(request.POST)  # Now it will work correctly
    return render(request, "some_template.html")

def weather(request):
    if request.method == 'POST':
        try:
            import json
            data = json.loads(request.body)
            city = data.get('city')
            
            # Get API key from environment variable
            api_key = os.getenv('WEATHER_API_KEY')
            
            # Make request to OpenWeatherMap API
            url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
            response = requests.get(url)
            weather_data = response.json()
            
            if response.status_code == 200:
                return JsonResponse({
                    'city': weather_data['name'],
                    'temperature': weather_data['main']['temp'],
                    'description': weather_data['weather'][0]['description'],
                    'humidity': weather_data['main']['humidity'],
                    'wind_speed': weather_data['wind']['speed']
                })
            else:
                return JsonResponse({'error': 'City not found'}, status=404)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request method'}, status=400)


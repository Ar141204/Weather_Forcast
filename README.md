# Weather Forecast Application

A Django-based weather forecast application that provides real-time weather information using the OpenWeatherMap API.

## Features

- Real-time weather data retrieval
- City-based weather search
- Current temperature, humidity, and wind speed display
- Weather condition descriptions
- Responsive design using Bootstrap
- Error handling for invalid requests
- Data analysis with Jupyter notebook
- CSV data export capability

## Project Structure

```
Weather_Forecast/
├── weatherProject/
│   ├── manage.py                # Django management script
│   ├── weatherProject/
│   │   ├── __init__.py
│   │   ├── settings.py         # Project settings
│   │   ├── urls.py            # Main URL configuration
│   │   ├── asgi.py            # ASGI configuration
│   │   └── wsgi.py            # WSGI configuration
│   └── forecast/
│       ├── __init__.py
│       ├── admin.py           # Admin interface configuration
│       ├── apps.py           # App configuration
│       ├── models.py         # Database models
│       ├── views.py          # View functions
│       ├── urls.py           # App URL patterns
│       └── tests.py          # Unit tests
├── templates/
│   ├── index.html            # Main page template
│   ├── 404.html             # Not found error page
│   └── 500.html             # Server error page
├── src/
│   └── weather_utils.py      # Weather utility functions
├── weather.ipynb             # Jupyter notebook for data analysis
├── weather.csv               # Weather data export
├── requirements.txt          # Project dependencies
├── .env.example              # Example environment configuration
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- OpenWeatherMap API key
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Weather_Forecast.git
cd Weather_Forecast
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables by copying .env.example:
```bash
cp .env.example .env
```
Then edit .env with your actual configuration values.

## Configuration

1. Get your API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Update the `.env` file with your API key and other settings
3. Configure database settings if needed (default is SQLite)

## Running the Application

1. Make sure your virtual environment is activated
2. Navigate to the weatherProject directory:
```bash
cd weatherProject
```
3. Run the Django application:
```bash
python manage.py runserver
```
4. Open your browser and navigate to `http://127.0.0.1:8000`

## Data Analysis

The project includes a Jupyter notebook (`weather.ipynb`) for analyzing weather data. The analyzed data can be exported to `weather.csv`.

## API Endpoints

- `GET /` - Main page
- `POST /weather` - Get weather data for a city
  - Request body: `{"city": "city_name"}`
  - Returns: Current weather and forecast data

## Error Handling

- 404 - Page Not Found
- 500 - Internal Server Error
- Invalid city names
- API connection errors

## Technologies Used

- Django - Web framework
- OpenWeatherMap API - Weather data
- Bootstrap 5 - Frontend styling
- Geopy - Geocoding service
- Python-dotenv - Environment management
- Jupyter - Data analysis
- SQLite - Database

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

Copyright 2025 Abdul Rahman M

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Author

Abdul Rahman M

## Acknowledgments

- OpenWeatherMap for providing the weather data API
- Bootstrap team for the frontend framework

## Contact

For any queries or suggestions, please open an issue in the repository.

---
Made with ❤️ by Abdul Rahman M 
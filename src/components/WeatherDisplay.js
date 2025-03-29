import React from 'react';
import './WeatherDisplay.css';

const WeatherDisplay = ({ weatherData }) => {
  return (
    <div className="weather-container">
      <div className="search-box">
        <span className="cloud-icon">☁️</span>
        <input type="text" defaultValue={weatherData.location} placeholder="Search location..." />
      </div>

      <div className="main-content">
        <div className="temperature-display">
          <h1 className="temp">{weatherData.temperature}°</h1>
          <p className="feels-like">Feels like: {weatherData.feelsLike}°</p>
        </div>

        <div className="info-boxes">
          <div className="info-box humidity">
            <h2>{weatherData.humidity}%</h2>
            <p>of humidity</p>
          </div>
          
          <div className="info-box clouds">
            <h2>{weatherData.clouds}%</h2>
            <p>of Clouds</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WeatherDisplay; 
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import json


class WeatherService:
    """
    A service class for retrieving weather data from the National Weather Service API
    and converting it to machine learning model categories.

    Features:
    - Current weather observations
    - Weather forecasts
    - Automatic weather category mapping for ML models
    - Smart fallbacks and error handling
    - Location geocoding
    """

    def __init__(self, default_location: str = "Pomona,CA", user_agent: str = None):
        """
        Initialize the WeatherService.

        Args:
            default_location: Default location for weather queries (format: "City,State")
            user_agent: User agent string for API requests
        """
        self.default_location = default_location
        self.user_agent = user_agent or "WeatherService/1.0 (weather@example.com)"
        self.logger = self._setup_logging()

        # Pre-defined coordinates for common locations (faster lookup)
        self.location_coords = {
            "pomona,ca": (34.0551, -117.7500),
            "los angeles,ca": (34.0522, -118.2437),
            "san francisco,ca": (37.7749, -122.4194),
            "san diego,ca": (32.7157, -117.1611),
            "sacramento,ca": (38.5816, -121.4944),
            "fresno,ca": (36.7378, -119.7871),
            "claremont,ca": (34.0967, -117.7198),
            "riverside,ca": (33.9533, -117.3962),
            "bakersfield,ca": (35.3733, -119.0187),
            "anaheim,ca": (33.8366, -117.9143),
            "long beach,ca": (33.7701, -118.1937),
            "oakland,ca": (37.8044, -122.2711),
            "san jose,ca": (37.3382, -121.8863),
            "pasadena,ca": (34.1478, -118.1445),
            "ontario,ca": (34.0633, -117.6509)
        }

        # ML model weather categories
        self.ml_categories = ["sunny", "cloudy", "rainy", "extreme_heat"]

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the weather service."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def get_current_weather(self, location: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current weather conditions for a location.

        Args:
            location: Location string (e.g., "Los Angeles,CA"). Uses default if None.

        Returns:
            Dict containing weather data and ML category
        """
        location = location or self.default_location

        try:
            # Get coordinates
            coords = self._get_coordinates(location)
            if not coords['success']:
                return self._get_default_weather_response(location, None, False, coords['error'])

            lat, lon = coords['latitude'], coords['longitude']

            # Get current weather
            weather_result = self._fetch_current_observations(lat, lon)
            if not weather_result['success']:
                return self._get_default_weather_response(location, None, False, weather_result['error'])

            # Process and return result
            return self._process_weather_result(weather_result['data'], location, coords, False)

        except Exception as e:
            self.logger.error(f"Error getting current weather: {e}")
            return self._get_default_weather_response(location, None, False, str(e))

    def get_forecast(self, location: Optional[str] = None, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get weather forecast for a location and date.

        Args:
            location: Location string (e.g., "Los Angeles,CA"). Uses default if None.
            date: Date string in YYYY-MM-DD format. Uses tomorrow if None.

        Returns:
            Dict containing forecast data and ML category
        """
        location = location or self.default_location

        # Parse or default date
        if date:
            try:
                target_date = datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                return {
                    'success': False,
                    'error': f'Invalid date format. Use YYYY-MM-DD. Received: {date}'
                }
        else:
            target_date = datetime.now() + timedelta(days=1)  # Tomorrow

        try:
            # Get coordinates
            coords = self._get_coordinates(location)
            if not coords['success']:
                return self._get_default_weather_response(location, date, True, coords['error'])

            lat, lon = coords['latitude'], coords['longitude']

            # Get forecast
            forecast_result = self._fetch_forecast(lat, lon, target_date)
            if not forecast_result['success']:
                return self._get_default_weather_response(location, date, True, forecast_result['error'])

            # Process and return result
            return self._process_weather_result(forecast_result['data'], location, coords, True, date)

        except Exception as e:
            self.logger.error(f"Error getting forecast: {e}")
            return self._get_default_weather_response(location, date, True, str(e))

    def get_weather_for_ml_prediction(self, location: Optional[str] = None, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get weather data specifically formatted for ML model predictions.

        Args:
            location: Location string
            date: Date string (YYYY-MM-DD) or None for current weather

        Returns:
            Dict with 'ml_category' and supporting data
        """
        if date:
            # Future date - get forecast
            result = self.get_forecast(location, date)
        else:
            # Current weather
            result = self.get_current_weather(location)

        if result['success']:
            return {
                'success': True,
                'ml_category': result['ml_category'],
                'temperature_celsius': result['weather_data']['temperature_celsius'],
                'temperature_fahrenheit': result['weather_data']['temperature_fahrenheit'],
                'description': result['weather_data']['weather_description'],
                'location': result['location_info']['resolved_location'],
                'data_source': result['weather_data']['data_source'],
                'mapping_explanation': result['mapping_explanation'],
                'timestamp': result['weather_data'].get('observation_time', result['weather_data'].get('forecast_time', '')),
                'is_forecast': date is not None
            }
        else:
            return result

    def map_weather_to_ml_category(self, weather_main: str, weather_description: str, temp_celsius: float) -> str:
        """
        Map weather conditions to ML model categories.

        Args:
            weather_main: Main weather category (e.g., "Clear", "Clouds", "Rain")
            weather_description: Detailed weather description
            temp_celsius: Temperature in Celsius

        Returns:
            ML category: "sunny", "cloudy", "rainy", or "extreme_heat"
        """
        # Check for extreme heat first (over 35°C / 95°F)
        if temp_celsius > 35:
            return "extreme_heat"

        # Normalize inputs
        weather_main = weather_main.lower().strip()
        weather_description = weather_description.lower().strip()

        # Map conditions to categories
        if any(word in weather_main for word in ["rain", "snow", "fog", "mist"]):
            return "rainy"
        elif "thunderstorm" in weather_description or "storm" in weather_description:
            return "rainy"
        elif weather_main == "clouds":
            # Distinguish between partly cloudy (sunny) and overcast (cloudy)
            if any(phrase in weather_description for phrase in ["partly", "few", "scattered", "mostly sunny", "mostly clear"]):
                return "sunny"
            else:
                return "cloudy"
        elif weather_main == "clear":
            return "sunny"
        else:
            # Default to sunny for unclear conditions
            return "sunny"

    def get_ml_categories(self) -> list:
        """Get list of available ML weather categories."""
        return self.ml_categories.copy()

    def _get_coordinates(self, location: str) -> Dict[str, Any]:
        """Get latitude/longitude coordinates for a location."""
        try:
            # Check predefined coordinates first
            location_key = location.lower().strip()
            if location_key in self.location_coords:
                lat, lon = self.location_coords[location_key]
                return {
                    'success': True,
                    'latitude': lat,
                    'longitude': lon,
                    'formatted_address': location,
                    'source': 'predefined'
                }

            # Try geocoding with US Census API
            try:
                geocode_url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
                params = {
                    'address': location,
                    'benchmark': 'Public_AR_Current',
                    'format': 'json'
                }

                response = requests.get(geocode_url, params=params, timeout=5)
                if response.status_code == 200:
                    geocode_data = response.json()
                    if geocode_data.get('result', {}).get('addressMatches'):
                        match = geocode_data['result']['addressMatches'][0]
                        coordinates = match['coordinates']
                        return {
                            'success': True,
                            'latitude': float(coordinates['y']),
                            'longitude': float(coordinates['x']),
                            'formatted_address': match['matchedAddress'],
                            'source': 'geocoded'
                        }
            except requests.exceptions.Timeout:
                self.logger.warning("Geocoding timeout, using default coordinates")
            except Exception as e:
                self.logger.warning(f"Geocoding failed: {e}")

            # Fallback to default location coordinates
            default_lat, default_lon = self.location_coords.get("pomona,ca", (34.0551, -117.7500))
            return {
                'success': True,
                'latitude': default_lat,
                'longitude': default_lon,
                'formatted_address': f"{location} (using default coordinates)",
                'source': 'fallback'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to get coordinates: {str(e)}'
            }

    def _fetch_current_observations(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch current weather observations from NWS API."""
        try:
            headers = {'User-Agent': self.user_agent}

            # Get nearest weather stations
            stations_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}/stations"
            response = requests.get(stations_url, headers=headers, timeout=10)

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'Failed to get weather stations: HTTP {response.status_code}'
                }

            stations_data = response.json()
            if not stations_data.get('features'):
                return {
                    'success': False,
                    'error': 'No weather stations found'
                }

            # Try multiple stations if needed
            for station in stations_data['features'][:3]:
                station_id = station['properties']['stationIdentifier']

                try:
                    obs_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
                    response = requests.get(obs_url, headers=headers, timeout=8)

                    if response.status_code == 200:
                        obs_data = response.json()
                        properties = obs_data['properties']

                        temp_celsius = properties.get('temperature', {}).get('value')
                        if temp_celsius is None:
                            continue  # Try next station

                        text_description = properties.get('textDescription', 'Clear')
                        weather_main, weather_description = self._parse_nws_conditions(text_description)

                        return {
                            'success': True,
                            'data': {
                                'station_id': station_id,
                                'station_name': station['properties'].get('name', 'Unknown'),
                                'observation_time': properties.get('timestamp', ''),
                                'temperature_celsius': round(temp_celsius, 1),
                                'temperature_fahrenheit': round(temp_celsius * 9/5 + 32, 1),
                                'weather_main': weather_main,
                                'weather_description': weather_description,
                                'humidity': properties.get('relativeHumidity', {}).get('value'),
                                'wind_speed': properties.get('windSpeed', {}).get('value'),
                                'data_source': 'nws_current'
                            }
                        }

                except requests.exceptions.Timeout:
                    continue  # Try next station

            return {
                'success': False,
                'error': 'All weather stations timed out or had no data'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error fetching observations: {str(e)}'
            }

    def _fetch_forecast(self, lat: float, lon: float, target_date: datetime) -> Dict[str, Any]:
        """Fetch weather forecast from NWS API."""
        try:
            headers = {'User-Agent': self.user_agent}

            # Get grid information
            points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
            response = requests.get(points_url, headers=headers, timeout=10)

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'Failed to get forecast grid: HTTP {response.status_code}'
                }

            points_data = response.json()
            forecast_url = points_data['properties']['forecast']

            # Get forecast data
            response = requests.get(forecast_url, headers=headers, timeout=12)

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'Failed to get forecast: HTTP {response.status_code}'
                }

            forecast_data = response.json()

            # Find best matching forecast period
            target_date_str = target_date.strftime('%Y-%m-%d')
            best_period = None

            for period in forecast_data['properties']['periods']:
                try:
                    period_start = datetime.fromisoformat(period['startTime'].replace('Z', '+00:00'))
                    period_date = period_start.strftime('%Y-%m-%d')

                    if period_date == target_date_str:
                        best_period = period
                        break
                except:
                    continue

            # Use first period if exact match not found
            if not best_period and forecast_data['properties']['periods']:
                best_period = forecast_data['properties']['periods'][0]

            if not best_period:
                return {
                    'success': False,
                    'error': f'No forecast data available for {target_date_str}'
                }

            # Extract forecast information
            temp_fahrenheit = best_period.get('temperature', 70)
            temp_celsius = round((temp_fahrenheit - 32) * 5/9, 1)

            forecast_text = best_period.get('detailedForecast', best_period.get('shortForecast', ''))
            weather_main, weather_description = self._parse_nws_conditions(forecast_text)

            return {
                'success': True,
                'data': {
                    'forecast_period': best_period.get('name', 'Unknown'),
                    'forecast_time': best_period.get('startTime', ''),
                    'temperature_celsius': temp_celsius,
                    'temperature_fahrenheit': temp_fahrenheit,
                    'weather_main': weather_main,
                    'weather_description': weather_description,
                    'detailed_forecast': forecast_text,
                    'is_daytime': best_period.get('isDaytime', True),
                    'data_source': 'nws_forecast'
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error fetching forecast: {str(e)}'
            }

    def _parse_nws_conditions(self, description: str) -> Tuple[str, str]:
        """Parse NWS weather description into simplified categories."""
        description_lower = description.lower()

        # Determine main weather category
        if any(word in description_lower for word in ['rain', 'shower', 'drizzle', 'thunderstorm', 'storm']):
            return ('Rain', description)
        elif any(word in description_lower for word in ['snow', 'sleet', 'ice', 'freezing']):
            return ('Snow', description)
        elif any(word in description_lower for word in ['fog', 'mist', 'haze']):
            return ('Fog', description)
        elif any(word in description_lower for word in ['cloud', 'overcast', 'partly']):
            if any(phrase in description_lower for phrase in ['partly cloudy', 'mostly sunny', 'few clouds', 'scattered clouds']):
                return ('Clear', description)
            else:
                return ('Clouds', description)
        elif any(word in description_lower for word in ['clear', 'sunny', 'fair']):
            return ('Clear', description)
        else:
            return ('Clear', description)  # Default to clear

    def _process_weather_result(self, weather_data: Dict, location: str, coords: Dict, is_forecast: bool, date: str = None) -> Dict[str, Any]:
        """Process weather data into standardized response format."""
        # Get ML category
        ml_category = self.map_weather_to_ml_category(
            weather_data['weather_main'],
            weather_data['weather_description'],
            weather_data['temperature_celsius']
        )

        # Build response
        return {
            'success': True,
            'ml_category': ml_category,
            'weather_data': weather_data,
            'location_info': {
                'requested_location': location,
                'resolved_location': coords['formatted_address'],
                'coordinates': {
                    'latitude': coords['latitude'],
                    'longitude': coords['longitude']
                },
                'coordinate_source': coords['source']
            },
            'mapping_explanation': self._get_mapping_explanation(
                weather_data['weather_main'],
                weather_data['weather_description'],
                weather_data['temperature_celsius'],
                ml_category
            ),
            'query_info': {
                'requested_date': date,
                'is_forecast': is_forecast,
                'data_source': weather_data['data_source']
            }
        }

    def _get_mapping_explanation(self, weather_main: str, description: str, temp_celsius: float, ml_category: str) -> str:
        """Generate explanation for weather mapping."""
        if ml_category == "extreme_heat":
            return f"Temperature {temp_celsius:.1f}°C exceeds extreme heat threshold (35°C)"
        elif ml_category == "rainy":
            return f"Weather condition '{weather_main}' indicates precipitation or poor visibility"
        elif ml_category == "cloudy":
            return f"Weather condition '{weather_main}' indicates overcast skies"
        else:  # sunny
            return f"Weather condition '{weather_main}' indicates clear or mostly clear skies"

    def _get_default_weather_response(self, location: str, date: str, is_forecast: bool, error_reason: str) -> Dict[str, Any]:
        """Generate default weather response when API fails."""
        # Determine seasonal default
        if date:
            try:
                target_date = datetime.strptime(date, '%Y-%m-%d')
                month = target_date.month
            except:
                month = datetime.now().month
        else:
            month = datetime.now().month

        # Southern California seasonal defaults
        if month in [6, 7, 8, 9]:  # Summer
            default_weather = "sunny"
            default_temp = 28.0
            description = "Clear and sunny (summer default)"
        elif month in [12, 1, 2]:  # Winter
            default_weather = "cloudy"
            default_temp = 18.0
            description = "Partly cloudy (winter default)"
        else:  # Spring/Fall
            default_weather = "sunny"
            default_temp = 22.0
            description = "Clear skies (spring/fall default)"

        return {
            'success': True,
            'ml_category': default_weather,
            'weather_data': {
                'temperature_celsius': default_temp,
                'temperature_fahrenheit': round(default_temp * 9/5 + 32, 1),
                'weather_main': 'Clear' if default_weather == 'sunny' else 'Clouds',
                'weather_description': description,
                'data_source': 'seasonal_default',
                'fallback_reason': error_reason
            },
            'location_info': {
                'requested_location': location,
                'resolved_location': f"{location} (default)",
                'coordinates': {'latitude': 34.0551, 'longitude': -117.7500},
                'coordinate_source': 'default'
            },
            'mapping_explanation': f"Using seasonal default for Southern California: {default_weather}",
            'query_info': {
                'requested_date': date,
                'is_forecast': is_forecast,
                'data_source': 'fallback'
            },
            'notice': f'Weather API unavailable ({error_reason}). Using seasonal default.',
            'api_status': f'Failed: {error_reason}'
        }


# # Example usage and testing
# if __name__ == "__main__":
#     # Initialize weather service
#     weather = WeatherService(default_location="Pomona,CA")
#
#     print("🌤️  Weather Service Test")
#     print("=" * 40)
#
#     # Test current weather
#     print("\n1. Current Weather:")
#     current = weather.get_current_weather()
#     print(f"   Success: {current['success']}")
#     if current['success']:
#         print(f"   ML Category: {current['ml_category']}")
#         print(f"   Temperature: {current['weather_data']['temperature_fahrenheit']}°F")
#         print(f"   Description: {current['weather_data']['weather_description']}")
#
#     # Test forecast
#     print("\n2. Tomorrow's Forecast:")
#     forecast = weather.get_forecast()
#     print(f"   Success: {forecast['success']}")
#     if forecast['success']:
#         print(f"   ML Category: {forecast['ml_category']}")
#         print(f"   Temperature: {forecast['weather_data']['temperature_fahrenheit']}°F")
#         print(f"   Description: {forecast['weather_data']['weather_description']}")
#
#     # Test ML prediction format
#     print("\n3. ML Prediction Format:")
#     ml_data = weather.get_weather_for_ml_prediction()
#     print(f"   Success: {ml_data['success']}")
#     if ml_data['success']:
#         print(f"   ML Category: {ml_data['ml_category']}")
#         print(f"   Explanation: {ml_data['mapping_explanation']}")
#
#     print(f"\n4. Available ML Categories: {weather.get_ml_categories()}")
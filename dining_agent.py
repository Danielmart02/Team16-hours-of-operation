import math
import numexpr
import json
import datetime
import sys
import os
import logging
import requests
import warnings  # Add this import
from typing import Literal, List, Dict, Any, Optional

import boto3
import pandas as pd

from weather_forecast import WeatherService

from langchain_core.tools import tool
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from inference import StaffingPredictor

# Suppress XGBoost model compatibility warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# Hide warnings from logger
logging.getLogger("langchain_aws").setLevel(logging.ERROR)

class DiningHallAgent:
    """
    A conversational agent for dining hall staffing predictions and historical data analysis.

    This class encapsulates the functionality for:
    - Predicting staffing requirements
    - Retrieving historical data
    - Managing conversation history
    - Providing staffing insights and recommendations
    - Retrieving and converting weather data for predictions
    """

    def __init__(
            self,
            model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            aws_region: str = None,
            tx_model_path: str = './tx_model.pkl',
            work_model_path: str = './work_model.pkl',
            data_file_path: str = 'df.csv',
            enable_tracing: bool = False,
            recursion_limit: int = 15,
            default_location: str = "Pomona,CA"
    ):
        """
        Initialize the Dining Hall Agent.

        Args:
            model_id: AWS Bedrock model identifier
            aws_region: AWS region for Bedrock client
            tx_model_path: Path to transaction prediction model
            work_model_path: Path to work hours prediction model
            data_file_path: Path to historical data CSV file
            enable_tracing: Whether to enable conversation tracing
            recursion_limit: Maximum recursion limit for the agent
            default_location: Default location for weather queries (format: "City,State")
        """
        self.logger = self._setup_logging()
        self.aws_region = aws_region or os.environ.get("AWS_REGION", "us-west-2")
        self.model_id = model_id
        self.data_file_path = data_file_path
        self.recursion_limit = recursion_limit
        self.default_location = default_location

        # Conversation state
        self.conversation_history: List[Any] = []
        self.system_prompt = self._get_default_system_prompt()

        # Tracing
        self.enable_tracing = enable_tracing
        self.trace_handle = None
        if self.enable_tracing:
            self._setup_tracing()

        # Initialize components
        self._setup_bedrock_client()
        self._setup_predictor(tx_model_path, work_model_path)
        self._setup_agent()

        # Initialize WeatherService
        self.weather_service = WeatherService(
            default_location=default_location,
            user_agent="DiningHallAgent/1.0 (staffing@example.com)"
        )

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
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

    def _setup_bedrock_client(self):
        """Initialize AWS Bedrock client and chat model."""
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=self.aws_region
            )
            self.chat_model = ChatBedrock(
                model_id=self.model_id,
                client=self.bedrock_client
            )
            self.logger.info(f"Bedrock client initialized with model: {self.model_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    def _setup_predictor(self, tx_model_path: str, work_model_path: str):
        """Initialize the staffing predictor with trained models."""
        try:
            self.predictor = StaffingPredictor()
            self.predictor.load_models(tx_model_path, work_model_path)
            self.logger.info("Staffing predictor models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load predictor models: {e}")
            raise

    def _setup_tracing(self):
        """Setup conversation tracing."""
        if self.enable_tracing:
            file_name = f"trace_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            self.trace_handle = open(file_name, 'w')
            self.trace_file_name = file_name

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent."""
        return (
            "You are a helpful dining hall staffing assistant. Answer questions as best you can using the provided tools. "
            "Do not make up answers. Think step by step. "
            f"Today's date is {datetime.datetime.now().strftime('%B %d, %Y')}. "
            "When asked about 'next week', interpret this as the 7 days starting from tomorrow. "
            "Use appropriate weather and event assumptions if not specified (e.g., 'sunny' weather and 'regular_day' events). "
            "Always use the prediction tools to get accurate staffing forecasts. "
            "You can retrieve current weather data to inform predictions. "
            "Provide clear, actionable insights based on the data and predictions."
        )

    def set_system_prompt(self, prompt: str):
        """Update the system prompt for the agent."""
        self.system_prompt = prompt
        self.logger.info("System prompt updated")

    def _output_trace(self, element: str, trace, node: bool = True):
        """Output tracing information if enabled."""
        if self.enable_tracing and self.trace_handle:
            print(datetime.datetime.now(), file=self.trace_handle)
            print(("Node: " if node else "Edge: ") + element, file=self.trace_handle)
            if element == "call_model (entry)":
                for single_trace in trace:
                    print(single_trace, file=self.trace_handle)
            else:
                print(trace, file=self.trace_handle)
            print('----', file=self.trace_handle)

    def _map_weather_to_model_category(self, weather_main: str, weather_description: str, temp_celsius: float) -> str:
        """
        Map National Weather Service weather data to model categories.

        Args:
            weather_main: Main weather category (e.g., "Clear", "Clouds", "Rain", "Snow", "Fog")
            weather_description: Detailed description
            temp_celsius: Temperature in Celsius

        Returns:
            Weather category for ML model: "sunny", "cloudy", "rainy", or "extreme_heat"
        """
        # Check for extreme heat first (over 35°C / 95°F)
        if temp_celsius > 35:
            return "extreme_heat"

        # Map weather conditions
        weather_main = weather_main.lower()
        weather_description = weather_description.lower()

        if weather_main in ["rain", "snow", "fog"]:
            return "rainy"
        elif weather_main in ["clouds"]:
            # Distinguish between partly cloudy (sunny) and overcast (cloudy)
            if any(phrase in weather_description for phrase in ["partly", "few", "scattered", "mostly sunny"]):
                return "sunny"
            else:
                return "cloudy"
        elif weather_main in ["clear"]:
            return "sunny"
        else:
            # Default fallback
            return "sunny"

    def _create_tools(self) -> List:
        """Create and return the list of tools for the agent."""

        @tool
        def history_retrieve(query):
            """
            Retrieves historical dining hall data from local database.

            Args:
                query (dict): Query parameters with exact structure:
                    For single day: {"query_type": "single", "date": "YYYY-MM-DD"}
                    For date range: {"query_type": "range", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}

            Examples:
                Single day: {"query_type": "single", "date": "2024-07-15"}
                Date range: {"query_type": "range", "start_date": "2024-07-01", "end_date": "2024-07-31"}

            Returns:
                Historical transaction and staffing data with success status.

            Use for: Analyzing past performance, comparing predictions to actuals, getting historical trends.
            """
            return self._history_retrieve_impl(query)

        @tool
        def predict_workhours(daterange, weather, event, target_roles=None):
            """
            Predicts staffing hours needed for dining hall operations.
            [Same as before - no changes needed]
            """
            return self._predict_workhours_impl(daterange, weather, event, target_roles)

        @tool
        def get_prediction_options():
            """
            Returns all valid options for weather, events, and staffing roles.
            [Same as before - no changes needed]
            """
            return self._get_prediction_options_impl()

        @tool
        def predict_transactions_only(date, weather, event):
            """
            Predicts customer transaction volume only.
            [Same as before - no changes needed]
            """
            return self._predict_transactions_only_impl(date, weather, event)

        @tool
        def get_weather_for_prediction(location=None, date=None):
            """
            Retrieves current or forecast weather data using National Weather Service API
            and converts it to ML model categories.

            Args:
                location: Optional location string (e.g., "Los Angeles,CA"). Defaults to Pomona, CA.
                date: Optional date string "YYYY-MM-DD" for forecast. If None, gets current weather.

            Returns: Weather data with ML model category, raw weather info, and location details.
            Use for: Getting real weather data for accurate staffing predictions, converting weather to model inputs.
            """
            return self._get_weather_for_prediction_impl(location, date)

        @tool
        def get_current_weather(location=None):
            """
            Get current weather conditions and ML category for a location.

            Args:
                location: Optional location string. Defaults to Pomona, CA.

            Returns: Current weather with ML model category.
            Use for: Real-time weather conditions for immediate staffing decisions.
            """
            return self._get_current_weather_impl(location)

        @tool
        def get_weather_forecast(location=None, date=None):
            """
            Get weather forecast and ML category for a specific date.

            Args:
                location: Optional location string. Defaults to Pomona, CA.
                date: Optional date string "YYYY-MM-DD". Defaults to tomorrow.

            Returns: Weather forecast with ML model category.
            Use for: Planning future staffing based on forecasted conditions.
            """
            return self._get_weather_forecast_impl(location, date)

        return [
            history_retrieve,
            predict_workhours,
            get_prediction_options,
            predict_transactions_only,
            get_weather_for_prediction,
            get_current_weather,
            get_weather_forecast
        ]

    def _get_weather_for_prediction_impl(self, location=None, date=None) -> Dict[str, Any]:
        """Implementation using WeatherService - keeps same interface for backward compatibility."""
        try:
            result = self.weather_service.get_weather_for_ml_prediction(location, date)

            if result['success']:
                # Format response to match original interface
                return {
                    'success': True,
                    'model_weather_category': result['ml_category'],
                    'raw_weather_data': {
                        'location': result['location'],
                        'temperature_celsius': result['temperature_celsius'],
                        'temperature_fahrenheit': result['temperature_fahrenheit'],
                        'weather_main': result['ml_category'].title(),
                        'weather_description': result['description'],
                        'data_source': result['data_source'],
                        'timestamp': result['timestamp']
                    },
                    'mapping_explanation': result['mapping_explanation'],
                    'query_info': {
                        'requested_location': location,
                        'requested_date': date,
                        'is_forecast': result['is_forecast']
                    }
                }
            else:
                return result

        except Exception as e:
            self.logger.error(f"Error in weather prediction: {e}")
            return {
                'success': False,
                'error': f'Weather retrieval failed: {str(e)}'
            }

    def _get_current_weather_impl(self, location=None) -> Dict[str, Any]:
        """Get current weather using WeatherService."""
        try:
            result = self.weather_service.get_current_weather(location)

            if result['success']:
                return {
                    'success': True,
                    'ml_category': result['ml_category'],
                    'weather_data': result['weather_data'],
                    'location_info': result['location_info'],
                    'mapping_explanation': result['mapping_explanation']
                }
            else:
                return result

        except Exception as e:
            self.logger.error(f"Error getting current weather: {e}")
            return {
                'success': False,
                'error': f'Current weather retrieval failed: {str(e)}'
            }

    def _get_weather_forecast_impl(self, location=None, date=None) -> Dict[str, Any]:
        """Get weather forecast using WeatherService."""
        try:
            result = self.weather_service.get_forecast(location, date)

            if result['success']:
                return {
                    'success': True,
                    'ml_category': result['ml_category'],
                    'weather_data': result['weather_data'],
                    'location_info': result['location_info'],
                    'mapping_explanation': result['mapping_explanation']
                }
            else:
                return result

        except Exception as e:
            self.logger.error(f"Error getting weather forecast: {e}")
            return {
                'success': False,
                'error': f'Weather forecast retrieval failed: {str(e)}'
            }


    def _get_coordinates_for_location(self, location: str) -> Dict[str, Any]:
        """Get coordinates for a location string using geocoding."""
        try:
            # Predefined coordinates for common California locations
            location_coords = {
                "pomona,ca": (34.0551, -117.7500),
                "los angeles,ca": (34.0522, -118.2437),
                "san francisco,ca": (37.7749, -122.4194),
                "san diego,ca": (32.7157, -117.1611),
                "sacramento,ca": (38.5816, -121.4944),
                "fresno,ca": (36.7378, -119.7871),
                "claremont,ca": (34.0967, -117.7198),
                "riverside,ca": (33.9533, -117.3962),
            }

            location_key = location.lower().strip()
            if location_key in location_coords:
                lat, lon = location_coords[location_key]
                return {
                    'success': True,
                    'latitude': lat,
                    'longitude': lon,
                    'formatted_address': location
                }

            # If not in predefined list, try US Census Geocoding API (free, no key required)
            geocode_url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
            params = {
                'address': location,
                'benchmark': 'Public_AR_Current',
                'format': 'json'
            }

            response = requests.get(geocode_url, params=params, timeout=10)
            if response.status_code == 200:
                geocode_data = response.json()
                if geocode_data.get('result', {}).get('addressMatches'):
                    match = geocode_data['result']['addressMatches'][0]
                    coordinates = match['coordinates']
                    return {
                        'success': True,
                        'latitude': float(coordinates['y']),
                        'longitude': float(coordinates['x']),
                        'formatted_address': match['matchedAddress']
                    }

            # Fallback to Pomona coordinates if geocoding fails
            self.logger.warning(f"Could not geocode location '{location}', using Pomona, CA coordinates")
            return {
                'success': True,
                'latitude': 34.0551,
                'longitude': -117.7500,
                'formatted_address': "Pomona, CA (fallback)"
            }

        except Exception as e:
            self.logger.error(f"Geocoding error: {e}")
            return {
                'success': False,
                'error': f'Failed to determine coordinates for location: {str(e)}'
            }

    def _get_nws_current_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get current weather from NWS API."""
        try:
            # NWS requires a User-Agent header
            headers = {
                'User-Agent': 'DiningHallAgent/1.0 (https://example.com/contact)'
            }

            # Get the nearest weather station
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
                    'error': 'No weather stations found for this location'
                }

            # Get the first (nearest) station
            station_id = stations_data['features'][0]['properties']['stationIdentifier']

            # Get current observations from the station
            obs_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
            response = requests.get(obs_url, headers=headers, timeout=10)

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'Failed to get current weather observations: HTTP {response.status_code}'
                }

            obs_data = response.json()
            properties = obs_data['properties']

            # Extract temperature (convert from Celsius if needed)
            temp_celsius = properties.get('temperature', {}).get('value')
            if temp_celsius is None:
                return {
                    'success': False,
                    'error': 'Temperature data not available from weather station'
                }

            # Extract weather conditions
            text_description = properties.get('textDescription', 'Unknown')

            # Map NWS conditions to our simplified categories
            weather_main, weather_description = self._parse_nws_conditions(text_description)

            weather_data = {
                'location': f"{lat:.4f}, {lon:.4f}",
                'station_id': station_id,
                'observation_time': properties.get('timestamp', ''),
                'temperature_celsius': round(temp_celsius, 1),
                'temperature_fahrenheit': round(temp_celsius * 9/5 + 32, 1),
                'weather_main': weather_main,
                'weather_description': weather_description,
                'humidity': properties.get('relativeHumidity', {}).get('value'),
                'wind_speed': properties.get('windSpeed', {}).get('value'),
                'data_source': 'nws_current'
            }

            return {
                'success': True,
                'data': weather_data
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error fetching current weather from NWS: {str(e)}'
            }

    def _get_nws_forecast(self, lat: float, lon: float, target_date: datetime.datetime) -> Dict[str, Any]:
        """Get forecast weather from NWS API."""
        try:
            headers = {
                'User-Agent': 'DiningHallAgent/1.0 (https://example.com/contact)'
            }

            # Get grid information for this location
            points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
            response = requests.get(points_url, headers=headers, timeout=10)

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'Failed to get forecast grid info: HTTP {response.status_code}'
                }

            points_data = response.json()
            properties = points_data['properties']

            # Get the forecast
            forecast_url = properties['forecast']
            response = requests.get(forecast_url, headers=headers, timeout=10)

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'Failed to get forecast data: HTTP {response.status_code}'
                }

            forecast_data = response.json()

            # Find the forecast period closest to target date
            target_date_str = target_date.strftime('%Y-%m-%d')
            best_period = None

            for period in forecast_data['properties']['periods']:
                period_start = datetime.datetime.fromisoformat(period['startTime'].replace('Z', '+00:00'))
                period_date = period_start.strftime('%Y-%m-%d')

                if period_date == target_date_str:
                    best_period = period
                    break

            if not best_period:
                # If exact date not found, use the first available period
                best_period = forecast_data['properties']['periods'][0] if forecast_data['properties']['periods'] else None

            if not best_period:
                return {
                    'success': False,
                    'error': f'No forecast data available for {target_date_str}'
                }

            # Extract forecast information
            temp_fahrenheit = best_period.get('temperature', 70)  # Default to 70F if missing
            temp_celsius = round((temp_fahrenheit - 32) * 5/9, 1)

            # Parse weather conditions from forecast text
            forecast_text = best_period.get('detailedForecast', best_period.get('shortForecast', ''))
            weather_main, weather_description = self._parse_nws_conditions(forecast_text)

            weather_data = {
                'location': f"{lat:.4f}, {lon:.4f}",
                'forecast_period': best_period.get('name', 'Unknown'),
                'forecast_time': best_period.get('startTime', ''),
                'temperature_celsius': temp_celsius,
                'temperature_fahrenheit': temp_fahrenheit,
                'weather_main': weather_main,
                'weather_description': weather_description,
                'detailed_forecast': forecast_text,
                'data_source': 'nws_forecast'
            }

            return {
                'success': True,
                'data': weather_data
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error fetching forecast from NWS: {str(e)}'
            }

    def _parse_nws_conditions(self, description: str) -> tuple:
        """Parse NWS weather description and return simplified categories."""
        description_lower = description.lower()

        # Determine main weather category based on keywords
        if any(word in description_lower for word in ['rain', 'shower', 'drizzle', 'thunderstorm', 'storm']):
            return ('Rain', description)
        elif any(word in description_lower for word in ['snow', 'sleet', 'ice', 'freezing']):
            return ('Snow', description)
        elif any(word in description_lower for word in ['fog', 'mist', 'haze']):
            return ('Fog', description)
        elif any(word in description_lower for word in ['cloud', 'overcast', 'partly']):
            if any(word in description_lower for word in ['partly cloudy', 'mostly sunny', 'few clouds']):
                return ('Clear', description)
            else:
                return ('Clouds', description)
        elif any(word in description_lower for word in ['clear', 'sunny', 'fair']):
            return ('Clear', description)
        else:
            # Default to clear if we can't categorize
            return ('Clear', description)

    def _get_weather_mapping_explanation(self, weather_main: str, weather_description: str, temp_celsius: float, model_category: str) -> str:
        """Provide explanation of how weather was mapped to model category."""
        explanations = {
            "extreme_heat": f"Temperature is {temp_celsius:.1f}°C (>{35}°C threshold)",
            "rainy": f"Weather condition '{weather_main}' ({weather_description}) indicates precipitation or poor visibility",
            "cloudy": f"Weather condition '{weather_main}' ({weather_description}) indicates overcast skies",
            "sunny": f"Weather condition '{weather_main}' ({weather_description}) indicates clear or mostly clear skies"
        }
        return explanations.get(model_category, f"Weather '{weather_main}' mapped to {model_category}")

    def _history_retrieve_impl(self, query) -> Dict[str, Any]:
        """Implementation of history_retrieve tool with improved input handling."""
        try:
            # Handle both string and dictionary inputs
            if isinstance(query, str):
                # If it's a string, try to parse it as JSON
                try:
                    query = json.loads(query)
                except json.JSONDecodeError:
                    # If it's not valid JSON, assume it's a date string for single query
                    return {
                        'statusCode': 400,
                        'body': json.dumps({
                            'success': False,
                            'error': f'Invalid query format. Expected dictionary with query_type, received string: {query}',
                            'help': 'Use format: {"query_type": "single", "date": "YYYY-MM-DD"} or {"query_type": "range", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}'
                        })
                    }

            # Ensure query is a dictionary
            if not isinstance(query, dict):
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'success': False,
                        'error': f'Query must be a dictionary, received {type(query).__name__}',
                        'help': 'Use format: {"query_type": "single", "date": "YYYY-MM-DD"} or {"query_type": "range", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}'
                    })
                }

            # Input validation
            query_type = query.get('query_type')
            if not query_type or query_type not in ['single', 'range']:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'success': False,
                        'error': 'Invalid or missing query_type. Must be "single" or "range"',
                        'received_query': query
                    })
                }

            # Validate date fields
            if query_type == 'single':
                if 'date' not in query:
                    return {
                        'statusCode': 400,
                        'body': json.dumps({
                            'success': False,
                            'error': 'Missing date field for single query',
                            'expected_format': '{"query_type": "single", "date": "YYYY-MM-DD"}'
                        })
                    }
                try:
                    target_date = pd.to_datetime(query['date']).date()
                except Exception as date_error:
                    return {
                        'statusCode': 400,
                        'body': json.dumps({
                            'success': False,
                            'error': f'Invalid date format: {query["date"]}. Use YYYY-MM-DD format',
                            'date_error': str(date_error)
                        })
                    }

            else:  # range query
                if 'start_date' not in query or 'end_date' not in query:
                    return {
                        'statusCode': 400,
                        'body': json.dumps({
                            'success': False,
                            'error': 'Missing start_date or end_date for range query',
                            'expected_format': '{"query_type": "range", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}'
                        })
                    }
                try:
                    start_date = pd.to_datetime(query['start_date']).date()
                    end_date = pd.to_datetime(query['end_date']).date()
                except Exception as date_error:
                    return {
                        'statusCode': 400,
                        'body': json.dumps({
                            'success': False,
                            'error': f'Invalid date format in range query. Use YYYY-MM-DD format',
                            'date_error': str(date_error)
                        })
                    }

                if start_date > end_date:
                    return {
                        'statusCode': 400,
                        'body': json.dumps({
                            'success': False,
                            'error': 'start_date cannot be after end_date'
                        })
                    }

            # Load and parse data file
            self.logger.info("Loading data from file")
            try:
                df = pd.read_csv(self.data_file_path, on_bad_lines='skip', index_col=0)
            except Exception as file_error:
                self.logger.error(f"Error loading data file: {file_error}")
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'success': False,
                        'error': f'Failed to load data file: {str(file_error)}',
                        'file_path': self.data_file_path
                    })
                }

            # Convert date column to datetime with error handling
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as date_error:
                self.logger.error(f"Error parsing dates: {date_error}")
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'success': False,
                        'error': f'Date parsing failed: {str(date_error)}'
                    })
                }

            # Filter data based on query type
            if query_type == 'single':
                filtered_df = df[df['date'].dt.date == target_date]
                self.logger.info(f"Retrieved {len(filtered_df)} records for date: {target_date}")
            else:
                filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
                self.logger.info(f"Retrieved {len(filtered_df)} records for range: {start_date} to {end_date}")

            # Return results
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': True,
                    'query_type': query_type,
                    'record_count': len(filtered_df),
                    'data': filtered_df.to_dict('records')
                }, default=str)
            }

        except Exception as e:
            self.logger.error(f"Error in history_retrieve: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'success': False,
                    'error': str(e),
                    'query_received': str(query) if 'query' in locals() else 'Unknown'
                })
            }

    def _predict_workhours_impl(self, daterange, weather, event, target_roles=None) -> Dict[str, Any]:
        """Implementation of predict_workhours tool."""
        try:
            # Validate that predictor is loaded
            if self.predictor.tx_estimator is None or self.predictor.work_estimator is None:
                return {
                    'success': False,
                    'error': 'Prediction models not loaded. Please check model files.'
                }

            # Validate weather and event options
            valid_weather = self.predictor.get_available_weather_conditions()
            valid_events = self.predictor.get_available_events()

            # Parse daterange input
            if isinstance(daterange, str):
                # Single date prediction
                try:
                    target_date = datetime.datetime.strptime(daterange, '%Y-%m-%d')
                    dates = [target_date]
                    is_single_date = True
                except ValueError:
                    return {
                        'success': False,
                        'error': f'Invalid date format. Use YYYY-MM-DD. Received: {daterange}'
                    }

            elif isinstance(daterange, dict):
                # Date range prediction
                if 'start_date' not in daterange or 'end_date' not in daterange:
                    return {
                        'success': False,
                        'error': 'Date range must include both start_date and end_date'
                    }

                try:
                    start_date = datetime.datetime.strptime(daterange['start_date'], '%Y-%m-%d')
                    end_date = datetime.datetime.strptime(daterange['end_date'], '%Y-%m-%d')

                    if start_date > end_date:
                        return {
                            'success': False,
                            'error': 'start_date cannot be after end_date'
                        }

                    # Generate date range
                    dates = []
                    current_date = start_date
                    while current_date <= end_date:
                        dates.append(current_date)
                        current_date += datetime.timedelta(days=1)

                    is_single_date = False

                except ValueError as e:
                    return {
                        'success': False,
                        'error': f'Invalid date format in range. Use YYYY-MM-DD. Error: {str(e)}'
                    }
            else:
                return {
                    'success': False,
                    'error': 'daterange must be either a date string or dict with start_date/end_date'
                }

            # Prepare weather and event lists
            if isinstance(weather, str):
                weather_list = [weather] * len(dates)
            elif isinstance(weather, list):
                if len(weather) != len(dates):
                    return {
                        'success': False,
                        'error': f'Weather list length ({len(weather)}) must match number of dates ({len(dates)})'
                    }
                weather_list = weather
            else:
                return {
                    'success': False,
                    'error': 'Weather must be a string or list of strings'
                }

            if isinstance(event, str):
                event_list = [event] * len(dates)
            elif isinstance(event, list):
                if len(event) != len(dates):
                    return {
                        'success': False,
                        'error': f'Event list length ({len(event)}) must match number of dates ({len(dates)})'
                    }
                event_list = event
            else:
                return {
                    'success': False,
                    'error': 'Event must be a string or list of strings'
                }

            # Validate weather conditions
            invalid_weather = [w for w in weather_list if w not in valid_weather]
            if invalid_weather:
                return {
                    'success': False,
                    'error': f'Invalid weather conditions: {invalid_weather}. Valid options: {valid_weather}'
                }

            # Validate events
            invalid_events = [e for e in event_list if e not in valid_events]
            if invalid_events:
                return {
                    'success': False,
                    'error': f'Invalid events: {invalid_events}. Valid options: {valid_events}'
                }

            # Make predictions
            self.logger.info(f"Making predictions for {len(dates)} dates")

            if is_single_date:
                # Single date prediction
                prediction = self.predictor.predict_staffing_requirements(
                    date=dates[0],
                    weather=weather_list[0],
                    event=event_list[0],
                    target_features=target_roles
                )

                result = {
                    'success': True,
                    'prediction_type': 'single_date',
                    'date': dates[0].strftime('%Y-%m-%d'),
                    'weather': weather_list[0],
                    'event': event_list[0],
                    'predictions': prediction
                }

            else:
                # Batch prediction
                batch_results = self.predictor.batch_predict(
                    dates=dates,
                    weather_conditions=weather_list,
                    events=event_list,
                    target_features=target_roles
                )

                result = {
                    'success': True,
                    'prediction_type': 'date_range',
                    'date_range': {
                        'start_date': dates[0].strftime('%Y-%m-%d'),
                        'end_date': dates[-1].strftime('%Y-%m-%d'),
                        'total_days': len(dates)
                    },
                    'predictions': batch_results.to_dict('records'),
                    'summary': {
                        'total_predicted_hours': batch_results['total_predicted_hours'].sum(),
                        'total_predicted_transactions': batch_results['predicted_transactions'].sum(),
                        'average_daily_hours': batch_results['total_predicted_hours'].mean(),
                        'average_daily_transactions': batch_results['predicted_transactions'].mean()
                    }
                }

            self.logger.info(f"Successfully generated predictions for {len(dates)} dates")
            return result

        except Exception as e:
            self.logger.error(f"Error in predict_workhours: {str(e)}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }

    def _get_prediction_options_impl(self) -> Dict[str, Any]:
        """Implementation of get_prediction_options tool."""
        try:
            return {
                'success': True,
                'weather_conditions': self.predictor.get_available_weather_conditions(),
                'events': self.predictor.get_available_events(),
                'default_staffing_roles': self.predictor.get_default_staffing_roles(),
                'weather_impact_map': self.predictor.WEATHER_IMPACT_MAP,
                'event_impact_map': self.predictor.EVENT_IMPACT_MAP
            }
        except Exception as e:
            self.logger.error(f"Error getting prediction options: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _predict_transactions_only_impl(self, date, weather, event) -> Dict[str, Any]:
        """Implementation of predict_transactions_only tool."""
        try:
            # Validate inputs
            try:
                target_date = datetime.datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                return {
                    'success': False,
                    'error': f'Invalid date format. Use YYYY-MM-DD. Received: {date}'
                }

            valid_weather = self.predictor.get_available_weather_conditions()
            valid_events = self.predictor.get_available_events()

            if weather not in valid_weather:
                return {
                    'success': False,
                    'error': f'Invalid weather: {weather}. Valid options: {valid_weather}'
                }

            if event not in valid_events:
                return {
                    'success': False,
                    'error': f'Invalid event: {event}. Valid options: {valid_events}'
                }

            # Make prediction
            predicted_transactions = self.predictor.predict_transactions(target_date, weather, event)

            # Get enrollment info for context
            enrollment_features = self.predictor.calculate_enrollment_features(target_date)

            return {
                'success': True,
                'date': date,
                'weather': weather,
                'event': event,
                'predicted_transactions': predicted_transactions,
                'context': {
                    'day_of_week': target_date.strftime('%A'),
                    'seasonal_multiplier': enrollment_features['seasonal_multiplier'],
                    'active_enrollment': enrollment_features['active_enrollment'],
                    'total_meal_plan_holders': enrollment_features['total_meal_plan_holders']
                }
            }

        except Exception as e:
            self.logger.error(f"Error in predict_transactions_only: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _call_model(self, state: MessagesState):
        """Agent node that calls the model and handles tool calls."""
        self._output_trace("call_model (entry)", state["messages"])

        messages = state["messages"]
        response = self.model_with_tools.invoke(messages)

        self._output_trace("call_model (response)", response)
        return {"messages": [response]}

    def _should_continue(self, state: MessagesState) -> Literal["tools", "__end__"]:
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        self._output_trace(
            "should_continue",
            f"Last message type: {type(last_message)}, has tool_calls: {hasattr(last_message, 'tool_calls') and last_message.tool_calls}"
        )

        # If the LLM makes a tool call, then we route to the "tools" node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # Otherwise, we stop (reply to the user)
        return "__end__"

    def _setup_agent(self):
        """Setup the agent workflow graph."""
        # Create tools and bind to model
        tools = self._create_tools()
        tool_node = ToolNode(tools)
        self.model_with_tools = self.chat_model.bind_tools(tools)

        # Create the graph
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", tool_node)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges("agent", self._should_continue)

        # Add normal edge from tools back to agent
        workflow.add_edge("tools", "agent")

        # Compile the graph
        self.react_agent = workflow.compile()

        self.logger.info("Agent workflow setup completed")

    def ask(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Ask the agent a question and get a response.

        Args:
            user_prompt: The user's question or request
            system_prompt: Optional system prompt to override default

        Returns:
            The agent's response as a string
        """
        try:
            # Use provided system prompt or default
            current_system_prompt = system_prompt or self.system_prompt

            # Create messages for this interaction
            messages = [SystemMessage(content=current_system_prompt)]

            # Add conversation history
            messages.extend(self.conversation_history)

            # Add current user message
            messages.append(HumanMessage(content=user_prompt))

            # Prepare inputs
            inputs = {"messages": messages}
            config = {"recursion_limit": self.recursion_limit}

            # Invoke the agent
            result = self.react_agent.invoke(inputs, config)

            # Extract response
            if result["messages"]:
                final_message = result["messages"][-1]
                response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)

                # Update conversation history
                self.conversation_history.append(HumanMessage(content=user_prompt))
                self.conversation_history.append(final_message)

                self.logger.info(f"Successfully processed user query: {user_prompt[:50]}...")
                return response_content
            else:
                return "I apologize, but I couldn't generate a response. Please try again."

        except Exception as e:
            self.logger.error(f"Error processing user query: {e}")
            return f"An error occurred while processing your request: {str(e)}"

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")

    def get_history(self) -> List[Any]:
        """Get the current conversation history."""
        return self.conversation_history.copy()

    def save_conversation(self, filename: str):
        """Save conversation history to a file."""
        try:
            with open(filename, 'w') as f:
                for message in self.conversation_history:
                    f.write(f"{type(message).__name__}: {message.content}\n")
            self.logger.info(f"Conversation saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving conversation: {e}")

    def close(self):
        """Clean up resources."""
        if self.enable_tracing and self.trace_handle:
            self.trace_handle.close()
            self.logger.info(f"Tracing saved to: {self.trace_file_name}")


# # Example usage
# if __name__ == "__main__":
#     # Initialize the agent (no API key needed for National Weather Service)
#     agent = DiningHallAgent(enable_tracing=True)
#
#     # Example conversation with weather integration
#     questions = [
#         "What's the current weather in Pomona and what ML category does it map to?",
#         "Get weather for tomorrow and predict staffing hours using that weather data",
#         "What is the predicted staffing work hours for the next week using current weather?",
#         "Can you show me the available weather conditions and events?",
#         "What were the transaction volumes last month?"
#     ]
#
#     for question in questions:
#         print(f"\n🤖 Question: {question}")
#         response = agent.ask(question)
#         print(f"📊 Response: {response}")
#         print("=" * 80)
#
#     # Clean up
#     agent.close()
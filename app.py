from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import json
import numpy as np

app = Flask(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Try to initialize the predictor with your models
predictor = None
models_loaded = False
try:
    from inference import StaffingPredictor
    predictor = StaffingPredictor()
    predictor.load_models('tx_model.pkl', 'work_model.pkl')
    models_loaded = True
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"⚠️  Warning: Could not load models - {e}")
    print("Running in demo mode with mock data")
    predictor = None
    models_loaded = False

# Worker type mapping for display
WORKER_DISPLAY_NAMES = {
    'actual_foh_general': 'General Purpose Worker',
    'actual_foh_cashier': 'Cashier',
    'actual_kitchen_prep': 'Chef',
    'actual_kitchen_line': 'Line Workers',
    'actual_dish_room': 'Dishwasher',
    'actual_management': 'Management'
}

# Mock data for demo mode
MOCK_WEATHER_OPTIONS = ['sunny', 'cloudy', 'rainy', 'extreme_heat']
MOCK_EVENT_OPTIONS = ['regular_day', 'club_fair', 'career_fair', 'sports_events', 'graduation', 
                     'parent_weekend', 'prospective_student_day', 'conference_hosting', 'campus_construction']

def get_mock_prediction(date, weather, event):
    """Generate mock prediction data for demo purposes"""
    import random
    random.seed(hash(f"{date}{weather}{event}"))  # Consistent results for same inputs
    
    base_multiplier = {
        'sunny': 1.0,
        'cloudy': 1.1,
        'rainy': 1.3,
        'extreme_heat': 0.9
    }.get(weather, 1.0)
    
    event_multiplier = {
        'regular_day': 1.0,
        'club_fair': 1.4,
        'career_fair': 1.2,
        'sports_events': 1.1,
        'graduation': 1.5,
        'parent_weekend': 1.3,
        'prospective_student_day': 1.2,
        'conference_hosting': 1.1,
        'campus_construction': 0.9
    }.get(event, 1.0)
    
    total_multiplier = base_multiplier * event_multiplier
    
    # Base hours for each worker type
    base_hours = {
        'actual_foh_general': 24.0,
        'actual_foh_cashier': 16.0,
        'actual_kitchen_prep': 20.0,
        'actual_kitchen_line': 18.0,
        'actual_dish_room': 12.0,
        'actual_management': 8.0
    }
    
    prediction = {}
    total_hours = 0
    
    for role, hours in base_hours.items():
        adjusted_hours = hours * total_multiplier * (0.8 + random.random() * 0.4)  # ±20% variation
        prediction[role] = round(adjusted_hours, 1)
        total_hours += adjusted_hours
    
    prediction['total_predicted_hours'] = round(total_hours, 1)
    prediction['predicted_transactions'] = int(450 * total_multiplier * (0.8 + random.random() * 0.4))
    
    return prediction

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/weather-options')
def get_weather_options():
    """Get available weather conditions"""
    if models_loaded and predictor:
        return jsonify(predictor.get_available_weather_conditions())
    else:
        return jsonify(MOCK_WEATHER_OPTIONS)

@app.route('/api/event-options')
def get_event_options():
    """Get available campus events"""
    if models_loaded and predictor:
        return jsonify(predictor.get_available_events())
    else:
        return jsonify(MOCK_EVENT_OPTIONS)

@app.route('/api/predict', methods=['POST'])
def predict_staffing():
    """API endpoint for staffing predictions"""
    try:
        data = request.get_json()
        
        # Parse input data
        date_str = data.get('date')
        weather = data.get('weather')
        event = data.get('event')
        
        # Validate inputs
        if not all([date_str, weather, event]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Parse date
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        # Get prediction
        if models_loaded and predictor:
            prediction = predictor.predict_staffing_requirements(date, weather, event)
            prediction = convert_numpy_types(prediction)
        else:
            prediction = get_mock_prediction(date_str, weather, event)
        
        # Format response with display names
        formatted_prediction = {}
        worker_predictions = {}
        
        for key, value in prediction.items():
            if key.startswith('actual_'):
                display_name = WORKER_DISPLAY_NAMES.get(key, key)
                worker_predictions[display_name] = value
            else:
                formatted_prediction[key] = value
        
        formatted_prediction['workers'] = worker_predictions
        
        return jsonify(formatted_prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict_staffing():
    """API endpoint for batch predictions over a date range"""
    try:
        data = request.get_json()
        
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')
        weather = data.get('weather')
        event = data.get('event')
        
        # Validate inputs
        if not all([start_date_str, end_date_str, weather, event]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Parse dates
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        # Generate date range
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Create weather and event lists
        weather_conditions = [weather] * len(dates)
        events = [event] * len(dates)
        
        # Get batch predictions
        if models_loaded and predictor:
            batch_predictions = predictor.batch_predict(dates, weather_conditions, events)
            # Format response
            formatted_predictions = []
            for _, row in batch_predictions.iterrows():
                row_dict = convert_numpy_types(row.to_dict())
                formatted_row = {
                    'date': row_dict['date'],
                    'weather': row_dict['weather'],
                    'event': row_dict['event'],
                    'predicted_transactions': row_dict.get('predicted_transactions', 0),
                    'total_predicted_hours': row_dict.get('total_predicted_hours', 0),
                    'workers': {}
                }
                
                # Add worker predictions with display names
                for key, value in row_dict.items():
                    if key.startswith('actual_'):
                        display_name = WORKER_DISPLAY_NAMES.get(key, key)
                        formatted_row['workers'][display_name] = value
                
                formatted_predictions.append(formatted_row)
        else:
            # Use mock data
            formatted_predictions = []
            for i, date in enumerate(dates):
                prediction = get_mock_prediction(date.strftime('%Y-%m-%d'), weather_conditions[i], events[i])
                formatted_row = {
                    'date': date.strftime('%Y-%m-%d'),
                    'weather': weather_conditions[i],
                    'event': events[i],
                    'predicted_transactions': prediction.get('predicted_transactions', 0),
                    'total_predicted_hours': prediction.get('total_predicted_hours', 0),
                    'workers': {}
                }
                
                # Add worker predictions with display names
                for key, value in prediction.items():
                    if key.startswith('actual_'):
                        display_name = WORKER_DISPLAY_NAMES.get(key, key)
                        formatted_row['workers'][display_name] = value
                
                formatted_predictions.append(formatted_row)
        
        return jsonify(formatted_predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/today-summary')
def get_today_summary():
    """Get quick summary for today with default conditions"""
    try:
        today = datetime.now()
        
        # Use default conditions for quick summary
        weather = 'sunny'
        event = 'regular_day'
        
        if models_loaded and predictor:
            prediction = predictor.predict_staffing_requirements(today, weather, event)
            prediction = convert_numpy_types(prediction)
        else:
            prediction = get_mock_prediction(today.strftime('%Y-%m-%d'), weather, event)
        
        # Calculate total workers needed (assuming 8-hour shifts)
        total_workers = 0
        for key, hours in prediction.items():
            if key.startswith('actual_'):
                workers_needed = max(1, round(hours / 8))  # At least 1 worker per role
                total_workers += workers_needed
        
        summary = {
            'date': today.strftime('%Y-%m-%d'),
            'total_workers_needed': total_workers,
            'people_expected': prediction.get('predicted_transactions', 0),
            'total_hours': prediction.get('total_predicted_hours', 0)
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

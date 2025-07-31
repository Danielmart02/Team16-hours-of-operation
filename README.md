# Cal Poly Pomona Dining Dashboard

A web-based dashboard for predicting staffing requirements at Cal Poly Pomona's dining facilities based on weather conditions, campus events, and dates.

## Features

- **Real-time Predictions**: Get staffing predictions based on weather, events, and dates
- **Interactive Dashboard**: Visual representation of staffing needs with stacked bar charts
- **Quick Information**: Today's summary with worker count, expected customers, and total hours
- **Batch Predictions**: Generate predictions for date ranges
- **Responsive Design**: Works on desktop and mobile devices

## Worker Types Predicted

- **General Purpose Worker** (FOH_General)
- **Cashier** (FOH_Cashier)
- **Chef** (Kitchen_Prep)
- **Line Workers** (Kitchen_Line)
- **Dishwasher** (Dish_Room)
- **Management** (Management)

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model Files are Present**:
   - `tx_model.pkl` - Transaction prediction model
   - `work_model.pkl` - Staffing prediction model
   - `inference.py` - Prediction logic

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Dashboard**:
   Open your browser and go to `http://localhost:5000`

## API Endpoints

- `GET /` - Main dashboard page
- `GET /api/weather-options` - Get available weather conditions
- `GET /api/event-options` - Get available campus events
- `POST /api/predict` - Single day prediction
- `POST /api/batch-predict` - Date range predictions
- `GET /api/today-summary` - Quick summary for today

## Weather Conditions

- Sunny
- Cloudy
- Rainy
- Extreme Heat

## Campus Events

- Regular Day
- Club Fair
- Career Fair
- Sports Events
- Graduation
- Parent Weekend
- Prospective Student Day
- Conference Hosting
- Campus Construction

## Usage

1. **Quick Information**: View today's predictions automatically loaded on the dashboard
2. **Custom Predictions**: 
   - Select start and end dates
   - Choose weather condition
   - Select campus event type
   - Click "Generate Predictions"
3. **View Results**: Interactive chart shows stacked hours by worker type
4. **Hover for Details**: Hover over chart bars to see detailed information

## File Structure

```
├── app.py                 # Flask application
├── inference.py           # Prediction logic
├── dataset_generator.py   # Data generation utilities
├── tx_model.pkl          # Transaction prediction model
├── work_model.pkl        # Staffing prediction model
├── requirements.txt      # Python dependencies
├── templates/
│   └── dashboard.html    # Main dashboard template
├── static/
│   ├── css/
│   │   └── style.css     # Styling with Cal Poly colors
│   ├── js/
│   │   └── dashboard.js  # Dashboard functionality
│   └── images/
│       └── cpp_logo.svg  # Cal Poly Pomona logo
└── README.md            # This file
```

## Color Scheme

The dashboard uses Cal Poly Pomona's official colors:
- **Green**: #1B5E20 (Primary)
- **Light Green**: #4CAF50 (Accent)
- **Yellow**: #FFC107 (Highlight)
- **White**: #FFFFFF (Background)

## Technologies Used

- **Backend**: Flask, Python
- **Frontend**: HTML5, CSS3, JavaScript
- **Charts**: Chart.js
- **Machine Learning**: XGBoost, scikit-learn, pandas
- **Styling**: CSS Grid, Flexbox, CSS Variables

import pandas as pd
import pickle
import datetime
from typing import Dict, List, Optional
from xgboost import XGBRegressor


class StaffingPredictor:
    """
    A class for predicting dining hall staffing requirements based on various factors.

    This predictor uses a two-stage approach:
    1. Predict total transactions based on date, weather, and events
    2. Predict staffing hours for each role based on predicted transactions and other features
    """

    # Constants for current operational parameters (2024-2025 baseline)
    OPERATIONAL_CONSTANTS = {
        # Student Population (Base values for 2024)
        'BASE_TOTAL_ENROLLMENT': 31000,
        'YOY_GROWTH_RATE': 0.022,  # 2.2% annual growth
        'RESIDENTIAL_STUDENT_RATIO': 0.152,  # ~15.2% live on campus

        # Meal plan participation rates
        'RESIDENTIAL_MANDATORY_RATE': 1.0,  # 100% of residential required
        'COMMUTER_VOLUNTARY_RATE': 0.078,   # 7.8% of commuters buy plans
    }

    # Weather impact mapping
    WEATHER_IMPACT_MAP = {
        'sunny': 1.0,
        'cloudy': 1.023,
        'rainy': 1.147,
        'extreme_heat': 0.891,
    }

    # Campus event impact mapping
    EVENT_IMPACT_MAP = {
        'regular_day': 1.0,
        'club_fair': 1.34,
        'career_fair': 1.23,
        'sports_events': 1.12,
        'graduation': 1.43,
        'parent_weekend': 1.38,
        'prospective_student_day': 1.19,
        'conference_hosting': 1.16,
        'campus_construction': 0.94,
    }

    # Special academic periods with exact dates
    SPECIAL_PERIODS = [
        {'name': 'move_in_week', 'dates': [(8, 15, 8, 22)], 'multiplier': 1.28},
        {'name': 'finals_weeks', 'dates': [(12, 8, 12, 15), (5, 8, 5, 15)], 'multiplier': 1.16},
        {'name': 'spring_break', 'dates': [(3, 18, 3, 25)], 'multiplier': 0.31},
        {'name': 'winter_intersession', 'dates': [(12, 16, 1, 14)], 'multiplier': 0.09},
        {'name': 'thanksgiving_week', 'dates': [(11, 23, 11, 29)], 'multiplier': 0.45},
    ]

    # Default staffing roles
    DEFAULT_STAFFING_ROLES = [
        'actual_foh_general', 'actual_foh_cashier', 'actual_kitchen_prep',
        'actual_kitchen_line', 'actual_dish_room', 'actual_management'
    ]

    def __init__(self, tx_model_path: Optional[str] = None, work_model_path: Optional[str] = None):
        """
        Initialize the StaffingPredictor.

        Args:
            tx_model_path: Path to the trained transaction prediction model (pickle file)
            work_model_path: Path to the trained staffing prediction model (pickle file)
        """
        self.tx_estimator = None
        self.work_estimator = None

        if tx_model_path:
            self.load_tx_model(tx_model_path)
        if work_model_path:
            self.load_work_model(work_model_path)

    def load_tx_model(self, model_path: str) -> None:
        """Load the transaction prediction model from a pickle file."""
        with open(model_path, 'rb') as f:
            self.tx_estimator = pickle.load(f)

    def load_work_model(self, model_path: str) -> None:
        """Load the staffing prediction model from a pickle file."""
        with open(model_path, 'rb') as f:
            self.work_estimator = pickle.load(f)

    def load_models(self, tx_model_path: str, work_model_path: str) -> None:
        """Load both models at once."""
        self.load_tx_model(tx_model_path)
        self.load_work_model(work_model_path)

    def get_academic_period_multiplier(self, date: datetime) -> float:
        """
        Get the seasonal multiplier for a given date based on academic calendar.

        Args:
            date: The date to analyze

        Returns:
            Seasonal multiplier float
        """
        month, day = date.month, date.day

        # Check for special periods first (highest priority)
        for period in self.SPECIAL_PERIODS:
            for date_range in period['dates']:
                start_month, start_day, end_month, end_day = date_range

                # Handle cross-year periods (like winter intersession)
                if start_month > end_month:
                    if (month >= start_month and day >= start_day) or (month <= end_month and day <= end_day):
                        return period['multiplier']
                else:
                    if (month > start_month or (month == start_month and day >= start_day)) and \
                            (month < end_month or (month == end_month and day <= end_day)):
                        return period['multiplier']

        # Regular academic periods
        # Fall semester: August 20 - December 15
        if (month > 8 or (month == 8 and day >= 20)) and (month < 12 or (month == 12 and day <= 15)):
            return 1.0  # fall_semester

        # Spring semester: January 15 - May 15
        elif (month > 1 or (month == 1 and day >= 15)) and (month < 5 or (month == 5 and day <= 15)):
            return 0.96  # spring_semester

        # Summer session: June 1 - August 15
        elif (month > 6 or (month == 6 and day >= 1)) and (month < 8 or (month == 8 and day <= 15)):
            return 0.32  # summer_session

        # Winter break (fallback)
        else:
            return 0.09  # winter_break

    def calculate_enrollment_features(self, date: datetime) -> Dict[str, float]:
        """
        Calculate enrollment-related features for the given date.

        Args:
            date: Date for enrollment calculation

        Returns:
            Dictionary with enrollment features
        """
        base_year = 2024
        years_elapsed = (date.year - base_year) + (date.timetuple().tm_yday / 365.25)

        # Apply year-over-year growth
        growth_factor = (1 + self.OPERATIONAL_CONSTANTS['YOY_GROWTH_RATE']) ** years_elapsed
        total_enrollment = int(self.OPERATIONAL_CONSTANTS['BASE_TOTAL_ENROLLMENT'] * growth_factor)

        # Get seasonal multiplier
        seasonal_multiplier = self.get_academic_period_multiplier(date)

        # Calculate dependent populations
        active_enrollment = int(total_enrollment * seasonal_multiplier)
        residential_students = int(total_enrollment * self.OPERATIONAL_CONSTANTS['RESIDENTIAL_STUDENT_RATIO'])
        commuter_students = total_enrollment - residential_students

        # Calculate meal plan holders
        residential_meal_plans = int(residential_students * self.OPERATIONAL_CONSTANTS['RESIDENTIAL_MANDATORY_RATE'])
        commuter_meal_plans = int(commuter_students * self.OPERATIONAL_CONSTANTS['COMMUTER_VOLUNTARY_RATE'])
        total_meal_plan_holders = int((residential_meal_plans + commuter_meal_plans) * seasonal_multiplier)

        return {
            'total_enrollment': total_enrollment,
            'active_enrollment': active_enrollment,
            'residential_students': residential_students,
            'commuter_students': commuter_students,
            'total_meal_plan_holders': total_meal_plan_holders,
            'enrollment_seasonal_factor': seasonal_multiplier,
            'seasonal_multiplier': seasonal_multiplier,
        }

    def create_model_features(self, date: datetime, weather: str, event: str) -> pd.DataFrame:
        """
        Create feature vector matching your exact training pipeline.

        Args:
            date: Date for prediction
            weather: Weather condition ('sunny', 'cloudy', 'rainy', 'extreme_heat')
            event: Campus event type (key from EVENT_IMPACT_MAP)

        Returns:
            DataFrame with exactly the features from your available_features list
        """
        # Validate inputs
        if weather not in self.WEATHER_IMPACT_MAP:
            raise ValueError(f"Invalid weather: {weather}. Must be one of {list(self.WEATHER_IMPACT_MAP.keys())}")

        if event not in self.EVENT_IMPACT_MAP:
            raise ValueError(f"Invalid event: {event}. Must be one of {list(self.EVENT_IMPACT_MAP.keys())}")

        # Basic date features
        day_of_week = date.weekday()
        month = date.month
        year = date.year
        day_of_year = date.timetuple().tm_yday
        week_of_year = date.isocalendar()[1]
        is_weekend = 1 if day_of_week >= 5 else 0

        # Get enrollment features
        enrollment_features = self.calculate_enrollment_features(date)

        # Environmental impacts
        weather_impact = self.WEATHER_IMPACT_MAP[weather]
        event_impact = self.EVENT_IMPACT_MAP[event]

        # Create DataFrame with exactly your available_features
        features = pd.DataFrame({
            # Date and time features (all numerical)
            'day_of_week': [day_of_week],
            'month': [month],
            'year': [year],
            'day_of_year': [day_of_year],
            'week_of_year': [week_of_year],
            'is_weekend': [is_weekend],

            # Academic calendar (numerical version)
            'seasonal_multiplier': [enrollment_features['seasonal_multiplier']],

            # Student population (known from enrollment data)
            'total_enrollment': [enrollment_features['total_enrollment']],
            'active_enrollment': [enrollment_features['active_enrollment']],
            'residential_students': [enrollment_features['residential_students']],
            'commuter_students': [enrollment_features['commuter_students']],
            'total_meal_plan_holders': [enrollment_features['total_meal_plan_holders']],
            'enrollment_seasonal_factor': [enrollment_features['enrollment_seasonal_factor']],

            # Environmental factors (numerical impacts)
            'weather_impact': [weather_impact],
            'event_impact': [event_impact],
        })

        return features

    def predict_transactions(self, date: datetime, weather: str, event: str) -> int:
        """
        First stage: Predict total transactions for a given date.

        Args:
            date: Date for prediction
            weather: Weather condition
            event: Campus event type

        Returns:
            Predicted number of transactions

        Raises:
            ValueError: If transaction model is not loaded
        """
        if self.tx_estimator is None:
            raise ValueError("Transaction model not loaded. Use load_tx_model() or load_models() first.")

        features = self.create_model_features(date, weather, event)
        predicted_transactions = self.tx_estimator.predict(features)[0]

        # Apply reasonable bounds
        return max(0, int(predicted_transactions))

    def predict_staffing_requirements(
            self,
            date: datetime,
            weather: str,
            event: str,
            target_features: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Two-stage prediction: First predict transactions, then predict staffing requirements.

        Args:
            date: Date for prediction
            weather: Weather condition
            event: Campus event type
            target_features: List of staffing role column names

        Returns:
            Dictionary with predicted staffing hours for each role

        Raises:
            ValueError: If either model is not loaded
        """
        if self.tx_estimator is None:
            raise ValueError("Transaction model not loaded. Use load_tx_model() or load_models() first.")
        if self.work_estimator is None:
            raise ValueError("Work model not loaded. Use load_work_model() or load_models() first.")

        if target_features is None:
            target_features = self.DEFAULT_STAFFING_ROLES.copy()

        # Stage 1: Predict transactions
        predicted_transactions = self.predict_transactions(date, weather, event)

        # Stage 2: Create features including predicted transactions
        features = self.create_model_features(date, weather, event)
        features['total_transactions'] = predicted_transactions

        # Predict staffing requirements
        staffing_predictions = self.work_estimator.predict(features)[0]

        # Create results dictionary
        results = {}
        for i, role in enumerate(target_features):
            predicted_hours = max(0.0, staffing_predictions[i])  # No negative hours
            results[role] = round(predicted_hours, 1)

        # Add summary metrics
        total_hours = sum(results.values())
        results['total_predicted_hours'] = round(total_hours, 1)
        results['predicted_transactions'] = predicted_transactions

        # Round all values to 1 decimal place
        for key in results:
            if isinstance(results[key], float):
                results[key] = round(results[key], 1)

        return results

    def batch_predict(
            self,
            dates: List[datetime],
            weather_conditions: List[str],
            events: List[str],
            target_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Batch prediction for multiple dates.

        Args:
            dates: List of dates for prediction
            weather_conditions: List of weather conditions (same length as dates)
            events: List of events (same length as dates)
            target_features: List of staffing role column names

        Returns:
            DataFrame with predictions for all dates

        Raises:
            ValueError: If input lists have different lengths or models not loaded
        """
        if not (len(dates) == len(weather_conditions) == len(events)):
            raise ValueError("All input lists must have the same length")

        if target_features is None:
            target_features = self.DEFAULT_STAFFING_ROLES.copy()

        results = []

        for date, weather, event in zip(dates, weather_conditions, events):
            try:
                prediction = self.predict_staffing_requirements(date, weather, event, target_features)
                prediction['date'] = date.strftime('%Y-%m-%d')
                prediction['weather'] = weather
                prediction['event'] = event
                results.append(prediction)
            except Exception as e:
                print(f"Error predicting for {date}: {e}")
                continue

        return pd.DataFrame(results)

    def get_available_weather_conditions(self) -> List[str]:
        """Get list of available weather conditions."""
        return list(self.WEATHER_IMPACT_MAP.keys())

    def get_available_events(self) -> List[str]:
        """Get list of available campus events."""
        return list(self.EVENT_IMPACT_MAP.keys())

    def get_default_staffing_roles(self) -> List[str]:
        """Get list of default staffing roles."""
        return self.DEFAULT_STAFFING_ROLES.copy()

    def __repr__(self) -> str:
        tx_loaded = "✓" if self.tx_estimator is not None else "✗"
        work_loaded = "✓" if self.work_estimator is not None else "✗"
        return f"StaffingPredictor(tx_model={tx_loaded}, work_model={work_loaded})"

# # Example usage:
# # Initialize the predictor
# predictor = StaffingPredictor()
#
# # Load your trained models
# predictor.load_models('./tx_model.pkl', './work_model.pkl')
#
# print("Example prediction pipeline:")
# test_date = datetime.datetime(2025, 10, 30)
# test_weather = 'sunny'
# test_event = 'club_fair'
#
# print(f"Date: {test_date.strftime('%A, %B %d, %Y')}")
# print(f"Weather: {test_weather}")
# print(f"Event: {test_event}")
#
# # predicted_transactions = predictor.predict_transactions(test_date, test_weather, test_event)
# # print(f"Predicted transactions: {predicted_transactions}")
#
# # Predict staffing requirements
# staffing_predictions = predictor.predict_staffing_requirements(
#     date=test_date,
#     weather=test_weather,
#     event=test_event,
#     target_features=['actual_foh_general', 'actual_foh_cashier', 'actual_kitchen_prep',
#                      'actual_kitchen_line', 'actual_dish_room', 'actual_management']
# )
# print("Staffing predictions:")
# print(staffing_predictions)
#
# # Test batch prediction
# batch_dates = [
#     datetime.datetime(2025, 7, 30), datetime.datetime(2025, 7, 31), datetime.datetime(2025, 8, 1),
#     datetime.datetime(2025, 8, 2), datetime.datetime(2025, 8, 3)
# ]
# batch_weather = ['sunny', 'cloudy', 'rainy', 'sunny', 'extreme_heat']
# batch_events = ['club_fair', 'regular_day', 'career_fair', 'sports_events', 'graduation']
#
# batch_predictions = predictor.batch_predict(
#     dates=batch_dates,
#     weather_conditions=batch_weather,
#     events=batch_events,
#     target_features=['actual_foh_general', 'actual_foh_cashier', 'actual_kitchen_prep',
#                      'actual_kitchen_line', 'actual_dish_room', 'actual_management']
# )
# print("Batch predictions:")
# print(batch_predictions)
#
# # Additional utility methods
# print(f"Available weather conditions: {predictor.get_available_weather_conditions()}")
# print(f"Available events: {predictor.get_available_events()}")
# print(f"Default staffing roles: {predictor.get_default_staffing_roles()}")
# print(f"Predictor status: {predictor}")
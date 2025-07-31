import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class CenterpointeDiningDataGenerator:
    """
    Comprehensive dataset generator for Cal Poly Pomona's Centerpointe Dining Commons
    Incorporates real operational patterns, academic calendars, and university-specific factors
    to generate realistic data for ML training to predict actual staffing needs.

    Based on research of CPP's actual operations:
    - 35,000 sq ft facility serving 680 students simultaneously
    - 8 dining platforms with biometric access
    - $2,611+ mandatory meal plans for residential students
    - Dual-currency system (Dining Dollars + Bronco Bucks)
    - Self-operated by CPP Foundation (not external contractor)
    """

    def __init__(self, config: Dict = None):
        """
        Initialize generator with comprehensive configuration

        Args:
            config: Dictionary of configuration parameters to override defaults
        """
        self.config = self._load_default_config()
        if config:
            self._update_config_recursively(self.config, config)

        # Set random seeds for reproducibility
        np.random.seed(self.config['random_seed'])
        random.seed(self.config['random_seed'])

        # Initialize operational components
        self._setup_academic_calendar()
        self._setup_meal_plan_cycles()
        self._initialize_operational_state()

        print(f"✓ Centerpointe Dataset Generator initialized")
        print(f"  Date range: {self.config['start_date']} to {self.config['end_date']}")
        print(f"  Base enrollment: {self.config['student_population']['total_enrollment_base']:,}")
        print(f"  YoY growth rate: {self.config['student_population']['yoy_growth_rate']:.1%}")

    def _update_config_recursively(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested configuration dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_config_recursively(base_dict[key], value)
            else:
                base_dict[key] = value

    def _load_default_config(self) -> Dict:
        """
        Comprehensive default configuration based on Cal Poly Pomona Centerpointe operations
        """
        return {
            # Core Parameters
            'random_seed': 42,
            'start_date': '2024-01-01',
            'end_date': '2026-12-31',
            'facility_name': 'Centerpointe Dining Commons',

            # Student Population Dynamics (Based on CPP ~31K enrollment)
            'student_population': {
                'total_enrollment_base': 31000,
                'yoy_growth_rate': 0.022,  # 2.2% annual growth (realistic for CPP)
                'residential_student_ratio': 0.152,  # ~15.2% live on campus
                'meal_plan_participation': {
                    'residential_mandatory_rate': 1.0,  # 100% of residential required
                    'commuter_voluntary_rate': 0.078,   # 7.8% of commuters buy plans
                    'faculty_staff_rate': 0.012,        # Faculty/staff meal plans
                },
                'enrollment_seasonal_variation': {
                    'fall_semester': 1.0,      # Peak enrollment
                    'spring_semester': 0.96,   # Some don't return
                    'summer_session': 0.32,    # Summer programs only
                    'winter_intersession': 0.08, # Minimal programs
                }
            },

            # CORRECTED CPP Meal Plan System - Based on REAL usage patterns
            'meal_plans': {
                'plan_types': {
                    'Unlimited': {
                        'cost_per_semester': 2611,
                        'swipes_per_day': 'unlimited',
                        'dining_dollars': 250,
                        'typical_daily_usage': 0.32,  # CORRECTED: Was 2.4, real data shows ~0.27
                        'utilization_rate': 0.847,   # How much of available they use
                        'student_distribution': 0.28, # % of students with this plan
                    },
                    'Block_220': {
                        'cost_per_semester': 2895,
                        'swipes_per_semester': 220,
                        'dining_dollars': 460,
                        'typical_daily_usage': 0.25,  # CORRECTED: Was 1.83, reduced proportionally
                        'utilization_rate': 0.923,
                        'student_distribution': 0.24,
                    },
                    'Block_180': {
                        'cost_per_semester': 2781,
                        'swipes_per_semester': 180,
                        'dining_dollars': 560,
                        'typical_daily_usage': 0.20,  # CORRECTED: Was 1.50
                        'utilization_rate': 0.891,
                        'student_distribution': 0.26,
                    },
                    'Block_140': {
                        'cost_per_semester': 2611,
                        'swipes_per_semester': 140,
                        'dining_dollars': 660,
                        'typical_daily_usage': 0.16,  # CORRECTED: Was 1.17
                        'utilization_rate': 0.874,
                        'student_distribution': 0.17,
                    },
                    'Suites_Flex': {
                        'cost_per_semester': 1915,
                        'swipes_per_semester': 70,
                        'dining_dollars': 750,
                        'typical_daily_usage': 0.08,  # CORRECTED: Was 0.58
                        'utilization_rate': 0.756,
                        'student_distribution': 0.05,
                    }
                },
                # Meal plan purchasing timing
                'purchase_cycles': {
                    'fall_registration': (7, 1, 8, 15),   # July 1 - Aug 15
                    'spring_registration': (11, 15, 1, 10), # Nov 15 - Jan 10
                    'summer_registration': (4, 1, 5, 30),   # Apr 1 - May 30
                }
            },

            # Centerpointe Operating Schedule (unchanged)
            'operating_hours': {
                'academic_year': {
                    'weekday': {
                        'breakfast': (7.0, 10.0),      # 7:00 AM - 10:00 AM
                        'lunch': (10.75, 14.5),       # 10:45 AM - 2:30 PM
                        'dinner': (17.0, 19.5),       # 5:00 PM - 7:30 PM
                        'late_night': (21.0, 23.0),   # 9:00 PM - 11:00 PM (limited)
                    },
                    'weekend': {
                        'brunch': (11.0, 15.0),       # 11:00 AM - 3:00 PM
                        'dinner': (17.0, 19.5),       # 5:00 PM - 7:30 PM
                        'late_night': (21.0, 22.5),   # 9:00 PM - 10:30 PM
                    }
                },
                'summer_session': {
                    'weekday': {
                        'lunch': (11.0, 14.0),        # 11:00 AM - 2:00 PM
                        'dinner': (17.0, 19.0),       # 5:00 PM - 7:00 PM
                    },
                    'weekend': {
                        'lunch': (11.5, 14.0),        # 11:30 AM - 2:00 PM
                        'dinner': (17.0, 18.5),       # 5:00 PM - 6:30 PM
                    }
                },
                'break_periods': {
                    'weekday': {
                        'lunch': (11.5, 13.5),        # 11:30 AM - 1:30 PM
                    },
                    'weekend': 'closed'
                }
            },

            # Academic Calendar & Seasonal Effects (unchanged)
            'academic_calendar': {
                'semester_dates': {
                    'fall_start': (8, 20),      # August 20
                    'fall_end': (12, 15),       # December 15
                    'spring_start': (1, 15),    # January 15
                    'spring_end': (5, 15),      # May 15
                    'summer_start': (6, 1),     # June 1
                    'summer_end': (8, 15),      # August 15
                },
                'special_periods': {
                    'move_in_week': {
                        'dates': [(8, 15, 8, 22)],  # Aug 15-22
                        'multiplier': 1.28,
                        'description': 'New students exploring dining options'
                    },
                    'finals_weeks': {
                        'dates': [(12, 8, 12, 15), (5, 8, 5, 15)], # Fall & Spring finals
                        'multiplier': 1.16,
                        'description': 'Extended hours, stress eating, all-nighters'
                    },
                    'spring_break': {
                        'dates': [(3, 18, 3, 25)],  # Varies yearly
                        'multiplier': 0.31,
                        'description': 'Most students away'
                    },
                    'winter_intersession': {
                        'dates': [(12, 16, 1, 14)],
                        'multiplier': 0.09,
                        'description': 'Minimal campus population'
                    },
                    'thanksgiving_week': {
                        'dates': [(11, 23, 11, 29)],
                        'multiplier': 0.45,
                        'description': 'Many students travel home'
                    }
                },
                'weekly_patterns': {
                    'monday': 0.92,      # Slow start to week
                    'tuesday': 1.05,     # Peak academic day
                    'wednesday': 1.08,   # Peak academic day (U-Hour)
                    'thursday': 1.03,    # Busy academic day
                    'friday': 0.89,      # Many leave campus early
                    'saturday': 0.71,    # Weekend dining
                    'sunday': 0.82,      # Students return, meal prep
                }
            },

            # Environmental Factors (unchanged)
            'environmental_factors': {
                'weather_patterns': {
                    'seasonal_probabilities': {
                        'winter': [0.58, 0.27, 0.14, 0.01],  # sunny, cloudy, rainy, extreme
                        'spring': [0.72, 0.19, 0.08, 0.01],
                        'summer': [0.79, 0.17, 0.02, 0.02],
                        'fall': [0.71, 0.22, 0.06, 0.01],
                    },
                    'weather_impacts': {
                        'sunny': 1.0,
                        'cloudy': 1.023,      # Slight increase in dining
                        'rainy': 1.147,       # Significant increase (stay indoors)
                        'extreme_heat': 0.891, # Decreased appetite
                    }
                },
                'campus_events': {
                    'event_calendar': {
                        'regular_day': {'probability': 0.823, 'impact': 1.0},
                        'club_fair': {'probability': 0.025, 'impact': 1.34, 'typical_dates': [(9, 5), (1, 25)]},
                        'career_fair': {'probability': 0.018, 'impact': 1.23, 'typical_dates': [(10, 15), (2, 20)]},
                        'sports_events': {'probability': 0.047, 'impact': 1.12},
                        'graduation': {'probability': 0.008, 'impact': 1.43, 'typical_dates': [(12, 16), (5, 18)]},
                        'parent_weekend': {'probability': 0.012, 'impact': 1.38, 'typical_dates': [(10, 12)]},
                        'prospective_student_day': {'probability': 0.015, 'impact': 1.19},
                        'conference_hosting': {'probability': 0.032, 'impact': 1.16},
                        'campus_construction': {'probability': 0.020, 'impact': 0.94}, # Reduced accessibility
                    }
                }
            },

            # CORRECTED Payment & Transaction Patterns - Based on REAL data
            'transaction_patterns': {
                'payment_methods': {
                    'meal_swipes': 0.978,        # CORRECTED: Was 0.703, real data shows 97.8%
                    'dining_dollars': 0.014,     # CORRECTED: Reduced proportionally
                    'bronco_bucks': 0.005,       # CORRECTED: Reduced proportionally
                    'credit_debit': 0.003,       # CORRECTED: Was 0.022, much smaller in reality
                },
                'platform_popularity': {
                    'Between_Two_Slices': 0.148,  # Sandwich station
                    'Firehouse': 0.142,           # Grilled items
                    'Fusion_Bar': 0.181,          # Asian cuisine (most popular)
                    'Gone_Global': 0.119,         # International foods
                    'Charred': 0.134,             # Carving station
                    'Sweet_Spot': 0.083,          # Desserts
                    'Sushi_Bar': 0.097,           # Sushi (generates waits)
                    'Salad_Bar': 0.096,           # Healthy options
                },
                'meal_period_distribution': {
                    'breakfast': 0.18,    # Light breakfast crowd
                    'lunch': 0.52,        # Peak meal period
                    'dinner': 0.27,       # Moderate dinner crowd
                    'late_night': 0.03,   # Limited late night
                },
                'guest_multiplier': 0.05,  # CORRECTED: Was 1.14, much lower guest traffic

                # NEW: Separate revenue calculation model
                'revenue_model': {
                    'meal_swipe_immediate_revenue': 0.0,  # Swipes don't generate daily cash revenue
                    'dollar_transaction_average': 12.54,  # Based on real data analysis
                    'include_meal_plan_revenue': False,   # Don't count prepaid meal plans as daily revenue
                }
            },

            # DRAMATICALLY CORRECTED Staffing Configuration
            'staffing_model': {
                'roles': {
                    'FOH_General': {
                        'description': 'Customer service, food serving, cleaning',
                        'base_hours_per_period': 5.0,     # CORRECTED: Was 11.8, reduced by ~58%
                        'volume_scaling_factor': 1.23,    # High sensitivity to volume
                        'minimum_coverage_hours': 4.0,    # CORRECTED: Was 8.0
                        'peak_hour_multiplier': 1.20,     # CORRECTED: Was 1.35, more realistic
                        'efficiency_factors': {
                            'experienced_staff': 0.95,    # CORRECTED: Lower efficiency assumption
                            'new_hires': 0.65,            # CORRECTED: Was 0.76
                            'student_workers': 0.80,      # CORRECTED: Was 0.89
                        }
                    },
                    'FOH_Cashier': {
                        'description': 'Transaction processing, guest relations, biometric system',
                        'base_hours_per_period': 2.5,     # CORRECTED: Was 5.7, reduced by ~56%
                        'volume_scaling_factor': 1.14,    # Moderate sensitivity
                        'minimum_coverage_hours': 2.0,    # CORRECTED: Was 4.0
                        'peak_hour_multiplier': 1.15,     # CORRECTED: Was 1.28
                        'efficiency_factors': {
                            'experienced_staff': 1.00,    # CORRECTED: More realistic
                            'new_hires': 0.65,            # CORRECTED: Was 0.71
                            'student_workers': 0.85,      # CORRECTED: Was 0.93
                        }
                    },
                    'Kitchen_Prep': {
                        'description': 'Food preparation, ingredient processing, station setup',
                        'base_hours_per_period': 7.0,     # CORRECTED: Was 15.4, reduced by ~55%
                        'volume_scaling_factor': 0.91,    # More consistent staffing needs
                        'minimum_coverage_hours': 6.0,    # CORRECTED: Was 12.0
                        'peak_hour_multiplier': 1.08,     # CORRECTED: Was 1.12
                        'efficiency_factors': {
                            'experienced_staff': 1.05,    # CORRECTED: Was 1.15
                            'new_hires': 0.60,            # CORRECTED: Was 0.68
                            'student_workers': 0.75,      # CORRECTED: Was 0.84
                        }
                    },
                    'Kitchen_Line': {
                        'description': 'Active cooking, food assembly, platform management',
                        'base_hours_per_period': 9.0,     # CORRECTED: Was 19.2, reduced by ~53%
                        'volume_scaling_factor': 1.02,    # Moderate scaling
                        'minimum_coverage_hours': 8.0,    # CORRECTED: Was 16.0
                        'peak_hour_multiplier': 1.10,     # CORRECTED: Was 1.18
                        'efficiency_factors': {
                            'experienced_staff': 1.00,    # CORRECTED: Was 1.11
                            'new_hires': 0.65,            # CORRECTED: Was 0.73
                            'student_workers': 0.78,      # CORRECTED: Was 0.87
                        }
                    },
                    'Dish_Room': {
                        'description': 'Dishwashing, sanitation, equipment cleaning',
                        'base_hours_per_period': 3.5,     # CORRECTED: Was 7.9, reduced by ~56%
                        'volume_scaling_factor': 1.17,    # Scales with total meals
                        'minimum_coverage_hours': 3.0,    # CORRECTED: Was 6.0
                        'peak_hour_multiplier': 1.20,     # CORRECTED: Was 1.31
                        'efficiency_factors': {
                            'experienced_staff': 1.00,    # CORRECTED: Was 1.09
                            'new_hires': 0.70,            # CORRECTED: Was 0.78
                            'student_workers': 0.85,      # CORRECTED: Was 0.91
                        }
                    },
                    'Management': {
                        'description': 'Supervision, coordination, problem resolution',
                        'base_hours_per_period': 2.0,     # CORRECTED: Was 3.8, reduced by ~47%
                        'volume_scaling_factor': 0.82,    # Least variable
                        'minimum_coverage_hours': 1.5,    # CORRECTED: Was 2.0
                        'peak_hour_multiplier': 1.05,     # CORRECTED: Was 1.09
                        'efficiency_factors': {
                            'experienced_staff': 1.10,    # CORRECTED: Was 1.18
                            'new_hires': 0.80,            # CORRECTED: Was 0.84
                            'student_workers': 0.0,       # No student managers
                        }
                    }
                },
                'labor_costs': {
                    'average_hourly_rate': 18.75,  # Blended rate including benefits
                    'student_worker_rate': 16.50,
                    'experienced_staff_rate': 21.25,
                    'management_rate': 28.50,
                },
                'scheduling_constraints': {
                    'student_worker_availability': {
                        'weekday_morning': 0.58,    # Many in class
                        'weekday_afternoon': 0.82,   # More available
                        'weekday_evening': 0.91,     # Most available
                        'weekend_day': 0.97,         # Highly available
                        'weekend_evening': 0.89,     # Some social conflicts
                        'finals_week': 0.47,         # Limited due to studying
                        'summer_session': 0.39,      # Many away from campus
                    },
                    'minimum_staffing_ratios': {
                        'customers_per_foh': 85,     # Max customers per FOH staff
                        'transactions_per_cashier': 120, # Max transactions per cashier
                    },
                    # NEW: Realistic productivity expectations
                    'productivity_targets': {
                        'transactions_per_labor_hour': 1.9,  # Based on real data
                        'revenue_per_labor_hour': 1.5,       # Based on real data (for dollar transactions)
                    }
                }
            },

            'facility_specs': {
                'maximum_simultaneous_capacity': 680,  # Actual Centerpointe capacity
                'total_square_footage': 35000,
                'number_of_dining_platforms': 8,
                'biometric_scanner_throughput': 45,    # Students per minute
                'peak_hour_duration': 3.0,             # Hours of peak traffic
                'kitchen_prep_capacity': 2400,         # Meals prep capacity per day
                'dish_room_capacity': 950,             # Place settings per hour
            }
        }



    def _setup_academic_calendar(self):
        """Initialize academic calendar and special periods"""
        self.academic_periods = {}
        self.special_periods = []

        # Process special periods into date ranges
        for period_name, period_data in self.config['academic_calendar']['special_periods'].items():
            for date_range in period_data['dates']:
                start_month, start_day, end_month, end_day = date_range
                self.special_periods.append({
                    'name': period_name,
                    'start': (start_month, start_day),
                    'end': (end_month, end_day),
                    'multiplier': period_data['multiplier'],
                    'description': period_data['description']
                })

    def _setup_meal_plan_cycles(self):
        """Initialize meal plan purchasing and utilization cycles"""
        self.meal_plan_cycles = {}
        for cycle_name, dates in self.config['meal_plans']['purchase_cycles'].items():
            self.meal_plan_cycles[cycle_name] = {
                'start': (dates[0], dates[1]),
                'end': (dates[2], dates[3]),
                'registration_intensity': 0.0  # Will be calculated dynamically
            }

    def _initialize_operational_state(self):
        """Initialize operational state tracking"""
        self.operational_state = {
            'current_semester': None,
            'days_into_semester': 0,
            'accumulated_staff_fatigue': 0.0,
            'equipment_reliability': 1.0,
            'recent_events_impact': 1.0,
        }

    def _get_academic_period_info(self, date: datetime) -> Dict:
        """
        Determine comprehensive academic period information for given date
        """
        month, day = date.month, date.day

        # Check for special periods first (highest priority)
        for period in self.special_periods:
            start_month, start_day = period['start']
            end_month, end_day = period['end']

            # Handle cross-year periods
            if start_month > end_month:  # Crosses year boundary
                if (month >= start_month and day >= start_day) or (month <= end_month and day <= end_day):
                    return {
                        'period_name': period['name'],
                        'period_type': 'special',
                        'multiplier': period['multiplier'],
                        'description': period['description']
                    }
            else:  # Same year
                if (month > start_month or (month == start_month and day >= start_day)) and \
                        (month < end_month or (month == end_month and day <= end_day)):
                    return {
                        'period_name': period['name'],
                        'period_type': 'special',
                        'multiplier': period['multiplier'],
                        'description': period['description']
                    }

        # Determine regular academic period
        semester_dates = self.config['academic_calendar']['semester_dates']

        # Fall semester: August 20 - December 15
        fall_start_month, fall_start_day = semester_dates['fall_start']  # (8, 20)
        fall_end_month, fall_end_day = semester_dates['fall_end']        # (12, 15)

        # Spring semester: January 15 - May 15
        spring_start_month, spring_start_day = semester_dates['spring_start']  # (1, 15)
        spring_end_month, spring_end_day = semester_dates['spring_end']        # (5, 15)

        # Summer session: June 1 - August 15
        summer_start_month, summer_start_day = semester_dates['summer_start']  # (6, 1)
        summer_end_month, summer_end_day = semester_dates['summer_end']        # (8, 15)

        # Check Fall Semester (August 20 - December 15)
        if (month > fall_start_month or (month == fall_start_month and day >= fall_start_day)) and \
                (month < fall_end_month or (month == fall_end_month and day <= fall_end_day)):
            return {
                'period_name': 'fall_semester',
                'period_type': 'regular',
                'multiplier': 1.0,
                'description': 'Regular fall semester'
            }

        # Check Spring Semester (January 15 - May 15)
        elif (month > spring_start_month or (month == spring_start_month and day >= spring_start_day)) and \
                (month < spring_end_month or (month == spring_end_month and day <= spring_end_day)):
            return {
                'period_name': 'spring_semester',
                'period_type': 'regular',
                'multiplier': 0.96,
                'description': 'Regular spring semester'
            }

        # Check Summer Session (June 1 - August 15)
        elif (month > summer_start_month or (month == summer_start_month and day >= summer_start_day)) and \
                (month < summer_end_month or (month == summer_end_month and day <= summer_end_day)):
            return {
                'period_name': 'summer_session',
                'period_type': 'regular',
                'multiplier': 0.32,
                'description': 'Regular summer session'
            }

        # Winter Break (December 16 - January 14) and any remaining gaps
        elif (month == 12 and day > fall_end_day) or \
                (month == 1 and day < spring_start_day) or \
                (month > spring_end_month and month < summer_start_month):  # May 16 - May 31
            return {
                'period_name': 'winter_break',
                'period_type': 'regular',
                'multiplier': 0.09,
                'description': 'Winter break / intersession'
            }

        # Fallback (shouldn't happen with proper date ranges)
        else:
            return {
                'period_name': 'unknown_period',
                'period_type': 'regular',
                'multiplier': 0.5,
                'description': 'Unknown academic period'
            }

    def _calculate_student_population(self, date: datetime) -> Dict:
        """Calculate dynamic student population considering growth and seasonal factors"""
        base_year = 2024
        years_elapsed = (date.year - base_year) + (date.timetuple().tm_yday / 365.25)

        # Apply year-over-year growth
        growth_factor = (1 + self.config['student_population']['yoy_growth_rate']) ** years_elapsed
        total_enrollment = int(self.config['student_population']['total_enrollment_base'] * growth_factor)

        # Apply seasonal enrollment variation
        period_info = self._get_academic_period_info(date)
        seasonal_factor = self.config['student_population']['enrollment_seasonal_variation'].get(
            period_info['period_name'], 1.0
        )

        active_enrollment = int(total_enrollment * seasonal_factor)
        residential_students = int(active_enrollment * self.config['student_population']['residential_student_ratio'])
        commuter_students = active_enrollment - residential_students

        # Calculate meal plan participation
        participation_rates = self.config['student_population']['meal_plan_participation']
        residential_meal_plans = int(residential_students * participation_rates['residential_mandatory_rate'])
        commuter_meal_plans = int(commuter_students * participation_rates['commuter_voluntary_rate'])

        total_meal_plan_holders = residential_meal_plans + commuter_meal_plans

        return {
            'total_enrollment': total_enrollment,
            'active_enrollment': active_enrollment,
            'residential_students': residential_students,
            'commuter_students': commuter_students,
            'total_meal_plan_holders': total_meal_plan_holders,
            'seasonal_factor': seasonal_factor,
        }

    def _generate_environmental_conditions(self, date: datetime) -> Dict:
        """Generate weather and campus event conditions"""
        month = date.month

        # Determine season for weather patterns
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        else:
            season = 'fall'

        # Generate weather
        weather_probs = self.config['environmental_factors']['weather_patterns']['seasonal_probabilities'][season]
        weather_types = ['sunny', 'cloudy', 'rainy', 'extreme_heat']
        weather = np.random.choice(weather_types, p=weather_probs)
        weather_impact = self.config['environmental_factors']['weather_patterns']['weather_impacts'][weather]

        # Generate campus events
        event_calendar = self.config['environmental_factors']['campus_events']['event_calendar']

        # Check for scheduled events first
        for event_type, event_data in event_calendar.items():
            if 'typical_dates' in event_data:
                for typical_date in event_data['typical_dates']:
                    event_month, event_day = typical_date
                    if abs(month - event_month) == 0 and abs(date.day - event_day) <= 2:
                        # Higher probability for scheduled events
                        if np.random.random() < 0.7:  # 70% chance if near scheduled date
                            return {
                                'weather': weather,
                                'weather_impact': weather_impact,
                                'campus_event': event_type,
                                'event_impact': event_data['impact'],
                                'event_scheduled': True
                            }

        # Random event generation
        event_types = list(event_calendar.keys())
        event_probs = [event_calendar[event]['probability'] for event in event_types]

        # Adjust probabilities based on academic period
        period_info = self._get_academic_period_info(date)
        if period_info['period_name'] in ['winter_break', 'summer_session']:
            # Much lower event probability during breaks
            event_probs = [0.95] + [p * 0.1 for p in event_probs[1:]]
            event_probs = [p / sum(event_probs) for p in event_probs]  # Normalize

        selected_event = np.random.choice(event_types, p=event_probs)
        event_impact = event_calendar[selected_event]['impact']

        return {
            'weather': weather,
            'weather_impact': weather_impact,
            'campus_event': selected_event,
            'event_impact': event_impact,
            'event_scheduled': False
        }

    def _calculate_base_transactions(self, date: datetime, population_data: Dict,
                                     period_info: Dict, environmental_data: Dict) -> Dict:
        """Calculate expected transactions based on all factors with corrected revenue model"""

        meal_plan_holders = population_data['total_meal_plan_holders']
        day_of_week = date.weekday()

        # Calculate base transaction rate per meal plan holder
        daily_transaction_rate = 0.0
        meal_plan_types = self.config['meal_plans']['plan_types']

        for plan_type, plan_data in meal_plan_types.items():
            plan_distribution = plan_data['student_distribution']
            daily_usage = plan_data['typical_daily_usage']
            utilization = plan_data['utilization_rate']

            daily_transaction_rate += (daily_usage * utilization * plan_distribution)

        # Apply day-of-week patterns
        day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_multiplier = self.config['academic_calendar']['weekly_patterns'][day_names[day_of_week]]

        # Calculate base transactions
        base_transactions = meal_plan_holders * daily_transaction_rate * day_multiplier

        # Apply period multiplier
        base_transactions *= period_info['multiplier']

        # Apply environmental factors
        base_transactions *= environmental_data['weather_impact']
        base_transactions *= environmental_data['event_impact']

        # Add guest transactions (non-meal plan holders) - CORRECTED multiplier
        guest_transactions = base_transactions * self.config['transaction_patterns']['guest_multiplier']

        total_transactions = int(base_transactions + guest_transactions)

        # Distribute by payment method - CORRECTED ratios
        payment_methods = self.config['transaction_patterns']['payment_methods']
        meal_swipes = int(total_transactions * payment_methods['meal_swipes'])
        dining_dollars = int(total_transactions * payment_methods['dining_dollars'])
        bronco_bucks = int(total_transactions * payment_methods['bronco_bucks'])
        credit_debit = total_transactions - meal_swipes - dining_dollars - bronco_bucks

        # CORRECTED: Calculate revenue from dollar transactions only
        revenue_config = self.config['transaction_patterns']['revenue_model']
        dollar_transactions = dining_dollars + bronco_bucks + credit_debit

        if revenue_config['include_meal_plan_revenue']:
            # If including meal plan revenue (legacy mode)
            avg_transaction_value = 13.25  # Old model
            daily_revenue = total_transactions * avg_transaction_value
        else:
            # NEW: Only dollar transactions generate immediate revenue
            avg_dollar_value = revenue_config['dollar_transaction_average']
            daily_revenue = dollar_transactions * avg_dollar_value

            # Meal swipes generate $0 immediate revenue (they're prepaid)
            meal_swipe_revenue = meal_swipes * revenue_config['meal_swipe_immediate_revenue']
            daily_revenue += meal_swipe_revenue

        # Distribute by meal period
        meal_periods = self.config['transaction_patterns']['meal_period_distribution']
        breakfast_transactions = int(total_transactions * meal_periods['breakfast'])
        lunch_transactions = int(total_transactions * meal_periods['lunch'])
        dinner_transactions = int(total_transactions * meal_periods['dinner'])
        late_night_transactions = total_transactions - breakfast_transactions - lunch_transactions - dinner_transactions

        return {
            'total_transactions': max(0, total_transactions),
            'base_transaction_rate': daily_transaction_rate,
            'payment_breakdown': {
                'meal_swipes': meal_swipes,
                'dining_dollars': dining_dollars,
                'bronco_bucks': bronco_bucks,
                'credit_debit': credit_debit,
            },
            'meal_period_breakdown': {
                'breakfast': breakfast_transactions,
                'lunch': lunch_transactions,
                'dinner': dinner_transactions,
                'late_night': late_night_transactions,
            },
            'guest_transactions': int(guest_transactions),
            # NEW: Revenue calculation results
            'revenue_data': {
                'daily_revenue': round(daily_revenue, 2),
                'dollar_transactions': dollar_transactions,
                'revenue_per_transaction': round(daily_revenue / max(1, total_transactions), 2),
                'revenue_per_dollar_transaction': round(daily_revenue / max(1, dollar_transactions), 2) if dollar_transactions > 0 else 0.0,
            }
        }


    def _calculate_staffing_requirements(self, date: datetime, transaction_data: Dict,
                                         period_info: Dict, environmental_data: Dict) -> Dict:
        """Calculate actual staffing hours needed for each role"""

        total_transactions = transaction_data['total_transactions']
        is_weekend = date.weekday() >= 5

        # Determine operating schedule
        if period_info['period_name'] == 'summer_session':
            schedule_key = 'summer_session'
        elif period_info['period_name'] in ['winter_break']:
            schedule_key = 'break_periods'
        else:
            schedule_key = 'academic_year'

        if is_weekend:
            day_schedule = self.config['operating_hours'][schedule_key].get('weekend', {})
        else:
            day_schedule = self.config['operating_hours'][schedule_key].get('weekday', {})

        if day_schedule == 'closed':
            # Return zero staffing for closed days
            return {f'actual_{role.lower()}': 0.0 for role in self.config['staffing_model']['roles'].keys()}

        # Calculate number of meal periods
        num_periods = len(day_schedule)
        total_operating_hours = sum(end - start for start, end in day_schedule.values())

        staffing_requirements = {}

        for role, role_config in self.config['staffing_model']['roles'].items():
            # Base staffing calculation
            base_hours = role_config['base_hours_per_period'] * num_periods

            # Scale with transaction volume (sublinear scaling)
            volume_factor = (total_transactions / 2000) ** 0.73  # Sublinear scaling
            scaled_hours = base_hours * volume_factor * role_config['volume_scaling_factor']

            # Apply peak hour adjustments
            peak_transactions = max(transaction_data['meal_period_breakdown'].values())
            if peak_transactions > 800:  # High peak volume
                scaled_hours *= role_config['peak_hour_multiplier']

            # Apply environmental factors
            if environmental_data['weather'] == 'rainy':
                scaled_hours *= 1.04  # Slightly more staff needed when busy

            if environmental_data['event_impact'] > 1.2:
                scaled_hours *= 1.07  # Special events require more coordination

            # Apply efficiency factors
            staff_mix_factor = 0.82  # Assuming mix of experienced and student workers
            if period_info['period_name'] == 'finals_week':
                staff_mix_factor *= 0.91  # Student workers less available
            elif period_info['period_name'] == 'summer_session':
                staff_mix_factor *= 0.76  # Fewer student workers available

            adjusted_hours = scaled_hours / staff_mix_factor

            # Apply minimum coverage requirements
            final_hours = max(role_config['minimum_coverage_hours'], adjusted_hours)

            # Apply operational constraints
            facility_capacity = self.config['facility_specs']['maximum_simultaneous_capacity']
            capacity_utilization = min(1.0, total_transactions / (facility_capacity * 2.5))  # 2.5 turns per day

            if capacity_utilization > 0.9:  # Near capacity
                final_hours *= 1.08  # Need more staff for crowd management

            staffing_requirements[f'actual_{role.lower()}'] = round(final_hours, 1)

        return staffing_requirements

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate comprehensive dataset for the specified date range
        """
        start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')

        total_days = (end_date - start_date).days + 1
        data = []
        current_date = start_date

        print(f"\n🏗️  Generating {total_days:,} days of Centerpointe dining data...")
        print(f"📅 Date range: {start_date.strftime('%B %d, %Y')} → {end_date.strftime('%B %d, %Y')}")

        # Progress tracking
        progress_milestones = [int(total_days * p) for p in [0.25, 0.5, 0.75, 1.0]]
        days_processed = 0

        while current_date <= end_date:
            days_processed += 1

            # Progress reporting
            if days_processed in progress_milestones:
                progress = (days_processed / total_days) * 100
                print(f"⏳ Progress: {progress:.0f}% ({days_processed:,}/{total_days:,} days)")

            # Basic date information
            day_of_week = current_date.weekday()
            is_weekend = day_of_week >= 5

            # Get academic period information
            period_info = self._get_academic_period_info(current_date)

            # Calculate student population for this date
            population_data = self._calculate_student_population(current_date)

            # Generate environmental conditions
            environmental_data = self._generate_environmental_conditions(current_date)

            # Calculate transaction data
            transaction_data = self._calculate_base_transactions(
                current_date, population_data, period_info, environmental_data
            )

            # Calculate staffing requirements
            staffing_data = self._calculate_staffing_requirements(
                current_date, transaction_data, period_info, environmental_data
            )

            # Calculate financial metrics
            avg_transaction_value = 13.25  # All-you-care-to-eat effective value
            daily_revenue = transaction_data['total_transactions'] * avg_transaction_value

            total_labor_hours = sum(staffing_data.values())
            avg_hourly_rate = self.config['staffing_model']['labor_costs']['average_hourly_rate']
            labor_cost = total_labor_hours * avg_hourly_rate

            # Create comprehensive daily record
            record = {
                # Date and Time Information
                'date': current_date.strftime('%Y-%m-%d'),
                'day_of_week': day_of_week,
                'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week],
                'is_weekend': is_weekend,
                'month': current_date.month,
                'year': current_date.year,
                'day_of_year': current_date.timetuple().tm_yday,
                'week_of_year': current_date.isocalendar()[1],

                # Academic Calendar Information
                'academic_period': period_info['period_name'],
                'period_type': period_info['period_type'],
                'seasonal_multiplier': period_info['multiplier'],
                'period_description': period_info['description'],

                # Student Population Data
                'total_enrollment': population_data['total_enrollment'],
                'active_enrollment': population_data['active_enrollment'],
                'residential_students': population_data['residential_students'],
                'commuter_students': population_data['commuter_students'],
                'total_meal_plan_holders': population_data['total_meal_plan_holders'],
                'enrollment_seasonal_factor': population_data['seasonal_factor'],

                # Environmental Factors
                'weather': environmental_data['weather'],
                'weather_impact': environmental_data['weather_impact'],
                'campus_event': environmental_data['campus_event'],
                'event_impact': environmental_data['event_impact'],
                'event_scheduled': environmental_data['event_scheduled'],

                # Transaction Data
                'total_transactions': transaction_data['total_transactions'],
                'guest_transactions': transaction_data['guest_transactions'],
                'base_transaction_rate': round(transaction_data['base_transaction_rate'], 4),

                # Payment Method Breakdown
                'meal_swipes': transaction_data['payment_breakdown']['meal_swipes'],
                'dining_dollars_transactions': transaction_data['payment_breakdown']['dining_dollars'],
                'bronco_bucks_transactions': transaction_data['payment_breakdown']['bronco_bucks'],
                'credit_debit_transactions': transaction_data['payment_breakdown']['credit_debit'],

                # Meal Period Distribution
                'breakfast_transactions': transaction_data['meal_period_breakdown']['breakfast'],
                'lunch_transactions': transaction_data['meal_period_breakdown']['lunch'],
                'dinner_transactions': transaction_data['meal_period_breakdown']['dinner'],
                'late_night_transactions': transaction_data['meal_period_breakdown']['late_night'],

                # Financial Metrics
                'estimated_daily_revenue': round(daily_revenue, 2),
                'avg_transaction_value': avg_transaction_value,
                'labor_cost_actual': round(labor_cost, 2),

                # Operational Metrics
                'transactions_per_meal_plan_holder': round(
                    transaction_data['total_transactions'] / max(1, population_data['total_meal_plan_holders']), 3
                ),
                'facility_capacity_utilization': min(1.0,
                                                     transaction_data['total_transactions'] / (self.config['facility_specs']['maximum_simultaneous_capacity'] * 2.5)
                                                     ),
                'peak_meal_period_volume': max(transaction_data['meal_period_breakdown'].values()),
            }

            # Add staffing requirements
            record.update(staffing_data)
            record['total_actual_hours'] = round(total_labor_hours, 1)

            # Add derived performance metrics
            if total_labor_hours > 0:
                record['revenue_per_labor_hour'] = round(daily_revenue / total_labor_hours, 2)
                record['transactions_per_labor_hour'] = round(transaction_data['total_transactions'] / total_labor_hours, 2)
                record['labor_cost_percentage'] = round((labor_cost / max(daily_revenue, 1)) * 100, 1)
            else:
                record['revenue_per_labor_hour'] = 0.0
                record['transactions_per_labor_hour'] = 0.0
                record['labor_cost_percentage'] = 0.0

            data.append(record)
            current_date += timedelta(days=1)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Final statistics and validation
        print(f"\n✅ Dataset generation complete!")
        print(f"📊 Generated {len(df):,} days of operational data")
        print(f"📈 Date range: {df['date'].min()} to {df['date'].max()}")

        print(f"\n📋 Key Statistics:")
        print(f"   Average daily transactions: {df['total_transactions'].mean():,.0f}")
        print(f"   Peak daily transactions: {df['total_transactions'].max():,}")
        print(f"   Average daily revenue: ${df['estimated_daily_revenue'].mean():,.2f}")
        print(f"   Average staffing hours: {df['total_actual_hours'].mean():.1f}")
        print(f"   Peak staffing hours: {df['total_actual_hours'].max():.1f}")

        # Academic period breakdown
        print(f"\n🎓 Academic Period Distribution:")
        period_summary = df.groupby('academic_period').agg({
            'total_transactions': ['count', 'mean'],
            'total_actual_hours': 'mean',
            'estimated_daily_revenue': 'mean'
        }).round(1)

        for period in period_summary.index:
            days = period_summary.loc[period, ('total_transactions', 'count')]
            avg_transactions = period_summary.loc[period, ('total_transactions', 'mean')]
            avg_hours = period_summary.loc[period, ('total_actual_hours', 'mean')]
            avg_revenue = period_summary.loc[period, ('estimated_daily_revenue', 'mean')]

            print(f"   {period.replace('_', ' ').title()}: {days} days, "
                  f"{avg_transactions:.0f} trans/day, {avg_hours:.1f} hrs/day, ${avg_revenue:,.0f}/day")

        return df

    def save_config(self, filepath: str):
        """Save current configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        print(f"Configuration saved to {filepath}")

    def load_config(self, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)
        self._update_config_recursively(self.config, loaded_config)
        print(f"Configuration loaded from {filepath}")

# Example usage:
# growth_config = {
#     'start_date': '2024-01-01',
#     'end_date': '2025-07-30',
#     'student_population': {
#         'yoy_growth_rate': 0.028,  # Higher growth scenario
#         'residential_student_ratio': 0.17,  # Planned housing expansion
#     }
# }
# generator = CenterpointeDiningDataGenerator(growth_config)
# df = generator.generate_dataset()
# df
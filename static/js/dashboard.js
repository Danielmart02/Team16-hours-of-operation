// Dashboard JavaScript for Cal Poly Pomona Dining
class DiningDashboard {
    constructor() {
        this.chart = null;
        this.detailedChart = null;
        this.eventOptions = [];
        this.currentPredictions = [];
        this.selectedDate = null;
        this.init();
    }

    async init() {
        // Ensure detailed analysis is hidden on load
        this.closeDetailedView();
        
        await this.loadEventOptions();
        this.setupEventListeners();
        this.setDefaultDates();
        await this.loadTomorrowSummary();
    }

    async loadEventOptions() {
        try {
            // Load event options
            const eventResponse = await fetch('/api/event-options');
            this.eventOptions = await eventResponse.json();
            this.populateSelect('event-select', this.eventOptions, 'regular_day');
        } catch (error) {
            console.error('Error loading options:', error);
            this.showError('Failed to load dropdown options');
        }
    }

    populateSelect(selectId, options, defaultValue = null) {
        const select = document.getElementById(selectId);
        select.innerHTML = '';
        
        options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = this.formatOptionText(option);
            if (option === defaultValue) {
                optionElement.selected = true;
            }
            select.appendChild(optionElement);
        });
    }

    formatOptionText(option) {
        // Convert snake_case to Title Case
        return option.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    setDefaultDates() {
        const today = new Date();
        const nextWeek = new Date(today);
        nextWeek.setDate(today.getDate() + 7);

        document.getElementById('start-date').value = this.formatDate(today);
        document.getElementById('end-date').value = this.formatDate(nextWeek);
    }

    formatDate(date) {
        return date.toISOString().split('T')[0];
    }

    setupEventListeners() {
        document.getElementById('predict-btn').addEventListener('click', () => {
            this.generateSimplePredictions();
        });

        document.getElementById('close-detailed').addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.closeDetailedView();
        });

        document.getElementById('update-detailed').addEventListener('click', () => {
            this.updateDetailedPrediction();
        });

        // Allow Enter key to trigger predictions
        document.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !document.getElementById('detailed-analysis').classList.contains('hidden')) {
                // If detailed view is open, update detailed prediction
                this.updateDetailedPrediction();
            } else if (e.key === 'Enter') {
                // Otherwise, generate simple predictions
                this.generateSimplePredictions();
            }
        });

        // Allow Escape key to close detailed view
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeDetailedView();
            }
        });
    }

    async loadTomorrowSummary() {
        try {
            const response = await fetch('/api/tomorrow-summary');
            const summary = await response.json();

            if (summary.error) {
                throw new Error(summary.error);
            }

            document.getElementById('workers-needed').textContent = summary.total_workers_needed;
            document.getElementById('people-expected').textContent = summary.people_expected.toLocaleString();
            document.getElementById('total-hours').textContent = `${summary.total_hours}h`;

            // Show weather note if weather data was used
            const weatherNote = document.getElementById('weather-note');
            if (summary.weather_from_api) {
                weatherNote.classList.remove('hidden');
            } else {
                weatherNote.classList.add('hidden');
            }
        } catch (error) {
            console.error('Error loading tomorrow summary:', error);
            document.getElementById('workers-needed').textContent = 'Error';
            document.getElementById('people-expected').textContent = 'Error';
            document.getElementById('total-hours').textContent = 'Error';
        }
    }

    async generateSimplePredictions() {
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;

        if (!startDate || !endDate) {
            this.showError('Please fill in both start and end dates');
            return;
        }

        if (new Date(startDate) > new Date(endDate)) {
            this.showError('Start date must be before end date');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch('/api/simple-batch-predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    start_date: startDate,
                    end_date: endDate
                })
            });

            const predictions = await response.json();

            if (predictions.error) {
                throw new Error(predictions.error);
            }

            this.currentPredictions = predictions;
            this.updateChart(predictions);
            
            // Close detailed view if it's open
            this.closeDetailedView();
        } catch (error) {
            console.error('Error generating predictions:', error);
            this.showError('Failed to generate predictions: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    updateChart(predictions) {
        const ctx = document.getElementById('staffing-chart').getContext('2d');

        // Prepare data for stacked bar chart
        const labels = predictions.map(p => {
            const date = new Date(p.date);
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        });

        const workerTypes = [
            'General Purpose Worker',
            'Cashier',
            'Chef',
            'Line Workers',
            'Dishwasher',
            'Management'
        ];

        const colors = [
            '#FF6B6B',
            '#4ECDC4',
            '#45B7D1',
            '#96CEB4',
            '#FFEAA7',
            '#DDA0DD'
        ];

        const datasets = workerTypes.map((workerType, index) => ({
            label: workerType,
            data: predictions.map(p => p.workers[workerType] || 0),
            backgroundColor: colors[index],
            borderColor: colors[index],
            borderWidth: 1
        }));

        // Destroy existing chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }

        // Create new chart
        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true,
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        stacked: true,
                        title: {
                            display: true,
                            text: 'Hours Required'
                        },
                        beginAtZero: true
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Staffing Predictions Overview (Date-Based Only)',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: false // We have our custom legend
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            title: function(tooltipItems) {
                                const index = tooltipItems[0].dataIndex;
                                const prediction = predictions[index];
                                return `${new Date(prediction.date).toLocaleDateString('en-US', { 
                                    weekday: 'long', 
                                    year: 'numeric', 
                                    month: 'long', 
                                    day: 'numeric' 
                                })}`;
                            },
                            afterTitle: function(tooltipItems) {
                                const index = tooltipItems[0].dataIndex;
                                const prediction = predictions[index];
                                return [
                                    `Expected Customers: ${prediction.predicted_transactions.toLocaleString()}`,
                                    `Total Hours: ${prediction.total_predicted_hours}h`,
                                    '',
                                    '🖱️ Click for detailed analysis with weather & events'
                                ];
                            },
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}h`;
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                onClick: (event, elements) => {
                    console.log('Chart clicked, elements:', elements); // Debug log
                    if (elements.length > 0) {
                        const index = elements[0].index;
                        const selectedPrediction = predictions[index];
                        console.log('Selected prediction:', selectedPrediction); // Debug log
                        
                        if (selectedPrediction && selectedPrediction.date) {
                            this.openDetailedView(selectedPrediction.date);
                        } else {
                            console.error('Invalid prediction data:', selectedPrediction);
                        }
                    } else {
                        console.log('No chart elements clicked');
                    }
                }
            }
        });
    }

    async openDetailedView(date) {
        console.log('Opening detailed view for date:', date); // Debug log
        
        this.selectedDate = date;
        
        // Check if detailed analysis element exists
        const detailedSection = document.getElementById('detailed-analysis');
        if (!detailedSection) {
            console.error('Detailed analysis section not found!');
            return;
        }
        
        // Update the detailed view header
        const dateObj = new Date(date);
        const detailedDateElement = document.getElementById('detailed-date');
        if (detailedDateElement) {
            detailedDateElement.textContent = 
                dateObj.toLocaleDateString('en-US', { 
                    weekday: 'long', 
                    year: 'numeric', 
                    month: 'long', 
                    day: 'numeric' 
                });
        }

        // Reset event selection to default
        const eventSelect = document.getElementById('event-select');
        if (eventSelect) {
            eventSelect.value = 'regular_day';
        }

        // Show the detailed analysis section
        console.log('Removing hidden class from detailed section'); // Debug log
        detailedSection.classList.remove('hidden');
        detailedSection.style.display = 'block'; // Force display
        
        // Scroll to the detailed section
        setTimeout(() => {
            detailedSection.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
        }, 100);

        // Load initial detailed prediction
        await this.updateDetailedPrediction();
    }

    async updateDetailedPrediction() {
        if (!this.selectedDate) return;

        const event = document.getElementById('event-select').value;
        
        this.showLoading(true);

        try {
            const response = await fetch('/api/detailed-predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    date: this.selectedDate,
                    event: event
                })
            });

            const prediction = await response.json();

            if (prediction.error) {
                throw new Error(prediction.error);
            }

            this.updateDetailedView(prediction);
        } catch (error) {
            console.error('Error getting detailed prediction:', error);
            this.showError('Failed to get detailed prediction: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    updateDetailedView(prediction) {
        // Update info cards
        document.getElementById('detailed-customers').textContent = 
            prediction.predicted_transactions.toLocaleString();
        document.getElementById('detailed-hours').textContent = 
            `${prediction.total_predicted_hours}h`;
        document.getElementById('detailed-weather').textContent = 
            this.formatOptionText(prediction.weather);

        // Show/hide weather note based on whether weather data came from API
        const weatherNote = document.getElementById('detailed-weather-note');
        if (prediction.weather_from_api) {
            weatherNote.classList.remove('hidden');
        } else {
            weatherNote.classList.add('hidden');
        }

        // Update detailed chart
        this.updateDetailedChart(prediction);
    }

    updateDetailedChart(prediction) {
        const ctx = document.getElementById('detailed-chart').getContext('2d');

        const workerTypes = Object.keys(prediction.workers);
        const hours = Object.values(prediction.workers);
        
        const colors = [
            '#FF6B6B',
            '#4ECDC4',
            '#45B7D1',
            '#96CEB4',
            '#FFEAA7',
            '#DDA0DD'
        ];

        // Destroy existing detailed chart if it exists
        if (this.detailedChart) {
            this.detailedChart.destroy();
        }

        // Create detailed chart
        this.detailedChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: workerTypes,
                datasets: [{
                    data: hours,
                    backgroundColor: colors.slice(0, workerTypes.length),
                    borderColor: colors.slice(0, workerTypes.length),
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Detailed Analysis - ${this.formatOptionText(prediction.event)} ${prediction.weather_from_api ? '(Weather Included)' : ''}`,
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const hours = context.parsed;
                                const workers = Math.max(1, Math.round(hours / 8));
                                return `${context.label}: ${hours.toFixed(1)}h (≈${workers} workers)`;
                            }
                        }
                    }
                }
            }
        });
    }

    closeDetailedView() {
        console.log('Closing detailed view'); // Debug log
        
        const detailedSection = document.getElementById('detailed-analysis');
        if (!detailedSection) {
            console.error('Detailed analysis section not found when trying to close!');
            return;
        }
        
        // Ensure the section is properly hidden
        detailedSection.classList.add('hidden');
        detailedSection.style.display = 'none';
        
        // Reset selected date
        this.selectedDate = null;
        
        // Destroy detailed chart if it exists
        if (this.detailedChart) {
            this.detailedChart.destroy();
            this.detailedChart = null;
        }
        
        // Clear any form data
        const eventSelect = document.getElementById('event-select');
        if (eventSelect) {
            eventSelect.value = 'regular_day';
        }
        
        console.log('Detailed view closed successfully'); // Debug log
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }

    showError(message) {
        alert('Error: ' + message);
    }

    // Helper method to check if a date is within 7 days from today
    isWithinSevenDays(dateString) {
        const today = new Date();
        const targetDate = new Date(dateString);
        const diffTime = targetDate - today;
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        return diffDays >= 0 && diffDays <= 7;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DiningDashboard();
});

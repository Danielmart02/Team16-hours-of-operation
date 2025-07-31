// Dashboard JavaScript for Cal Poly Pomona Dining
class DiningDashboard {
    constructor() {
        this.chart = null;
        this.weatherOptions = [];
        this.eventOptions = [];
        this.init();
    }

    async init() {
        await this.loadOptions();
        this.setupEventListeners();
        this.setDefaultDates();
        await this.loadTodaySummary();
    }

    async loadOptions() {
        try {
            // Load weather options
            const weatherResponse = await fetch('/api/weather-options');
            this.weatherOptions = await weatherResponse.json();
            this.populateSelect('weather-select', this.weatherOptions, 'sunny');

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
            this.generatePredictions();
        });

        // Allow Enter key to trigger predictions
        document.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.generatePredictions();
            }
        });
    }

    async loadTodaySummary() {
        try {
            const response = await fetch('/api/today-summary');
            const summary = await response.json();

            if (summary.error) {
                throw new Error(summary.error);
            }

            document.getElementById('workers-needed').textContent = summary.total_workers_needed;
            document.getElementById('people-expected').textContent = summary.people_expected.toLocaleString();
            document.getElementById('total-hours').textContent = `${summary.total_hours}h`;
        } catch (error) {
            console.error('Error loading today summary:', error);
            document.getElementById('workers-needed').textContent = 'Error';
            document.getElementById('people-expected').textContent = 'Error';
            document.getElementById('total-hours').textContent = 'Error';
        }
    }

    async generatePredictions() {
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        const weather = document.getElementById('weather-select').value;
        const event = document.getElementById('event-select').value;

        if (!startDate || !endDate || !weather || !event) {
            this.showError('Please fill in all fields');
            return;
        }

        if (new Date(startDate) > new Date(endDate)) {
            this.showError('Start date must be before end date');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch('/api/batch-predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    start_date: startDate,
                    end_date: endDate,
                    weather: weather,
                    event: event
                })
            });

            const predictions = await response.json();

            if (predictions.error) {
                throw new Error(predictions.error);
            }

            this.updateChart(predictions);
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
                        text: `Staffing Predictions - ${this.formatOptionText(document.getElementById('weather-select').value)} Weather, ${this.formatOptionText(document.getElementById('event-select').value)}`,
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
                                    `Weather: ${dashboard.formatOptionText(prediction.weather)}`,
                                    `Event: ${dashboard.formatOptionText(prediction.event)}`,
                                    `Expected Customers: ${prediction.predicted_transactions.toLocaleString()}`,
                                    `Total Hours: ${prediction.total_predicted_hours}h`
                                ];
                            },
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}h`;
                            },
                            footer: function(tooltipItems) {
                                const index = tooltipItems[0].dataIndex;
                                const prediction = predictions[index];
                                let totalWorkers = 0;
                                Object.values(prediction.workers).forEach(hours => {
                                    totalWorkers += Math.max(1, Math.round(hours / 8));
                                });
                                return `Estimated Workers Needed: ${totalWorkers}`;
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'index',
                    intersect: false
                }
            }
        });
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
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DiningDashboard();
});

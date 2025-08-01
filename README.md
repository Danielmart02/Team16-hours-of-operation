# Cal Poly Pomona Dining Dashboard

A sophisticated web-based dashboard for predicting staffing requirements at Cal Poly Pomona's dining facilities. Features real-time predictions, interactive visualizations, and an AI-powered assistant for intelligent insights.

## 🚀 Features

### **Core Functionality**
- **Real-time Predictions**: Get staffing predictions based on weather, events, and dates
- **Interactive Dashboard**: Visual representation of staffing needs with stacked bar charts
- **Quick Information**: Today's summary with worker count, expected customers, and total hours
- **Batch Predictions**: Generate predictions for date ranges
- **Detailed Analysis**: Click on chart bars for in-depth breakdowns with weather and event scenarios

### **AI-Powered Assistant**
- **Intelligent Chat Interface**: AI agent powered by AWS Bedrock and LangChain
- **Contextual Responses**: Understands staffing data, weather impacts, and event considerations
- **Historical Data Access**: Query past performance and trends
- **Prediction Explanations**: Get insights into why certain staffing levels are recommended
- **Natural Language Queries**: Ask questions in plain English about dining operations

### **Enhanced User Interface**
- **Modern Design**: Cal Poly Pomona branded interface with smooth animations
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Resizable Chat Widget**: Drag to resize or use full-screen mode for extended conversations
- **Interactive Charts**: Click-to-explore functionality with hover details
- **Accessibility Features**: Keyboard navigation, focus indicators, and screen reader support

## 🤖 AI Assistant Capabilities

The integrated AI assistant can help with:

- **Staffing Predictions**: "What staffing do I need for next week if it's sunny with no events?"
- **Weather Impact Analysis**: "How does rainy weather affect dining traffic?"
- **Event Planning**: "How many extra workers do I need for graduation?"
- **Historical Insights**: "Show me data from last month's career fair"
- **Operational Guidance**: "How should I schedule my employees for optimal coverage?"

### **Chat Features**
- **Resizable Interface**: Drag from bottom-left corner to resize
- **Full-Screen Mode**: Press F11 or click the expand button for immersive conversations
- **Minimize Option**: Collapse to header-only view
- **Keyboard Shortcuts**: F11 (fullscreen), Escape (close), Enter (send)
- **Smart Fallbacks**: Helpful responses even when AI agent is unavailable

## 📊 Worker Types Predicted

- **General Purpose Worker** (FOH_General)
- **Cashier** (FOH_Cashier)
- **Chef** (Kitchen_Prep)
- **Line Workers** (Kitchen_Line)
- **Dishwasher** (Dish_Room)
- **Management** (Management)

## 🌤️ Weather Conditions

- Sunny
- Cloudy
- Rainy
- Extreme Heat

## 🎓 Campus Events

- Regular Day
- Club Fair
- Career Fair
- Sports Events
- Graduation
- Parent Weekend
- Prospective Student Day
- Conference Hosting
- Campus Construction

## 🛠️ Setup Instructions

### **Prerequisites**
- Python 3.8 or higher
- AWS Account with Bedrock access (for AI features)
- AWS CLI configured with appropriate credentials

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. AWS Configuration (Required for AI Assistant)**
```bash
# Install AWS CLI if not already installed
pip install awscli

# Configure your AWS credentials
aws configure
```

You'll need:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-west-2`)
- Bedrock model access for Claude 3.5 Sonnet

### **3. Enable AWS Bedrock Access**
1. Go to AWS Bedrock console
2. Navigate to "Model access" in the left sidebar
3. Request access to "Claude 3.5 Sonnet" model
4. Wait for approval (usually immediate for most accounts)

### **4. Verify Model Files**
Ensure these files are present in your project directory:
- `tx_model.pkl` - Transaction prediction model
- `work_model.pkl` - Staffing prediction model
- `df.csv` - Historical data file
- `inference.py` - Prediction logic

### **5. Run the Application**
```bash
python app.py
```

### **6. Access the Dashboard**
Open your browser and go to `http://localhost:5000`

## 🔧 Configuration Options

### **Environment Variables**
```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2

# Optional: Custom model paths
export TX_MODEL_PATH=./tx_model.pkl
export WORK_MODEL_PATH=./work_model.pkl
export DATA_FILE_PATH=./df.csv
```

### **Agent Configuration**
The AI agent can be customized in `app.py`:
```python
dining_agent = DiningHallAgent(
    enable_tracing=False,
    recursion_limit=50,  # Adjust for complex queries
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)
```

## 🌐 API Endpoints

### **Dashboard APIs**
- `GET /` - Main dashboard page
- `GET /api/weather-options` - Get available weather conditions
- `GET /api/event-options` - Get available campus events
- `POST /api/predict` - Single day prediction
- `POST /api/batch-predict` - Date range predictions
- `GET /api/today-summary` - Quick summary for today
- `POST /api/simple-batch-predict` - Simplified batch predictions
- `POST /api/detailed-predict` - Detailed single-day analysis

### **AI Chat APIs**
- `POST /api/chat` - Send message to AI assistant
- `POST /api/chat/clear` - Clear conversation history
- `POST /api/chat/reset` - Reset agent state
- `GET /api/chat/status` - Check agent availability

## 💻 Usage Guide

### **Quick Start**
1. **View Today's Predictions**: Automatically loaded on dashboard load
2. **Generate Custom Predictions**: 
   - Select start and end dates
   - Click "Generate Predictions"
3. **Explore Details**: Click on any chart bar for detailed analysis
4. **Chat with AI**: Click the chat button (🤖) in bottom-right corner

### **Advanced Features**
- **Resize Chat**: Drag the bottom-left corner of the chat window
- **Full-Screen Chat**: Press F11 or click the expand button (⛶)
- **Keyboard Navigation**: Use Tab, Enter, and Escape keys
- **Mobile Support**: Fully responsive design for all devices

### **AI Assistant Tips**
- Ask specific questions about dates, weather, or events
- Request explanations for predictions
- Inquire about historical trends
- Get operational recommendations
- Use natural language - no special syntax required

## 📁 File Structure

```
├── app.py                    # Flask application with AI integration
├── inference.py              # Prediction logic
├── dining_agent.py           # AI agent implementation
├── cpp_agent.py              # Agent initialization
├── dataset_generator.py      # Data generation utilities
├── tx_model.pkl             # Transaction prediction model
├── work_model.pkl           # Staffing prediction model
├── df.csv                   # Historical data
├── requirements.txt         # Python dependencies
├── templates/
│   └── dashboard.html       # Enhanced dashboard template
├── static/
│   ├── css/
│   │   └── style.css        # Modern styling with Cal Poly colors
│   ├── js/
│   │   ├── dashboard.js     # Dashboard functionality
│   │   └── chat.js          # AI chat widget
│   └── images/
│       └── cpp_logo.svg     # Cal Poly Pomona logo
└── README.md               # This file
```

## 🎨 Design System

### **Color Scheme**
The dashboard uses Cal Poly Pomona's official colors:
- **Primary Green**: #1B5E20
- **Accent Green**: #4CAF50
- **Highlight Yellow**: #FFC107
- **Background**: #F8F9FA
- **Text**: #495057

### **Typography**
- **Font Stack**: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto'
- **Responsive Sizing**: Scales appropriately across devices
- **Accessibility**: High contrast ratios and readable font sizes

## 🔧 Technologies Used

### **Backend**
- **Flask**: Web framework
- **Python**: Core language
- **AWS Bedrock**: AI model hosting
- **LangChain**: AI agent framework
- **LangGraph**: Agent workflow management

### **Frontend**
- **HTML5**: Modern markup
- **CSS3**: Advanced styling with Grid and Flexbox
- **JavaScript**: Interactive functionality
- **Chart.js**: Data visualization

### **Machine Learning**
- **XGBoost**: Prediction models
- **scikit-learn**: Data processing
- **pandas**: Data manipulation
- **numpy**: Numerical computations

### **AI & NLP**
- **AWS Bedrock**: Claude 3.5 Sonnet model
- **LangChain**: Agent framework
- **boto3**: AWS SDK

## 🚨 Troubleshooting

### **Common Issues**

#### **AI Agent Not Working**
```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify Bedrock access
aws bedrock list-foundation-models --region us-west-2
```

#### **Recursion Limit Errors**
- The system automatically handles these with fallback responses
- Clear chat history if issues persist
- Use `/api/chat/reset` endpoint to reset agent state

#### **Model Files Missing**
Ensure all `.pkl` files are in the project root directory

#### **Port Already in Use**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use a different port
python app.py --port 5001
```

### **Performance Optimization**
- Clear chat history periodically for better performance
- Use specific date ranges rather than very large ranges
- Monitor AWS Bedrock usage to manage costs

## 📈 Future Enhancements

- **Multi-location Support**: Extend to other dining facilities
- **Advanced Analytics**: Trend analysis and forecasting
- **Mobile App**: Native iOS/Android applications
- **Integration APIs**: Connect with existing campus systems
- **Real-time Updates**: Live data feeds and notifications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is developed for Cal Poly Pomona's dining services. Please contact the development team for usage permissions.

## 📞 Support

For technical support or questions:
- Check the troubleshooting section above
- Review AWS Bedrock documentation for AI-related issues
- Ensure all dependencies are properly installed
- Verify AWS credentials and permissions

---

**Built with ❤️ for Cal Poly Pomona Dining Services**

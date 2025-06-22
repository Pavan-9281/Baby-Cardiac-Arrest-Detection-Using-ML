# Cardiac Arrest Prediction System

A Django-based machine learning application for predicting cardiac arrest risk using health parameters and statistical models.

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- pip (Python package installer)

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd a_machine_learning_approach_using_statistical_models
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install required packages:**
   ```bash
   pip install Django>=5.0.2
   pip install pandas>=2.0.0
   pip install scikit-learn>=1.2.0
   pip install numpy>=1.22.0
   ```

5. **Run database migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

7. **Access the application:**
   Open your browser and go to: http://127.0.0.1:8000/

## 👥 Default Users

### Service Provider (Admin)
- **Username:** Admin
- **Password:** Admin

### Sample Users
- **Username:** Gopalan | **Password:** Gopalan
- **Username:** Manjunath | **Password:** Manjunath

## 📊 Features

### For Users
- User registration and login
- Cardiac arrest risk prediction
- View prediction history
- Profile management

### For Service Providers (Admin)
- User management
- Model training with datasets
- Analytics and charts
- Prediction ratio analysis
- User training comparison

## 🎯 How to Use

### Making Predictions
1. Login as a user
2. Navigate to "Predict Cardiac Arrest Type"
3. Enter health parameters:
   - Age in days
   - Sex (M/F)
   - Chest pain type
   - Resting blood pressure
   - Resting ECG results
   - Maximum heart rate
   - Exercise angina
   - Oldpeak
   - ST slope
   - Other parameters (slp, caa, thall)

### Training Models (Admin)
1. Login as Admin
2. Go to "Browse and Train & Test Traffic Data Sets"
3. Select a user for training
4. The system will train multiple ML models:
   - Random Forest
   - Logistic Regression
   - Support Vector Machine
   - Decision Tree

### Viewing Analytics
- **Prediction Ratios:** View distribution of cardiac arrest predictions
- **Model Accuracy:** Compare performance of different ML models
- **User Training Comparison:** Analyze training results by user

## 📁 Project Structure

```
a_machine_learning_approach_using_statistical_models/
├── Remote_User/                 # User-facing application
│   ├── models.py               # User and prediction models
│   ├── views.py                # User views and prediction logic
│   └── templates/              # User interface templates
├── Service_Provider/           # Admin application
│   ├── models.py               # Admin models
│   ├── views.py                # Admin views and analytics
│   └── templates/              # Admin interface templates
├── Template/                   # Static assets
│   ├── htmls/                  # HTML templates
│   └── images/                 # Image assets
├── Datasets.csv               # Training dataset
├── manage.py                  # Django management script
└── settings.py                # Django settings
```

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'sklearn'**
   ```bash
   pip install scikit-learn
   ```

2. **Database migration errors**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

3. **Port already in use**
   ```bash
   python manage.py runserver 8001
   ```

### Python Version Compatibility
- This project is optimized for Python 3.11+
- If using Python 3.13, ensure Django 5.0+ is installed

## 📈 Machine Learning Models

The system uses ensemble learning with multiple algorithms:
- **Random Forest Classifier**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**

All models are combined using a Voting Classifier for improved prediction accuracy.

## 🎨 UI Features

- Modern glassmorphism design
- Interactive charts using Chart.js and CanvasJS
- Responsive design for mobile and desktop
- Animated elements and smooth transitions
- Purple-blue gradient theme

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions, please create an issue in the repository.

---

**Note:** This is a demonstration project for cardiac arrest prediction using machine learning. Always consult healthcare professionals for medical decisions. 
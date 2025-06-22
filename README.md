# Cardiac Arrest Prediction using Machine Learning

A Django web application that predicts the risk of cardiac arrest using machine learning models trained on real health data. The project features user and admin dashboards, modern UI, interactive charts, and supports user-specific model training and analytics.

## Features
- Predict cardiac arrest risk based on health parameters
- User registration, login, and profile management
- Service provider (admin) dashboard for analytics and model training
- Interactive, animated charts and modern glassmorphism UI
- User-specific training and comparison analytics
- Data visualization for prediction ratios and model accuracy

## Tech Stack
- Python 3.11+
- Django 5.0+
- pandas, scikit-learn, numpy
- HTML5, CSS3, JavaScript (Chart.js, CanvasJS)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name/project-main
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply migrations:**
   ```bash
   cd a_machine_learning_approach_using_statistical_models
   python manage.py migrate
   ```

5. **Run the development server:**
   ```bash
   python manage.py runserver
   ```

6. **Access the app:**
   Open your browser and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## Default Login Credentials
- **Service Provider (Admin):**
  - Username: `Admin`
  - Password: `Admin`
- **Sample Users:**
  - Username: `Gopalan` / Password: `Gopalan`
  - Username: `Manjunath` / Password: `Manjunath`

## Usage
- Register as a new user or login with the provided credentials.
- For service provider: login as Admin to access analytics, train models, and view user statistics.
- For users: predict cardiac arrest risk, view your profile, and see your prediction history.
- Use the "Browse and Train & Test Traffic Data Sets" option to train models with the provided dataset (`Datasets.csv`).
- View interactive charts for prediction ratios and model accuracy.

## Project Structure
- `a_machine_learning_approach_using_statistical_models/` - Main Django project
- `Remote_User/` - User-facing app
- `Service_Provider/` - Admin/service provider app
- `Template/` - HTML templates and static assets
- `Datasets.csv` - Health dataset for training and prediction

## Screenshots
_Add screenshots of the UI and charts here._

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE) (or specify your license) 
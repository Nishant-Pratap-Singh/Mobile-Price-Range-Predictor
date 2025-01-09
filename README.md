# Mobile-Price-Range-Predictor
A machine learning project that predicts the price range of mobile phones based on key features such as RAM, screen resolution, battery power, internal memory, and more. This project utilizes classification techniques to analyze and categorize mobile phones into predefined price ranges.
Check it out : https://nps-mobile-price-range-predictor.streamlit.app/
#  🛠 Features
1. **Dataset Exploration**:
- Visualize correlations and trends in the dataset.
- Graphs showcasing relationships between price range and features like RAM, Pixel Width, Battery Power, etc.
- Inter-correlation heatmap for feature analysis.
2. **Mobile Price Prediction**:
- Predicts the price range (0-3: Low to High) of mobile phones using Support Vector Machines (SVC).
- Interactive sliders to input feature values (e.g., Battery Power, RAM, Pixel Dimensions).
3. **About Developer**:
Learn about the developer and find links to their social platforms.

# Tools and Technologies
1. **Frontend**: Streamlit for building an interactive web app.
2. **Backend**: Python (Machine Learning using scikit-learn).
3. **Dataset**: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
4. **Visualization**: Matplotlib, Seaborn for graphical insights.

# 📊 Dataset Overview
The dataset contains features like:

- battery_power: Total energy a battery can store (mAh).
- px_height & px_width: Pixel resolution of the screen.
- ram: Random Access Memory (MB).
- int_memory: Internal memory (GB).
- price_range: Target variable (0 = Low, 1 = Medium, 2 = High, 3 = Very High).
  
For a detailed description of all features, check the Dataset Exploration section of the app.
link - https://nps-mobile-price-range-predictor.streamlit.app/

# 📂 Repository Structure
Mobile Price Range Predictor/
- ├── app_new.py         # Streamlit app for data exploration and prediction
- ├── ml_model.py        # Machine learning model and helper functions
- ├── train.csv          # Dataset used for training and exploration
- ├── new1.jpg           # Image used in the Streamlit app
- ├── requirements.txt   # Python dependencies for running the app
- └── README.md          # Project documentation

# 📈 Machine Learning Model
1. **Algorithm**: Support Vector Machine (SVC)
2. **Evaluation**:
- Accuracy: 95.0%
- Cross-validation score: 94.43%

# 👤 About the Developer
Nishant Pratap Singh

- Email: npratapsingh084@gmail.com
- LinkedIn: [Nishant Pratap Singh](https://www.linkedin.com/in/nishant-pratap-singh-b96871257/)
- GitHub: [Nishant Pratap Singh](https://github.com/Nishant-Pratap-Singh)

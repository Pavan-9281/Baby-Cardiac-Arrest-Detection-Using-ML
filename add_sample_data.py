#!/usr/bin/env python3
"""
Simple script to add sample prediction data to the database
"""

import sqlite3
import os

def add_sample_data():
    # Connect to the SQLite database
    db_path = 'db.sqlite3'
    
    if not os.path.exists(db_path):
        print("Database file not found. Please run migrations first.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Clear existing prediction data
        cursor.execute("DELETE FROM Remote_User_cardiac_arrest_prediction")
        cursor.execute("DELETE FROM Remote_User_detection_ratio")
        cursor.execute("DELETE FROM Remote_User_detection_accuracy")
        
        # Add sample prediction data
        sample_predictions = [
            ('1', '45', 'M', 'ATA', '140', 'Normal', '150', 'N', '0.0', 'Up', '0', '0', '0', 'No Cardiac Arrest Found'),
            ('2', '52', 'F', 'NAP', '130', 'ST', '140', 'N', '1.5', 'Flat', '1', '1', '2', 'Cardiac Arrest Found'),
            ('3', '38', 'M', 'ASY', '160', 'LVH', '180', 'Y', '2.0', 'Down', '2', '2', '3', 'Cardiac Arrest Found'),
            ('4', '65', 'F', 'ATA', '120', 'Normal', '110', 'N', '0.0', 'Up', '0', '0', '1', 'No Cardiac Arrest Found'),
            ('5', '48', 'M', 'NAP', '145', 'ST', '155', 'Y', '1.0', 'Flat', '1', '1', '2', 'Cardiac Arrest Found'),
            ('6', '29', 'F', 'ASY', '135', 'Normal', '165', 'N', '0.5', 'Up', '0', '0', '0', 'No Cardiac Arrest Found'),
            ('7', '56', 'M', 'TA', '150', 'LVH', '140', 'Y', '2.5', 'Down', '2', '2', '3', 'Cardiac Arrest Found'),
            ('8', '42', 'F', 'ATA', '125', 'Normal', '145', 'N', '0.0', 'Up', '0', '0', '1', 'No Cardiac Arrest Found'),
            ('9', '61', 'M', 'NAP', '155', 'ST', '135', 'Y', '1.8', 'Flat', '1', '1', '2', 'Cardiac Arrest Found'),
            ('10', '35', 'F', 'ASY', '140', 'Normal', '160', 'N', '0.3', 'Up', '0', '0', '0', 'No Cardiac Arrest Found'),
        ]
        
        # Insert sample predictions
        for prediction in sample_predictions:
            cursor.execute("""
                INSERT INTO Remote_User_cardiac_arrest_prediction 
                (Fid, Age_In_Days, Sex, ChestPainType, RestingBP, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, slp, caa, thall, Prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, prediction)
        
        # Calculate and add ratio data
        cursor.execute("SELECT COUNT(*) FROM Remote_User_cardiac_arrest_prediction WHERE Prediction = 'No Cardiac Arrest Found'")
        no_arrest_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Remote_User_cardiac_arrest_prediction WHERE Prediction = 'Cardiac Arrest Found'")
        arrest_count = cursor.fetchone()[0]
        
        total_count = no_arrest_count + arrest_count
        
        if total_count > 0:
            no_arrest_ratio = (no_arrest_count / total_count) * 100
            arrest_ratio = (arrest_count / total_count) * 100
            
            cursor.execute("INSERT INTO Remote_User_detection_ratio (names, ratio) VALUES (?, ?)", 
                         ('No Cardiac Arrest Found', str(no_arrest_ratio)))
            cursor.execute("INSERT INTO Remote_User_detection_ratio (names, ratio) VALUES (?, ?)", 
                         ('Cardiac Arrest Found', str(arrest_ratio)))
        
        # Add sample accuracy data
        accuracy_data = [
            ('Artificial Neural Network (ANN)', '87.5'),
            ('SVM', '89.2'),
            ('Logistic Regression', '91.8'),
            ('Decision Tree Classifier', '85.3'),
        ]
        
        for accuracy in accuracy_data:
            cursor.execute("INSERT INTO Remote_User_detection_accuracy (names, ratio) VALUES (?, ?)", accuracy)
        
        conn.commit()
        print("✅ Sample data added successfully!")
        print(f"📊 Added {len(sample_predictions)} prediction records")
        print(f"📈 No Cardiac Arrest Found: {no_arrest_ratio:.1f}%")
        print(f"📈 Cardiac Arrest Found: {arrest_ratio:.1f}%")
        print("🎯 Model accuracy data added")
        
    except Exception as e:
        print(f"❌ Error adding sample data: {str(e)}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == "__main__":
    add_sample_data() 
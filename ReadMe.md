Project Overview:

This project aims to predict whether an employee will leave the company or not, assisting the HR department at Salifort Motors in identifying factors contributing to employee attrition and predicting which employees are likely to leave. These insights will help HR improve employee satisfaction and retention, thereby reducing the costs and time associated with hiring new employees.


PACE Strategy:

The project follows the PACE strategy to ensure a structured and efficient approach:

Plan: Define the objectives, data sources, and tools required for the project.
Analyze: Perform exploratory data analysis (EDA) to uncover key insights and relationships within the data.
Construct: Build and evaluate multiple models to predict employee attrition.
Execute: Deploy the best model and generate actionable insights for HR.



Steps and Deliverables:

 Data Collection and Ingestion: Raw employee data ingested via a data_ingestion.py script.

 Data Preprocessing: Cleaned dataset saved as CSV, automated by data_preprocessing.py.

 Exploratory Data Analysis (EDA): Visual charts and graphs, including scatter plots, heatmaps, and a confusion matrix, created to uncover relationships in the data. A Jupyter Notebook with these visualizations and a summary of key insights.

 Feature Engineering: Feature engineered dataset saved as CSV, automated via feature_engineering.py.

 Model Building: Multiple models built, including Logistic Regression, SVM, Decision Tree, and XGBoost.

 Model Evaluation: Performance metrics for each model, including accuracy, precision, recall, and F1 score, generated using model_evaluation.py.

 Best Model Selection: XGBoost identified as the best performing model based on the highest F1 score among Logistic Regression, SVM, Decision Tree, and XGBoost. Automated via best_model.py.


 Model Deployment: 
         Web application (app.py)  in flask framework for HR to predict employee attrition.

 Version Control: 
 
        Code versioned using GitHub, data versioned using DVC. A DVC pipeline was created to manage and reproduce the entire process.
        
 Model Registration:
 Best model registered in MLflow Model Registry, automated by model_registration.py.


 Reporting and Insights: 
        Final report summarizing analysis, model performance, and recommendations.

Conclusion, Recommendations: 

 The analysis and models reveal that employees at Salifort Motors are likely overworked, contributing to high attrition rates. XGBoost was identified as the best performing model based on the highest F1 score. Key features contributing to employee attrition include the number of projects, average monthly hours, and tenure.
 Cap Project Workload: Limit the number of projects an employee can handle to prevent burnout.
 Promotion Policy: Consider promoting employees who have been with the company for at least four years, or investigate the root causes of dissatisfaction among employees with similar tenure.

 Overtime Compensation: Reward employees for working extra hours or avoid requiring excessive overtime.
Fair Evaluation: Avoid reserving high evaluation scores for employees who work 200+ hours per month. Implement a proportionate scale for rewarding employees based on their contributions and effort.


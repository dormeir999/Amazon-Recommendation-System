@echo off

TITLE Running Amazon Recommendation App
REM Activate Anaconda
CALL C:\Users\dor.meir\AppData\Local\anaconda3\Scripts\activate.bat

REM Run Streamlit application
streamlit run amazon_recommendation_app.py

REM Pause the script to view any output before the window closes
pause
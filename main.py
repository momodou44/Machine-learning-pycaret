# 1. Library imports
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# 2. Create the app object
app = FastAPI()

# . Load trained Pipeline
model = load_model('Model-final')

# Define predict function

"""
@app.post('/predict')
def predict(carat_weight, cut, color, clarity, polish, symmetry, report):
    data = pd.DataFrame(
        [[carat_weight, cut, color, clarity, polish, symmetry, report]])
    data.columns = ['Carat Weight', 'Cut', 'Color',
                    'Clarity', 'Polish', 'Symmetry', 'Report']

    predictions = predict_model(model, data=data)
    return {'prediction': int(predictions['Label'][0])} 
"""


@app.post('/predict')
def predict(state, account_length, area_code, international_plan, voice_mail_plan, number_vmail_messages, reptotal_day_minutesort, total_day_calls, total_day_charge, total_eve_minutes,
            total_eve_calls, total_eve_charge, total_night_minutes,
            total_night_calls, total_night_charge, total_intl_minutes,
            total_intl_calls, total_intl_charge, customer_service_calls):
    data = pd.DataFrame(
        [[state, account_length, area_code, international_plan, voice_mail_plan, number_vmail_messages, reptotal_day_minutesort, total_day_calls, total_day_charge, total_eve_minutes,
          total_eve_calls, total_eve_charge, total_night_minutes,
          total_night_calls, total_night_charge, total_intl_minutes,
          total_intl_calls, total_intl_charge, customer_service_calls,]])
    data.columns = ['state', 'account length', 'area code', 'international plan',
                    'voice mail plan', 'number vmail messages', 'total day minutes',
                    'total day calls', 'total day charge', 'total eve minutes',
                    'total eve calls', 'total eve charge', 'total night minutes',
                    'total night calls', 'total night charge', 'total intl minutes',
                    'total intl calls', 'total intl charge', 'customer service calls',
                    ]

    predictions = predict_model(model, data=data)
    return {'prediction': int(predictions['Label'][0])}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

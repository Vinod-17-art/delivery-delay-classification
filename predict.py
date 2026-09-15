import joblib
import pandas as pd

model = joblib.load("delivery_delay_classifier.pkl")

sample = pd.DataFrame([{
    "distance_km": 28.5,
    "weather": "Rain",
    "traffic_level": "High",
    "vehicle_type": "Bike",
    "order_type": "Food",
    "preparation_time_min": 42,
    "driver_experience_years": 2,
    "peak_hour": 1
}])

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("Predicted Delivery Status:", prediction)
print("\nClass probabilities:")
for label, probability in zip(model.classes_, probabilities):
    print(f"{label}: {probability:.2%}")

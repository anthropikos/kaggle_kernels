# Outline

Goal: End-to-end feature extraction to classify physiological vs pathological state

Motivation: 
- Capture the finer details such as waveform shape, sharpness for the determination of physio vs patho states because current methods calculates statistics that ignores such information.
- Replicate the model and see if I can tweak it with my private dataset that has much lower sampling rate with similar performance.

Look out: 
- Challenges on how to label the physio vs patho state on my private dataset.
- Don't worry about other datasets and just focus on processing this set.

Plan: 
- Read and understand the structure of the data
- Replicate model without lightning module
- Implement train, test, and validation methods
- Train the model and plot performance
- Consider implementing RayTune for hyperparameter tuning


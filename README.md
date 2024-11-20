# ML-Transformer_Project
Our task was to predict future vessel movement using a subset of ais data. It consisted of about 700 vessels over 5 months, vith varying degrees of representation. This repository is one model I made to solve the task, based on the transformer architecture

## Preprocessing
The data was filled with nan values and values that had no reasonable representation. All these values was converted to nan, before forward and backward filling them based on the individual vessel representations

Other things I did was to encode the circular values such as lat and longitude, with the ROT, COG, and HEADING, and converting timestamps to seconds.
+ Much more
  
## Feature Engineering
I created new feaures named time_diff and seconds_to_eta, based on the time difference beteen the last known location and the prediction, and the total seconds to eta from the last known entry

## Training Data
Both the training and test data had to be generated. Mainly grouping each vessel and sort based on time, then for each timestep create a window of training data with a future timestep. I found that generating more data even though it is similar, results in better performance. Therefore we iterate through multiple timesteps to generate more data. I was however constrained with 32GB of RAM for testing (Project limit) , so I was not able to generate the maximum possible amount of data. Even though this is not a classification problem, in order to reduce vessel imbalance issues as some vessels had more instances in the training data, I capped the maximum training instances per vessel. 

## Model
Multiple different architectures have been used. I was hardware limited so the smallest model possible was used, however a bigger model might give better results. Also initially I used the time differences as  the positional encoding values. This worked fine, but after implementing a trained positional encoder instead I got the same results, so I went with that instead as the time diff encoding gave extra overhead.

## Training
Training is slow without a GPU, The amount of training data makes it slow no matter how small the model is. Parameters were found through trial and error search, as grid search was not possible. 

# Results
The project competition was on Kaggle, where we got a 130 in score, with lower beeing better. Most other boosting algorithms hovered around 70-150. I also trained an XGboost model, and got a score of 85, which is why this Transformer model was not submitted. It was dissapointing as it took some time to make it work correctly, but on the bright side the predictions generated are far more realistic.

# Mentor-Mentee Matching Coefficient (3MC)

> 3MC is used to quantify how a certain mentor matches with a mentee. 

A simple feedforward neural network was deployed to find out the pattern between mentee's surveyed response and his performance (for here, it is their pitch score). We definitely should include other mentee's information like how much investments they raised, how many investors, number of patent technology, number of patent citations, and etc. Also,  

As there are more available cleaned data for us, our model will be tested and trained against the newer data. 

Definitely this model is only a **proof of concept** since we only use mentee's surveyed data to predict their pitch performance. 

Proposed idea to improve: 

    1) Include the specific mentor and mentee pairs 
    2) Use mentor's feedback to help predict mentee's performance (i.e. the sentiment analysis with NLP)

In this repo, we will have two jupyter notebook 
  1) Keras_3MCNet-Final-prod.ipynb
    It compares how each model (Baseline Neural network, GridSearchCV Neueral Network, Random Forest, XGBoost) perform.
    XGBoost performs the best with least MSE yet the scalablility and intepretability is not optimistic. As we want to import more variables and include more variables (as well as Mentor's comment in the Mentor's report and vice versa), Neural Network as a basic framework will be a good approach as we include more data into our model. 
    
    
  2) Keras-kopt-3MCNet2-final.ipynb
    We figured doing GridSearch is not efficient nor effective to find the optimal parameter. For this reason, we used Hyperopt and Kopt in a GPU instance to optimize our parameters to fine tuned our neural network with 500 trials.
    A dropout layer is added.
  
  
  3) Keras-kopt-3MCNet4-final.ipynb
  From last notebook, we learned that having 2 variables are not enough. Because the mentor/mentee's past history and their interaction play significant roles. However, we don't want to create a huge lookup table so we add in some dropout layers and some regulizers to avoid overfitting.
  Our 3MCNet4(3MC Kopt tuned neural network with 4 variables) doesn't have a better MSE than XGBoost. 3MCNet perform close to perfection on non-outliers. 3MCNet4 wasn't optimized to reduce the MSE, instead it was attempting to reduce the huber_loss. It turns out that our 3MCNet4 performs better than XGBoost in term of getting correct prediction while being robust against outliers.
    
    $$L(y-\hat{y}) = \begin{cases} \frac{1}{2}(y-\hat{y})^2 & if &| y-\hat{y} | \leq \delta \\
                               \delta |y-\hat{y}| -\frac{1}{2} \delta^2 & o.w. \\
                               \end{cases}$$
                               
                          

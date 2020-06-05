import pandas as pd
import numpy as np

def mult(X_test_gyro):
   
   for k in range(5260):
      n = np.random.randint(0,100)
      w = X_test_gyro.iloc[[n]]
    
      X_test_gyro = pd.concat([X_test_gyro,w], ignore_index = True)
   return(X_test_gyro)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
from sklearn import metrics as ms
import numpy as np

def main(argv):
    data = pd.read_csv('final_dataset.csv')

    #Separate into feature set and target variable Removing columns not in use
    #FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
    Training_data = data.iloc[0:3420]
    Testing_data = data[3420:]   
    data_teams = Testing_data[['HomeTeam', 'AwayTeam']]
    
    # Separate into feature set and target variable Removing columns not in use
    #FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
    X_train = Training_data.drop(columns=['FTR', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTFormPtsStr', 'ATFormPtsStr', 'HTGS', 'ATGS','HTGC','ATGC', 'HTGD', 'ATGD'])
    y_train = Training_data['FTR']  

    X_test = Training_data.drop(columns=['FTR', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTFormPtsStr', 'ATFormPtsStr', 'HTGS', 'ATGS','HTGC','ATGC', 'HTGD', 'ATGD'])
    y_test = Training_data['FTR']  


    #Predicition using Deep Neural Network
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=X_train,
      y=y_train,
      batch_size=500,
      num_epochs=None,
      shuffle=True
    )

    test_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      num_epochs=1,
      shuffle=False
    )


  # Feature columns describe how to use the input.
    feature_columns = [
      tf.feature_column.numeric_column(key='HS'),
      tf.feature_column.numeric_column(key='AS'),
      tf.feature_column.numeric_column(key='HST'),
      tf.feature_column.numeric_column(key='AST'),
      tf.feature_column.numeric_column(key='HF'),
      tf.feature_column.numeric_column(key='AF'),
      tf.feature_column.numeric_column(key='HC'),
      tf.feature_column.numeric_column(key='AC'),
      tf.feature_column.numeric_column(key='HY'),
      tf.feature_column.numeric_column(key='AY'),
      tf.feature_column.numeric_column(key='HR'),
      tf.feature_column.numeric_column(key='AR'),
      tf.feature_column.numeric_column(key='HTP'),
      tf.feature_column.numeric_column(key='ATP'),
      tf.feature_column.numeric_column(key='HomeTeamLP'),
      tf.feature_column.numeric_column(key='AwayTeamLP'),
      tf.feature_column.numeric_column(key='HTFormPts'),
      tf.feature_column.numeric_column(key='ATFormPts'),
      tf.feature_column.numeric_column(key='HTWinStreak3'),
      tf.feature_column.numeric_column(key='ATWinStreak3'),
      tf.feature_column.numeric_column(key='HTWinStreak5'),
      tf.feature_column.numeric_column(key='ATWinStreak5'),
      tf.feature_column.numeric_column(key='HTLossStreak3'),
      tf.feature_column.numeric_column(key='ATLossStreak3'),
      tf.feature_column.numeric_column(key='HTLossStreak5'),
      tf.feature_column.numeric_column(key='ATLossStreak5'),
      tf.feature_column.numeric_column(key='DiffPts'),
      tf.feature_column.numeric_column(key='DiffFormPts'),
      tf.feature_column.numeric_column(key='DiffLP'),
    ]



    # Build a Deep Neuron Network
    # with 1 hidden layer & 10  and
    # classifying 3 output classes.

    model = tf.estimator.DNNClassifier(
      model_dir='model/',
      hidden_units=[10],
      feature_columns=feature_columns,
      n_classes=3,
      label_vocabulary=['H', 'D', 'A'],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

    #target variables 
    expected = ['H', 'D', 'A']

    
    for i in range(0, 200):
      # Train the Model.
      model.train(input_fn=train_input_fn, steps=100)

      # Evaluate the model.
      evaluation_result = model.evaluate(input_fn=test_input_fn)
      print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**evaluation_result))    
      # Test set accuracy: 0.603

      #make model prediction.
      predictions = list(model.predict(input_fn=test_input_fn))


  #loop through individual predictions and compare it to the ground truth
    for HT, AT, pred_dict, expec in zip(data_teams['HomeTeam'], data_teams['AwayTeam'], predictions, y_test):
       template = ('\n {} vs. {}: Prediction is "{}" ({:.1f}%), expected "{}"')
      
       class_id = pred_dict['class_ids'][0]
       probability = pred_dict['probabilities'][class_id]
      
       print(template.format(HT,AT,expected[class_id],
                          100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/yunah7/OSS_Project2.git

import sys
def load_dataset(dataset_path):
	#To-Do: Implement this function
        import pandas as pd
        data_df = pd.read_csv(dataset_path)
        return data_df

def dataset_stat(dataset_df):
    feats = len(dataset_df.columns)-1
    class0 = len(dataset_df.loc[dataset_df["target"]==0])
    class1 = len(dataset_df.loc[dataset_df["target"]==1])
    return feats, class0, class1
	#To-Do: Implement this function

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
    from sklearn.model_selection import train_test_split
    X = dataset_df.drop(columns="target", axis=1)
    y = dataset_df["target"]
    return train_test_split(X, y, test_size=testset_size, random_state=46)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import precision_score
  from sklearn.metrics import recall_score
  dt_cls = DecisionTreeClassifier()
  dt_cls.fit(x_train, y_train)
  return accuracy_score(y_test, dt_cls.predict(x_test)), precision_score(y_test, dt_cls.predict(x_test)), recall_score(y_test, dt_cls.predict(x_test))

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import precision_score
  from sklearn.metrics import recall_score
  dt_cls = RandomForestClassifier()
  dt_cls.fit(x_train, y_train)
  return accuracy_score(y_test, dt_cls.predict(x_test)), precision_score(y_test, dt_cls.predict(x_test)), recall_score(y_test, dt_cls.predict(x_test))


def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
  from sklearn.svm import SVC
  from sklearn.pipeline import Pipeline, make_pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import precision_score
  from sklearn.metrics import recall_score
  svm_pipe = make_pipeline(
      StandardScaler(),
      SVC()
  )
  svm_pipe.fit(x_train, y_train)
  return accuracy_score(y_test, svm_pipe.predict(x_test)), precision_score(y_test, svm_pipe.predict(x_test)), recall_score(y_test, svm_pipe.predict(x_test))


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)

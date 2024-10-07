import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy import stats 


#######This part deals with data reading and data segmentation.#########

# Read file 

df_train = pd.read_csv('train1.csv') # this is training data 
df_test  = pd.read_csv('test1.csv')  # this is testing  data

# Fill missing values

df_train = df_train.fillna(df_train.mean())
df_test = df_test.fillna(df_test.mean())

# Spilt into features and labels (x is feature and y is label)

x_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values 
x_test  = df_test.iloc[:,1:].values
y_test  = df_test.iloc[:,0].values

####### Train data #######################################################

##define fnction 

# define the posterier function 
def posterior_function(x) :  
    posterior = np.array([]) #create the null space of posterior function
    for i in range(num_classes):
     log_prior = np.log(priors[i])
     gauss = stats.norm.pdf(x,loc=means[i],scale=np.sqrt(variances))
     log_likehood = np.sum(np.log(gauss))
     A = log_likehood + log_prior
     posterior=np.append(posterior,A)
   
    return classes[np.argmax(posterior)]



## Caculate prior probabilities through training data 

# classes classification type (0,1,2) ; counts records the number of this type
classes, counts = np.unique(y_train, return_counts = True )  

# prior probability = counts of type / total trainimg data 
priors = counts / len(y_train)

print(f'Classes: {classes}')
print(f'Prior probabilities: {priors}')

# Calculate mean and variance for each feature for each class

num_classes = len(classes)
means = np.zeros((num_classes, x_train.shape[1]))
variances = np.zeros((num_classes, x_train.shape[1]))

for i, c in enumerate(classes):
    x_c = x_train[y_train == c]
    means[i, :] = x_c.mean(axis=0)
    variances[i, :] = x_c.var(axis=0)


######## Evaluate accuracy through test data #########################

# Predict which label the test DATA belongs to 

predict = np.array([posterior_function(i) for i in x_test])

predict = np.array(predict)
print(f'y_test:{y_test}')
print(f'predict:{predict}')

# Check  each class in the predictions

print(f'Predicted classes: {np.unique(predict)}')

# Calculate Accuray 

accuracy = np.mean (predict == y_test)
print(f'Accuracy: {accuracy * 100:f}%')

if accuracy <= 0.90:
    raise ValueError(f"Accuracy did not exceed 90%. Actual accuracy: {accuracy * 100:f}%")

######## plot PCA diagram #########################################

# plot the PCA visualized result of testing data

print(x_test)
pca = PCA(n_components=13) # each feature 
newX = pca.fit_transform(x_test)
invX = pca.inverse_transform(newX)
#print(newX)
#print(invX)
#print(pca.explained_variance_)
#print(pca.explained_variance_ratio_)
#print(pca.explained_variance_ratio_.cumsum)
exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
plt.bar(range(0,len(exp_var_pca)),exp_var_pca, alpha = 0.5,
        align = 'center',label = 'Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)),cum_sum_eigenvalues, 
        where = 'mid',label='Cumulative explained variance')
plt.xlabel('Principal component index')
plt.ylabel('Explained Variance ratio')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


######### plot as confusion_matrix in heatmap ############################

# define plot function 

def plot(y_true,y_pred):
   #labels = classes
   column = [f'Predicted_Wine{label}' for label in classes]
   indices = [f'Actual_wine {label}' for label in classes]
   table = pd.DataFrame(confusion_matrix(y_true,y_pred),columns = column , index = indices)

   return sns.heatmap(table,cmap='Blues',annot=True,fmt='d',cbar=True)

# confusion_matrix

matrix_array=confusion_matrix(y_test,predict)
print(matrix_array)
plot(y_test,predict)
plt.show()




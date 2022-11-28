from dataclasses import dataclass
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
plt.style.use('ggplot')

# Class with functions to help data modification
class DataTool():

    def __init__(self, input_file: str) -> None:

        pd_file = pandas.read_csv(input_file)
        pd_file.pop('id')

        self.data: Dict(str, np.ndarray) = {'labels': None, 'features': None}
        self.data['labels'] = np.array(pd_file.pop('CLASS_LABEL'))
        self.data['features'] = np.array(pd_file)

    def get_labels(self):
        return self.data['labels']

    def get_features(self):
        return self.data['features']

data: DataTool = DataTool('data/Phishing_Legitimate_full.csv')

x = data.get_features()
y = data.get_labels()

#x=x[:,[1,2,3,4,5,6,7,9,10,11,12,13,15,16,17,18,19,20,22,23,24,26,28,29,30,31,32,33,34,35,36,41,44,45,46,47]]
#x=x[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]]

x = StandardScaler().fit_transform(x)




#pca = PCA(n_components=48)
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#principalDf = pd.DataFrame(data = principalComponents)


#np.savetxt('pcaResults.csv',pca.explained_variance_ratio_)
#print(pca.explained_variance_ratio_)




labelDf = pd.DataFrame(y, columns=['target'])
finalDf = pd.concat([principalDf, labelDf[['target']]], axis = 1)



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1,0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
    #ax.scatter(finalDf.loc[indicesToKeep]
               , finalDf.loc[indicesToKeep, 'principal component 2']
               #, finalDf.loc[indicesToKeep]
               , c = color
               , s = 1 )
ax.legend(targets)
ax.grid()

plt.show()






from matplotlib import pyplot as plt
import pandas as pd
import itertools

def scatterplot(df,num_features,path ,DV):
    print(path)
    for f_pair in list(itertools.combinations(num_features,2)):
        df.plot.scatter(x= f_pair[0],y =f_pair[1], c = DV, colormap='viridis')
        plt.xlabel(f_pair[0])
        plt.ylabel(f_pair[1])
        plt.savefig(path+f_pair[0]+' vs '+f_pair[1]+'.png')
        plt.close()

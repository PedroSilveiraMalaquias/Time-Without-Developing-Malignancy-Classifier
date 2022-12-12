import time
from datetime import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from NeuralNetwork import  NeuralNetwork
from lifelines import KaplanMeierFitter

from category_encoders import LeaveOneOutEncoder
from MalignancyRelatedColumns import malignancyRelatedColumns

def getPatientYearsWithoutMalignancy(patientId, timeWithoutMalignancy, malignancyType):
    # print(patientId, timeWithoutMalignancy.at[patientId, malignancyType] )
    patientDaysWithoutMalignancy, developed = eval(timeWithoutMalignancy.at[patientId, malignancyType])
    if patientId in timeWithoutMalignancy.index and patientDaysWithoutMalignancy != -1:
        return (int(patientDaysWithoutMalignancy / 365) + 1, int(developed))
    else:
        return (-1, int(developed))


def getOutputClass(patientInputIds, timeWithoutMalignancy, malignancyType):
    outputClass = patientInputIds.apply(
        lambda id: getPatientYearsWithoutMalignancy(id, timeWithoutMalignancy, malignancyType))
    return outputClass


def getOutput(patientInputIds, timeWithoutMalignancy, malignancyType):
    [output, developed] = [*zip(*getOutputClass(patientInputIds, timeWithoutMalignancy, malignancyType))]
    return (output, developed)

def kaplanMeierSurvivalAnalysis(liverDataFrame, timeWithoutMalignancy):
    for malignancyType in list(malignancyRelatedColumns.keys())[:10]:
        output, developed = getOutput(liverDataFrame['TRR_ID_CODE'], timeWithoutMalignancy, malignancyType)
        kmf = KaplanMeierFitter()
        kmf.fit(pd.Series(np.array(output)), pd.Series(np.array(developed)), label=malignancyType)
        # print(kmf.survival_function_)
        years = kmf.survival_function_.index
        (low, high) = np.transpose(kmf.confidence_interval_survival_function_.values)
        plt.fill_between(years, low, high, color='gray', alpha=0.3)
        kmf.survival_function_.plot(ax=plt.gca())

    plt.legend(loc='lower left', fancybox=True, framealpha=0)
    plt.ylabel('Survival function ')
    plt.show()


def stringDateToTimestamp(date):
    if pd.isnull(date):
        return np.nan
    else:
        return time.mktime(datetime.strptime(date, '%Y-%m-%d').timetuple())

"""
    1- Drop columns with one category
    2- Turn date columns in timestamps (Replacing nan values by the column mean)
    3- Drop id code
    4- Perform one hot encoding in 2 categories columns
    5- Perform leave one out encoding in the rest
    6- Remember that a 3 categories with many nan is actually a 2 categories column
"""

def categoricalsToNumeric(liverDataFrame, targetClasses):
    columnsWithOneCategory = liverDataFrame.columns[liverDataFrame.nunique() <= 1].tolist()
    liverDataFrame.drop(columns=columnsWithOneCategory + ['TRR_ID_CODE'], inplace=True)

    dateColumns = list(filter(lambda columnName: columnName.find('DATE') >= 0, liverDataFrame.columns.values))
    liverDataFrame[dateColumns] = liverDataFrame[dateColumns].apply(lambda dateColumn : dateColumn.apply(stringDateToTimestamp))
    liverDataFrame[dateColumns] = liverDataFrame[dateColumns].apply(lambda dateColumn: dateColumn.replace(np.nan, dateColumn.mean()))

    categoricals = list(liverDataFrame.dtypes[liverDataFrame.dtypes == object].index)

    liverDataFrame[categoricals] = liverDataFrame[categoricals].fillna('MISSING')
    columnsWithThreeCategories = liverDataFrame[categoricals].columns[liverDataFrame[categoricals].nunique() <= 3].tolist()

    liverDataFrame = pd.get_dummies(liverDataFrame, columns=columnsWithThreeCategories, drop_first=True)

    categoricals = list(liverDataFrame.dtypes[liverDataFrame.dtypes == object].index)

    enc = LeaveOneOutEncoder(cols=categoricals)
    liverDataFrame = enc.fit_transform(liverDataFrame, targetClasses)

    return liverDataFrame

    #print(columnsWithThreeCategories)

if __name__ == '__main__':

    start_time = time.time()

    # Create dataset
    liverDataFrame = pd.read_csv('./liver_data.csv')
    timeWithoutMalignancy = pd.read_csv('./timeWithoutMalignancy.csv', index_col=0)

    filt = liverDataFrame['TRR_ID_CODE'].isin(timeWithoutMalignancy.index.tolist())
    liverDataFrame.drop(index=liverDataFrame[~filt].index, inplace=True)

    numericals = list(liverDataFrame.dtypes[liverDataFrame.dtypes != object].index)

    categoricals = list(liverDataFrame.dtypes[liverDataFrame.dtypes == object].index)
    print("Number of categorical columns before: ", len(categoricals))
    #print(categoricals)

    #kaplanMeierSurvivalAnalysis(liverDataFrame, timeWithoutMalignancy)

    """
    -1 will be not developed.
    0 means developed in the first year.
    1 means developed in the second year.
    .
    .
    .
    23 means developed in the 24th year
    """
    output, developed = getOutput(liverDataFrame['TRR_ID_CODE'], timeWithoutMalignancy, 'LUNG')
    targetClasses = []
    for output, developed in zip(output, developed):
        if developed:
            targetClasses.append(output)
        else:
            targetClasses.append(0)

    outputClasses = np.unique(targetClasses)
    nOutputClasses = np.max(outputClasses) - np.min(outputClasses) + 1
    print("Output Classes:", nOutputClasses)

    liverDataFrame = categoricalsToNumeric(liverDataFrame, targetClasses)

    categoricals = list(liverDataFrame.dtypes[liverDataFrame.dtypes == object].index)
    print("Number of categorical columns after: ", len(categoricals))

    liverDataFrame.fillna(0, inplace=True)
    #liverDataFrame.to_csv('input.csv', index=False)


    data = list(zip(liverDataFrame.astype(float).to_numpy(), targetClasses))
    random.shuffle(data)

   #print(data)

    trainingDataLen = int(len(data)*0.8)
    trainingData = data[:trainingDataLen]
    testData = data[trainingDataLen:]


    neuralNetwork = NeuralNetwork([172, 200, 200, nOutputClasses])
    neuralNetwork.train(trainingData, testData)
    print('It took: ', time.time() - start_time, " seconds to run.")

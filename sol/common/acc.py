from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas
import sys


class AccResult:
    def __init(self):
        self.r2sq = None
        self.mse = None
        self.rmse = None
        self.mae = None

    def save_to_file(self, f):
        f.write("Rsquared: %0.6f\n"%self.r2sq)
        f.write("MSE: %0.6f\n"%self.mse)
        f.write("RMSE: %0.6f\n"%self.rmse)
        f.write("MAE: %0.6f"%self.mae)


def calc_accuracy(resultset):
    r = AccResult()
    
    r.r2sq = r2_score(resultset['Actual'], resultset['Predicted'])
    r.mse = mean_squared_error(resultset['Actual'], resultset['Predicted'])
    r.rmse = np.sqrt(r.mse)
    r.mae = mean_absolute_error(resultset['Actual'], resultset['Predicted'])

    return r

if __name__ == "__main__":

    filename = sys.argv[1]
    resultset = pandas.read_csv(filename)
    r = calc_accuracy(resultset)

    print("Rsquared: %0.6f"%r.r2sq)
    print("MSE: %0.6f"%r.mse)
    print("RMSE: %0.6f"%r.rmse)
    print("MAE: %0.6f"%r.mae)


import os
import smtplib
import numpy as np
import pandas as pd
from sys import stdout
from time import time
from prettytable import PrettyTable
from scipy.linalg.special_matrices import toeplitz
from mlpp.inference import LearnerLogReg
from mlpp.preprocessing import FeaturesBinarizer
from sklearn.externals.joblib import Parallel, delayed
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE
import warnings
warnings.filterwarnings('ignore')

def features_normal_cov_toeplitz(n_samples, n_features, rho=0.5):
    cov = toeplitz(rho ** np.arange(0, n_features))
    return np.random.multivariate_normal(
        np.zeros(n_features), cov, size=n_samples)

def sigmoid(t):
    return (1./(1. + np.e ** (-t)))

def compute_time():
    n_samples = 10000
    X = features_normal_cov_toeplitz(n_samples, n_features)

    intercept = np.random.randint(0, 5)
    beta = np.random.randint(0, 10, n_features)
    Z = intercept + X.dot(beta)
    pr = sigmoid(Z)
    u = np.random.rand(n_samples)
    y = (u <= pr).astype(int)
    y = 2 * (y != y[0]) - 1

    # binarize data
    n_cuts = 10
    binarizer = FeaturesBinarizer(n_cuts=n_cuts)
    X_bin = binarizer.fit_transform(X)
    blocks_start = binarizer.feature_indices[:-1, ]
    blocks_length = binarizer.n_values

    # binarsity
    C = 1e2
    learner = LearnerLogReg(penalty='binarsity', solver='svrg', C=C,
                            step=1e-3, blocks_start=blocks_start,
                            blocks_length=blocks_length)
    start = time()
    learner.fit(X_bin, y)
    times_bina = time() - start
    
    # lasso
    C = 1.5e2
    learner = LearnerLogReg(penalty='l1', solver='svrg', C=C,
                            step=1e-3)
    start = time()
    learner.fit(X_bin, y)
    times_lasso = time() - start
        
    return times_bina, times_lasso


start_init = time()
n_features_grid = [10, 50, 100, 200, 500] 
n_simu = 100

columns = ['time', 'Algorithms', 'n_features']
results = pd.DataFrame(columns=columns)

for n_features in n_features_grid:  
    print("n_features = %d" % n_features)    
    parallel = Parallel(n_jobs=100, verbose=10)
    times = parallel(delayed(compute_time)() for _ in range(n_simu))
    
    times = np.array(times, dtype=np.dtype('float,float'))
    times_bina = pd.Series(times['f0'], name='time')
    times_lasso = pd.Series(times['f1'], name='time')

    algo_bina = np.chararray(n_simu, itemsize=9, unicode=True)
    algo_bina[:] = 'Binarsity'
    algo_bina = pd.Series(algo_bina, name='Algorithms')
    algo_lasso = np.chararray(n_simu, itemsize=9, unicode=True)
    algo_lasso[:] = 'Lasso'
    algo_lasso = pd.Series(algo_lasso, name='Algorithms')

    n_features_array = pd.Series(
                            np.full(n_simu, n_features),
                            name='n_features')

    tmp = pd.concat([times_bina, 
                    algo_bina, 
                    n_features_array], 
                    axis=1)
    
    results = results.append(tmp, ignore_index=False)

    tmp = pd.concat([times_lasso, 
                    algo_lasso, 
                    n_features_array], 
                    axis=1)
    
    results = results.append(tmp, ignore_index=False)

results[['n_features']] = results[['n_features']].astype(int)
results.to_csv("./computing_times_simu.csv", index=False)
    
print("\n global time: %.1f sec" % (time() - start_init))
os.system('say "computation finished"')

# send results by email
send_from = 'simon.bussy@upmc.fr'
send_to = ['simon.bussy@gmail.com']
subject = "computation finished"
text = "results available \n"
files = "./computing_times_simu.csv"

msg = MIMEMultipart()
msg['From'] = send_from
msg['To'] = COMMASPACE.join(send_to)
msg['Subject'] = subject

msg.attach(MIMEText(text))

with open(files, "rb") as fil:
    part = MIMEApplication(
        fil.read(),
        Name="computing_times_simu.csv"
    )
    part[
        'Content-Disposition'] = 'attachment; filename=' \
                                 '"computing_times_simu.csv"'
    msg.attach(part)

try:
    smtp = smtplib.SMTP('smtp.upmc.fr')
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()
    print("Successfully sent email")
except smtplib.SMTPException:
    print("Error: unable to send email")

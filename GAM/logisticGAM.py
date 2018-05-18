import os
import smtplib
import numpy as np
import pandas as pd
from sys import argv
from prettytable import PrettyTable
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from tick.preprocessing import FeaturesBinarizer
from sklearn.metrics import roc_auc_score
from pygam import LogisticGAM
import warnings

warnings.filterwarnings('ignore')

# get command-line arguments
if len(argv) > 2:

    filename = argv[1]
    is_header = argv[2]

    if is_header == 'true':
        header = "infer"
    else:
        header = None

    # Get data
    os.chdir('./datasets/%s' % filename)
    df = pd.read_csv('./%s' % filename, header=header)

    n, p = df.shape
    if not isinstance(df, pd.DataFrame) or not n > 0 or not p > 0:
        raise ValueError("problem when loading data")

    try:
        test = argv[3] == 'test'
    except:
        test = False
        pass

else:
    raise ValueError("at least 3 command-line arguments expected, %s given",
                     len(argv) - 1)

# drop lines with NaN values
df.dropna(axis=0, how='any', inplace=True)

# if dataset churn: drop phone feature
if filename == 'churn':
    df = df.drop(df.columns[[3]], axis=1)

# get label (have to be the last column!)
idx_label_column = -1
labels = df.iloc[:, idx_label_column]
labels = (labels.values != labels.values[0]).astype(int)

# drop it from df
df = df.drop(df.columns[[idx_label_column]], axis=1)

# shuffle and split training and test sets
X, X_test, y, y_test = train_test_split(
    df, labels, test_size=.33, random_state=0, stratify=labels)

del df

if test:
    n_restrict = 200
    X = X.iloc[:n_restrict, :]
    y = y[:n_restrict]
    X_test = X_test.iloc[:n_restrict, :]
    y_test = y_test[:n_restrict]

# get categorical features index
cate_feat_idx = []
for i in range(X.shape[1]):
    feature_type = FeaturesBinarizer._detect_feature_type(X.ix[:, i])
    if feature_type == 'discrete':
        cate_feat_idx.append(i)

original_feature_names = X.columns

feature_names_cont = list()
for i, name in enumerate(original_feature_names):
    if i not in cate_feat_idx:
        feature_names_cont.append(name)

# separate continuous and categorical features
X_cat = X[X.columns[cate_feat_idx]]
X_test_cat = X_test[X_test.columns[cate_feat_idx]]
X_cat.reset_index(drop=True, inplace=True)
X_test_cat.reset_index(drop=True, inplace=True)

X_cont = X.drop(X.columns[cate_feat_idx], axis=1)
X_test_cont = X_test.drop(X_test.columns[cate_feat_idx], axis=1)
X_cont.reset_index(drop=True, inplace=True)
X_test_cont.reset_index(drop=True, inplace=True)

print("Training:")
print(X.shape)
print("Test:")
print(X_test.shape)

# Center and reduce continuous data
standardscaler = StandardScaler()
X_std = pd.DataFrame(standardscaler.fit_transform(X_cont))
X_test_std = pd.DataFrame(standardscaler.transform(X_test_cont))
print("data centered and reduced")

# use only 10k examples max for Cross-Val
n_restrict_cv = 10000

os.system('rm -rR ./results')
os.makedirs('./results/y_pred')
os.makedirs('./results/learning_curves')
np.save('./results/y_test', y_test)

start_init = time()
model = "GAM"
print("\n launch %s" % model)

n_grid = 10
lam_grid = np.logspace(0, 3, n_grid)
n_splines_grid = np.linspace(5, 50, n_grid).astype(int)

gam = LogisticGAM(dtype='numerical')
grid_result = gam.gridsearch(X_std.iloc[:n_restrict_cv, :],
               y[:n_restrict_cv],
               lam=lam_grid,
               n_splines=n_splines_grid,
               return_scores=True,
               keep_best=True)

scores = pd.DataFrame(columns=n_splines_grid)
line = list()
for key, value in grid_result.items():
    params = key.get_params()
    lam = params.get('lam')
    n_splines = params.get('n_splines')
    line.append(value)

    if len(line) == len(n_splines_grid):
        scores = scores.append(pd.Series(line, index=n_splines_grid),
                               ignore_index=True)
        line = list()

scores.index = ["%.1f" % val for val in lam_grid]
scores.to_csv("./results/learning_curves/9-%s" % model)

best_params = gam.get_params()
print("Hyper-parameters chosen: \nlam = %.2f, \nn_splines=%s"
      % (best_params.get('lam'), best_params.get('n_splines')))
np.save('./results/learning_curves/best_params_%s.npy' % model, best_params)

start = time()
gam.fit(X_std, y)
y_pred = gam.predict_proba(X_test_std)
np.save('./results/y_pred/9-%s' % model, y_pred)
auc = roc_auc_score(y_test, y_pred)
auc = max(auc, 1 - auc)

t = PrettyTable(['Algos', 'AUC', 'time'])
t.add_row([model, "%g" % auc, "%.3f" % (time() - start)])

# Final performances comparison
print("\n global time: %s s" % (time() - start_init))
print(t)
results = open("./results/results.txt", "w")
results.write('%s' % t)
results.write("\n global time: %s s" % (time() - start_init))
results.close()

# # compress results and send it by email
# os.system('say "computation finished"')
# os.system('zip -r results.zip results')
#
# send_from = 'simon.bussy@upmc.fr'
# send_to = ['simon.bussy@gmail.com']
# subject = "computation finished for %s" % filename
# text = "results available \n"
# files = "./results.zip"
#
# msg = MIMEMultipart()
# msg['From'] = send_from
# msg['To'] = COMMASPACE.join(send_to)
# msg['Subject'] = subject
#
# msg.attach(MIMEText(text))
#
# with open(files, "rb") as fil:
#     part = MIMEApplication(
#         fil.read(),
#         Name="result_%s.zip" % filename
#     )
#     part[
#         'Content-Disposition'] = 'attachment; filename=' \
#                                  '"result_%s.zip"' % filename
#     msg.attach(part)
#
# try:
#     smtp = smtplib.SMTP('smtp.upmc.fr')
#     smtp.sendmail(send_from, send_to, msg.as_string())
#     smtp.close()
#     print("Successfully sent email")
# except smtplib.SMTPException:
#     print("Error: unable to send email")
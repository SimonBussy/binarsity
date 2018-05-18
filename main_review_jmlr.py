import os
import smtplib
import numpy as np
import pandas as pd
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE
from sys import stdout, argv
from prettytable import PrettyTable
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from tick.inference import LogisticRegression
from tick.preprocessing import FeaturesBinarizer
from sklearn.utils.validation import indexable
from sklearn.model_selection import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.externals.joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')

# get command-line arguments
if len(argv) > 4:

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

    K = int(argv[3])
    if not K > 0:
        raise ValueError("problem with K, %s given", K)

    selection = argv[4]
    if selection not in ['1st', 'min']:
        raise ValueError("problem with selection, %s given", selection)

    # default
    test = False

    try:
        test = argv[5] == 'test'
    except:
        pass

    # default
    n_cuts_min = 10
    n_cuts_max = 80
    n_cuts_grid_size = 20

    try:
        n_cuts_min = int(argv[6])
        n_cuts_max = int(argv[7])
        n_cuts_grid_size = int(argv[8])
    except:
        pass

else:
    raise ValueError("at least 4 command-line arguments expected, %s given",
                     len(argv) - 1)


def cross_val_score_(estimators, X, y=None, groups=None, scoring=None,
                     cv=None, n_jobs=1, verbose=0, fit_params=None):
    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=True)
    cv_iter = list(cv.split(X, y, groups))

    parallel = Parallel(n_jobs=n_jobs, verbose=0)

    scores = parallel(delayed(_fit_and_score)(estimators[i], X, y,
                                              check_scoring(estimators[i],
                                                            scoring=scoring),
                                              train, test, verbose, None,
                                              fit_params)
                      for i, (train, test) in enumerate(cv_iter))

    return np.array(scores)[:, 0]


def compute_score(clf, X, y, K, verbose=True, fit_params=None):
    scores = cross_val_score_(clf, X, y, cv=K, verbose=0,
                              n_jobs=1, scoring="roc_auc",
                              fit_params=fit_params)
    score_mean = scores.mean()
    score_std = 2 * scores.std()
    if verbose:
        print("\n AUC: %0.3f (+/- %0.3f)" % (score_mean, score_std))
    return score_mean, score_std


with_categorical = False

# drop lines with NaN values
df.dropna(axis=0, how='any', inplace=True)

# if dataset churn: drop phone feature
if filename == 'churn':
    df = df.drop(df.columns[[3]], axis=1)

# get label (have to be the last column!)
idx_label_column = -1
labels = df.iloc[:, idx_label_column]
labels = 2 * (labels.values != labels.values[0]) - 1
# drop it from df
df = df.drop(df.columns[[idx_label_column]], axis=1)

# shuffle and split training and test sets
X, X_test, y, y_test = train_test_split(
    df, labels, test_size=.33, random_state=0, stratify=labels)

del df

# speed up restriction
# n_restrict = 1000000  # 200k examples max
if test:
    n_restrict = 200
    C_grid_size = 4
    n_cuts_grid_size = 3
    X = X.iloc[:n_restrict, :]
    y = y[:n_restrict]
    X_test = X_test.iloc[:n_restrict, :]
    y_test = y_test[:n_restrict]
else:
    C_grid_size = 25

# get categorical features index
cate_feat_idx = []
for i in range(X.shape[1]):
    feature_type = FeaturesBinarizer._detect_feature_type(X.ix[:, i])
    if feature_type == 'discrete':
        cate_feat_idx.append(i)

if (len(cate_feat_idx) == 0):
    with_categorical = False

original_feature_names = X.columns

if not with_categorical:
    feature_names_cont = list()
    for i, name in enumerate(original_feature_names):
        if i not in cate_feat_idx:
            feature_names_cont.append(name)
else:
    feature_names_cont = original_feature_names

n_cuts_grid = np.linspace(n_cuts_min, n_cuts_max, n_cuts_grid_size, dtype=int)

# separate continuous and categorical features
X_cat = X[X.columns[cate_feat_idx]]
X_test_cat = X_test[X_test.columns[cate_feat_idx]]
X_cat.reset_index(drop=True, inplace=True)
X_test_cat.reset_index(drop=True, inplace=True)

if with_categorical:
    binarizer = FeaturesBinarizer()
    binarizer.fit(pd.concat([X_cat, X_test_cat], axis=0))
    X_cat_bin = pd.DataFrame(binarizer.transform(X_cat).toarray())
    X_test_cat_bin = pd.DataFrame(binarizer.transform(X_test_cat).toarray())

#del X_cat, X_test_cat

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
os.makedirs('./results/beta')
os.makedirs('./results/cvg')
os.makedirs('./results/learning_curves')
np.save('./results/y_test', y_test)


def run_models(model_):
    result = list()

    C_grid = np.logspace(1, 3, C_grid_size)

    if model_ in ['bina', 'group_TV']:

        if model_ == 'bina':
            # logistic regression on binarized features, binarsity penalization
            model = "Binarsity"
            C_grid = np.logspace(1, 4, C_grid_size)
        if model_ == 'group_TV':
            # logistic regression on binarized features, group-TV penalization
            model = "Group_TV"
            C_grid = np.logspace(1, 4, C_grid_size)

        print("\n launch %s" % model)

        if with_categorical:
            X_final = pd.concat([X_cont, X_cat], axis=1)
            X_test_final = pd.concat([X_test_cont, X_test_cat], axis=1)
        else:
            X_final = X_cont
            X_test_final = X_test_cont

        # prendre une gde valeur de n_cut puis cross valider sur C
        n_cuts_chosen = 30

        binarizer = FeaturesBinarizer(n_cuts=n_cuts_chosen)
        binarizer.fit(pd.concat([X_final, X_test_final], axis=0))

        if with_categorical:
            X_final = pd.concat([X_cont.iloc[:n_restrict_cv, :],
                                 X_cat.iloc[:n_restrict_cv, :]],
                                axis=1)
            X_test_final = pd.concat([X_test_cont.iloc[:n_restrict_cv, :],
                                      X_test_cat.iloc[:n_restrict_cv, :]],
                                     axis=1)

        else:
            X_final = X_cont.iloc[:n_restrict_cv, :]
            X_test_final = X_test_cont.iloc[:n_restrict_cv, :]

        X_bin = binarizer.transform(X_final)
        X_test_bin = binarizer.transform(X_test_final)

        # cross validation on C
        avg_scores, score_test = np.empty(0), []
        tmp = 0
        for i, C_ in enumerate(C_grid):
            tmp += 1
            print("CV %s: %d%%" % (
                model, tmp * 100 / C_grid_size))
            stdout.flush()

            learners = [
                LogisticRegression(penalty='binarsity', solver='svrg', C=C_,
                              verbose=False, step=1e-3,
                              blocks_start=binarizer.feature_indices[:-1, ],
                              blocks_length=binarizer.n_values)
                for _ in range(K)]
            auc = compute_score(learners, X_bin, y[:n_restrict_cv], K,
                                verbose=False)[0]

            avg_scores = np.append(avg_scores, max(auc, 1 - auc))

            learner = LogisticRegression(penalty='binarsity', solver='svrg',
                                    C=C_, verbose=False, step=1e-3,
                                    blocks_start=binarizer.feature_indices[
                                                 :-1, ],
                                    blocks_length=binarizer.n_values)
            learner.fit(X_bin, y[:n_restrict_cv])
            y_pred = learner.predict_proba(X_test_bin)[:, 1]
            score_test.append(roc_auc_score(y_test[:n_restrict_cv], y_pred))

        idx_best = np.unravel_index(avg_scores.argmax(),
                                    avg_scores.shape)[0]
        C_best = C_grid[idx_best]
        if selection == 'min':
            C_chosen = C_best
        if selection == '1st':
            max_ = avg_scores.max()
            min_ = avg_scores.min()
            idx = [i for i, is_up in enumerate(
                list(avg_scores >= max_ - .05 * (max_ - min_)))
                   if is_up]
            idx_chosen = min(idx) if len(idx) > 0 else idx_best
            C_chosen = C_grid[idx_chosen]

        # learning curves
        learning_curves = np.column_stack((C_grid, avg_scores, score_test))
        np.save('./results/learning_curves/5-%s-selection_%s' % (
            model, selection), learning_curves)

        if with_categorical:
            X_final = pd.concat([X_cont, X_cat], axis=1)
            X_test_final = pd.concat([X_test_cont, X_test_cat], axis=1)
        else:
            X_final = X_cont
            X_test_final = X_test_cont

        binarizer = FeaturesBinarizer(n_cuts=n_cuts_chosen)
        binarizer.fit(pd.concat([X_final, X_test_final], axis=0))
        X_bin = binarizer.transform(X_final)
        X_test_bin = binarizer.transform(X_test_final)

        blocks_start = binarizer.feature_indices[:-1, ]
        blocks_length = binarizer.n_values
        np.save('./results/beta/blocks_start-%s' % model, blocks_start)

        learner = LogisticRegression(penalty='binarsity', solver='svrg', C=C_chosen,
                                verbose=False, step=1e-3,
                                blocks_start=blocks_start,
                                blocks_length=blocks_length)
        start = time()
        learner.fit(X_bin, y)
        y_pred = learner.predict_proba(X_test_bin)[:, 1]
        np.save('./results/y_pred/5-%s' % model, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        auc = max(auc, 1 - auc)
        result.append([model.replace('_', ' '), "%g" % auc,
                  "%.3f" % (time() - start)])
        print("\n %s done, AUC: %.3f" % (model, auc))

        # cvg check
        cvg_history = np.column_stack(
            (learner._solver_obj.get_history("n_iter"),
             learner._solver_obj.get_history("obj")))
        np.save('./results/cvg/5-%s' % model, cvg_history)

        coeffs = learner.weights
        np.save('./results/beta/5-%s' % model, coeffs)

    if model_ in ['group_L1']:

        C_grid = np.logspace(1, 4, C_grid_size)

        # logistic regression on binarized features, group-L1 penalization
        model = "Group_L1"
        print("\n launch %s" % model)

        if with_categorical:
            X_final = pd.concat([X_cont, X_cat], axis=1)
            X_test_final = pd.concat([X_test_cont, X_test_cat], axis=1)
        else:
            X_final = X_cont
            X_test_final = X_test_cont

        # prendre une gde valeur de n_cut puis cross valider sur C
        n_cuts_chosen = 30

        binarizer = FeaturesBinarizer(n_cuts=n_cuts_chosen)
        binarizer.fit(pd.concat([X_final, X_test_final], axis=0))

        if with_categorical:
            X_final = pd.concat([X_cont.iloc[:n_restrict_cv, :],
                                 X_cat.iloc[:n_restrict_cv, :]],
                                axis=1)
            X_test_final = pd.concat([X_test_cont.iloc[:n_restrict_cv, :],
                                      X_test_cat.iloc[:n_restrict_cv, :]],
                                     axis=1)

        else:
            X_final = X_cont.iloc[:n_restrict_cv, :]
            X_test_final = X_test_cont.iloc[:n_restrict_cv, :]

        X_bin = binarizer.transform(X_final)
        X_test_bin = binarizer.transform(X_test_final)

        # cross validation on C
        avg_scores, score_test = np.empty(0), []
        tmp = 0
        for i, C_ in enumerate(C_grid):
            tmp += 1
            print("CV %s: %d%%" % (
                model, tmp * 100 / C_grid_size))
            stdout.flush()

            learners = [
                LogisticRegression(penalty='group-L1', solver='svrg', C=C_,
                              verbose=False, step=1e-3,
                              blocks_start=binarizer.feature_indices[:-1, ],
                              blocks_length=binarizer.n_values)
                for _ in range(K)]
            auc = compute_score(learners, X_bin, y[:n_restrict_cv], K,
                                verbose=False)[0]

            avg_scores = np.append(avg_scores, max(auc, 1 - auc))

            learner = LogisticRegression(penalty='group-L1', solver='svrg',
                                    C=C_, verbose=False, step=1e-3,
                                    blocks_start=binarizer.feature_indices[
                                                 :-1, ],
                                    blocks_length=binarizer.n_values)
            learner.fit(X_bin, y[:n_restrict_cv])
            y_pred = learner.predict_proba(X_test_bin)[:, 1]
            score_test.append(roc_auc_score(y_test[:n_restrict_cv], y_pred))

        idx_best = np.unravel_index(avg_scores.argmax(),
                                    avg_scores.shape)[0]
        C_best = C_grid[idx_best]
        if selection == 'min':
            C_chosen = C_best
        if selection == '1st':
            max_ = avg_scores.max()
            min_ = avg_scores.min()
            idx = [i for i, is_up in enumerate(
                list(avg_scores >= max_ - .05 * (max_ - min_)))
                   if is_up]
            idx_chosen = min(idx) if len(idx) > 0 else idx_best
            C_chosen = C_grid[idx_chosen]

        # learning curves
        learning_curves = np.column_stack((C_grid, avg_scores, score_test))
        np.save('./results/learning_curves/6-%s-selection_%s' % (
            model, selection), learning_curves)

        if with_categorical:
            X_final = pd.concat([X_cont, X_cat], axis=1)
            X_test_final = pd.concat([X_test_cont, X_test_cat], axis=1)
        else:
            X_final = X_cont
            X_test_final = X_test_cont

        binarizer = FeaturesBinarizer(n_cuts=n_cuts_chosen)
        binarizer.fit(pd.concat([X_final, X_test_final], axis=0))
        X_bin = binarizer.transform(X_final)
        X_test_bin = binarizer.transform(X_test_final)

        blocks_start = binarizer.feature_indices[:-1, ]
        blocks_length = binarizer.n_values
        np.save('./results/beta/blocks_start-%s' % model, blocks_start)

        learner = LogisticRegression(penalty='group-L1', solver='svrg', C=C_chosen,
                                verbose=False, step=1e-3,
                                blocks_start=blocks_start,
                                blocks_length=blocks_length)
        start = time()
        learner.fit(X_bin, y)
        y_pred = learner.predict_proba(X_test_bin)[:, 1]
        np.save('./results/y_pred/6-%s' % model, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        auc = max(auc, 1 - auc)
        result.append([model.replace('_', ' '), "%g" % auc,
                  "%.3f" % (time() - start)])
        print("\n %s done, AUC: %.3f" % (model, auc))

        # cvg check
        cvg_history = np.column_stack(
            (learner._solver_obj.get_history("n_iter"),
             learner._solver_obj.get_history("obj")))
        np.save('./results/cvg/6-%s' % model, cvg_history)

        coeffs = learner.weights
        np.save('./results/beta/6-%s' % model, coeffs)

    return result


t = PrettyTable(['Algos', 'AUC', 'time'])
start_init = time()


# models = ['group_L1'] # bina
models = ['group_TV']

# models = ['quick_ones', 'bina']
parallel = Parallel(n_jobs=4)
result = parallel(delayed(run_models)(model_) for model_ in models)

for res in result:
    if isinstance(res[0], list):
        for val in res:
            t.add_row(val)
    else:
        t.add_row(res)

# Final performances comparison
print("\n global time: %s s" % (time() - start_init))
print(t)
results = open("./results/results.txt", "w")
results.write('%s' % t)
results.write("\n global time: %s s" % (time() - start_init))
results.close()

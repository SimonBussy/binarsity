import os
import smtplib
import numpy as np
import pandas as pd
import pylab as pl
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE
from sys import stdout, argv
from prettytable import PrettyTable
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from mlpp.inference import LearnerLogReg
from mlpp.preprocessing import FeaturesBinarizer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import indexable
from sklearn.model_selection import check_cv, cross_val_score, \
    StratifiedShuffleSplit
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.externals.joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RandomForest, \
    GradientBoostingClassifier as GradientBoosting
from scipy.stats import randint
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
C_grid = np.logspace(1, 3, C_grid_size)

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
os.system('rm ./results.zip')
os.makedirs('./results/y_pred')
os.makedirs('./results/beta')
os.makedirs('./results/cvg')
os.makedirs('./results/learning_curves')
np.save('./results/y_test', y_test)


def run_models(model_):
    result = list()

    if model_ == 'quick_ones':

        # # logistic regression on raw features, no penalization
        # model = "no_pen_raw_feat"
        # print("\n launch %s" % model)
        # start = time()
        # learner = LearnerLogReg(C=1e10, solver='svrg', step=1e-3)
        # if with_categorical:
        #     X_final = pd.concat([X_std, X_cat_bin], axis=1)
        #     X_test_final = pd.concat([X_test_std, X_test_cat_bin], axis=1)
        # else:
        #     X_final = X_std
        #     X_test_final = X_test_std
        #
        # learner.fit(X_final, y)
        # y_pred = learner.predict_proba(X_test_final)[:, 1]
        #
        # np.save('./results/y_pred/1-%s' % model, y_pred)
        # auc = roc_auc_score(y_test, y_pred)
        # auc = max(auc, 1 - auc)
        #
        # result.append(
        #     [model.replace('_', ' '), "%g" % auc, "%.3f" % (time() - start)])
        # print("\n %s done, AUC: %.3f" % (model, auc))
        #
        # coeffs = learner.coef_
        # np.save('./results/beta/1-%s' % model, coeffs)
        #
        # # cvg check
        # cvg_history = np.column_stack(
        #     (learner._solver_obj.get_history("n_iter"),
        #      learner._solver_obj.get_history("obj")))
        # np.save('./results/cvg/1-%s' % model, cvg_history)

        # logistic regression on raw features, l1 & l2 penalization
        # penalties = ['l2', 'l1']
        penalties = ['l1']
        for penalty in penalties:
            if penalty == 'l2':
                model = "l2_pen_raw_feat"
            else:
                model = "Lasso"
            print("\n launch %s" % model)

            if with_categorical:
                X_final = pd.concat([X_std.iloc[:n_restrict_cv, :],
                                     X_cat_bin.iloc[:n_restrict_cv, :]],
                                    axis=1)
                X_test_final = pd.concat([X_test_std.iloc[:n_restrict_cv, :],
                                          X_test_cat_bin.iloc[:n_restrict_cv,
                                          :]],
                                         axis=1)
            else:
                X_final = X_std.iloc[:n_restrict_cv, :]
                X_test_final = X_test_std.iloc[:n_restrict_cv, :]

            # cross validation on C
            avg_scores, score_test = np.empty(0), []
            for i, C_ in enumerate(C_grid):
                print("CV %s: %d%%" % (model, (i + 1) * 100 / C_grid_size))
                stdout.flush()

                learners = [LearnerLogReg(penalty=penalty, solver='svrg',
                                          C=C_, verbose=False, step=1e-3)
                            for _ in range(K)]
                auc = compute_score(learners, X_final, y[:n_restrict_cv],
                                    K, verbose=False)[0]

                avg_scores = np.append(avg_scores, max(auc, 1 - auc))
                learner = LearnerLogReg(penalty=penalty, solver='svrg',
                                        C=C_, verbose=False, step=1e-3)

                learner.fit(X_final, y[:n_restrict_cv])
                y_pred = learner.predict_proba(X_test_final)[:, 1]
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
            np.save('./results/learning_curves/2-%s-selection_%s' % (
                model, selection),
                    learning_curves)

            start = time()
            learner = LearnerLogReg(penalty=penalty, C=C_chosen, solver='svrg',
                                    step=1e-3)

            if with_categorical:
                X_final = pd.concat([X_std, X_cat_bin], axis=1)
                X_test_final = pd.concat([X_test_std, X_test_cat_bin], axis=1)
            else:
                X_final = X_std
                X_test_final = X_test_std

            learner.fit(X_final, y)
            y_pred = learner.predict_proba(X_test_final)[:, 1]
            np.save('./results/y_pred/2-%s' % model, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            auc = max(auc, 1 - auc)
            result.append([model.replace('_', ' '), "%g" % auc,
                           "%.3f" % (time() - start)])
            print("\n %s done, AUC: %.3f" % (model, auc))

            # cvg check
            cvg_history = np.column_stack(
                (learner._solver_obj.get_history("n_iter"),
                 learner._solver_obj.get_history("obj")))
            np.save('./results/cvg/2-%s' % model, cvg_history)

            coeffs = learner.coef_
            np.save('./results/beta/2-%s' % model, coeffs)

    #     # logistic regression on binarized features, no penalization
    #     model = "no_pen_bin_feat"
    #     print("\n launch %s" % model)
    #
    #     if with_categorical:
    #         X_final = X.iloc[:n_restrict_cv, :]
    #         X_test_final = X_test.iloc[:n_restrict_cv, :]
    #     else:
    #         X_final = X_std.iloc[:n_restrict_cv, :]
    #         X_test_final = X_test_std.iloc[:n_restrict_cv, :]
    #
    #         # cross validation on n_cuts
    #     avg_scores, score_test = np.empty(0), []
    #     for i, n_cuts_ in enumerate(n_cuts_grid):
    #         print("CV %s: %d%%" % (model, i * 100 / n_cuts_grid_size))
    #         stdout.flush()
    #
    #         binarizer = FeaturesBinarizer(n_cuts=n_cuts_)
    #         binarizer.fit(pd.concat([X_final, X_test_final], axis=0))
    #         X_bin = binarizer.transform(X_final)
    #         X_test_bin = binarizer.transform(X_test_final)
    #
    #         learners = [
    #             LearnerLogReg(C=1e10, solver='svrg', verbose=False, step=1e-3)
    #             for _ in range(K)]
    #         auc = compute_score(learners, X_bin, y[:n_restrict_cv],
    #                             K, verbose=False)[0]
    #         avg_scores = np.append(avg_scores, max(auc, 1 - auc))
    #         learner = LearnerLogReg(C=1e10, solver='svrg', verbose=False,
    #                                 step=1e-3)
    #         learner.fit(X_bin, y[:n_restrict_cv])
    #         y_pred = learner.predict_proba(X_test_bin)[:, 1]
    #         score_test.append(roc_auc_score(y_test[:n_restrict_cv], y_pred))
    #
    #     idx_best = np.unravel_index(avg_scores.argmax(), avg_scores.shape)[0]
    #     n_cuts_chosen = n_cuts_grid[idx_best]
    #
    #     # learning curves
    #     learning_curves = np.column_stack((n_cuts_grid, avg_scores, score_test))
    #     np.save('./results/learning_curves/3-%s' % model, learning_curves)
    #
    #     if with_categorical:
    #         X_final = X
    #         X_test_final = X_test
    #     else:
    #         X_final = X_std
    #         X_test_final = X_test_std
    #
    #     binarizer = FeaturesBinarizer(n_cuts=n_cuts_chosen)
    #     binarizer.fit(pd.concat([X_final, X_test_final], axis=0))
    #     X_bin = binarizer.transform(X_final)
    #     X_test_bin = binarizer.transform(X_test_final)
    #
    #     blocks_start = binarizer.feature_indices[:-1, ]
    #     np.save('./results/beta/blocks_start-%s' % model, blocks_start)
    #
    #     start = time()
    #     learner = LearnerLogReg(C=1e10, solver='svrg', step=1e-3)
    #     learner.fit(X_bin, y)
    #     y_pred = learner.predict_proba(X_test_bin)[:, 1]
    #     np.save('./results/y_pred/3-%s' % model, y_pred)
    #     auc = roc_auc_score(y_test, y_pred)
    #     auc = max(auc, 1 - auc)
    #     result.append([model.replace('_', ' '), "%g" % auc,
    #                    "%.3f" % (time() - start)])
    #     print("\n %s done, AUC: %.3f" % (model, auc))
    #
    #     # cvg check
    #     cvg_history = np.column_stack(
    #         (learner._solver_obj.get_history("n_iter"),
    #          learner._solver_obj.get_history("obj")))
    #     np.save('./results/cvg/3-%s' % model, cvg_history)
    #
    #     coeffs = learner.coef_
    #     np.save('./results/beta/3-%s' % model, coeffs)
    #
    # # logistic regression on binarized features, l1 & l2 penalization
    # if model_ in ['l1_bin', 'l2_bin']:
    #
    #     penalty = model_.split('_bin')[0]
    #     if penalty == 'l2':
    #         model = "l2_pen_bin_feat"
    #     else:
    #         model = "l1_pen_bin_feat"
    #     print("\n launch %s" % model)
    #
    #     if with_categorical:
    #         X_final = X.iloc[:n_restrict_cv, :]
    #         X_test_final = X_test.iloc[:n_restrict_cv, :]
    #     else:
    #         X_final = X_std.iloc[:n_restrict_cv, :]
    #         X_test_final = X_test_std.iloc[:n_restrict_cv, :]
    #
    #     # cross validation on C and n_cuts
    #     avg_scores = np.empty((C_grid_size, n_cuts_grid_size))
    #     score_test = np.empty((C_grid_size, n_cuts_grid_size))
    #     tmp = 0
    #     for i, C_ in enumerate(reversed(C_grid)):
    #         for j, n_cuts_ in enumerate(n_cuts_grid):
    #             print("CV %s: %d%%" % (
    #                 model, tmp * 100 / (C_grid_size * n_cuts_grid_size)))
    #             stdout.flush()
    #             tmp += 1
    #
    #             binarizer = FeaturesBinarizer(n_cuts=n_cuts_)
    #             binarizer.fit(pd.concat([X_final, X_test_final], axis=0))
    #             X_bin = binarizer.transform(X_final)
    #             X_test_bin = binarizer.transform(X_test_final)
    #
    #             learners = [LearnerLogReg(penalty=penalty, solver='svrg',
    #                                       C=C_, verbose=False, step=1e-3)
    #                         for _ in range(K)]
    #             auc = compute_score(learners, X_bin, y[:n_restrict_cv], K,
    #                                 verbose=False)[0]
    #             avg_scores[i, j] = max(auc, 1 - auc)
    #             learner = LearnerLogReg(penalty=penalty, solver='svrg',
    #                                     C=C_, verbose=False, step=1e-3)
    #             learner.fit(X_bin, y[:n_restrict_cv])
    #             y_pred = learner.predict_proba(X_test_bin)[:, 1]
    #             score_test[i, j] = roc_auc_score(y_test[:n_restrict_cv], y_pred)
    #
    #     # learning curves
    #     learning_curves = np.column_stack((avg_scores, score_test))
    #     np.save('./results/learning_curves/4-%s' % model, learning_curves)
    #
    #     id_C, id_n_cuts = np.where(avg_scores == avg_scores.max())
    #     C_chosen = C_grid[len(C_grid) - id_C[0] - 1]
    #     n_cuts_chosen = n_cuts_grid[id_n_cuts[0]]
    #
    #     if with_categorical:
    #         X_final = X
    #         X_test_final = X_test
    #     else:
    #         X_final = X_std
    #         X_test_final = X_test_std
    #
    #     binarizer = FeaturesBinarizer(n_cuts=n_cuts_chosen)
    #     binarizer.fit(pd.concat([X_final, X_test_final], axis=0))
    #     X_bin = binarizer.transform(X_final)
    #     X_test_bin = binarizer.transform(X_test_final)
    #
    #     blocks_start = binarizer.feature_indices[:-1, ]
    #     np.save('./results/beta/blocks_start-%s' % model, blocks_start)
    #
    #     start = time()
    #     learner = LearnerLogReg(penalty=penalty, solver='svrg', C=C_chosen,
    #                             verbose=False, step=1e-3)
    #     learner.fit(X_bin, y)
    #     y_pred = learner.predict_proba(X_test_bin)[:, 1]
    #     np.save('./results/y_pred/4-%s' % model, y_pred)
    #
    #     auc = roc_auc_score(y_test, y_pred)
    #     auc = max(auc, 1 - auc)
    #     result = [model.replace('_', ' '), "%g" % auc,
    #               "%.3f" % (time() - start)]
    #     print("\n %s done, AUC: %.3f" % (model, auc))
    #
    #     # cvg check
    #     cvg_history = np.column_stack(
    #         (learner._solver_obj.get_history("n_iter"),
    #          learner._solver_obj.get_history("obj")))
    #     np.save('./results/cvg/4-%s' % model, cvg_history)
    #
    #     coeffs = learner.coef_
    #     np.save('./results/beta/4-%s' % model, coeffs)

    if model_ == 'bina':

        # logistic regression on binarized features, binarsity penalization
        model = "bina_pen_bin_feat"
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
                LearnerLogReg(penalty='binarsity', solver='svrg', C=C_,
                              verbose=False, step=1e-3,
                              blocks_start=binarizer.feature_indices[:-1, ],
                              blocks_length=binarizer.n_values)
                for _ in range(K)]
            auc = compute_score(learners, X_bin, y[:n_restrict_cv], K,
                                verbose=False)[0]

            avg_scores = np.append(avg_scores, max(auc, 1 - auc))

            learner = LearnerLogReg(penalty='binarsity', solver='svrg',
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

        learner = LearnerLogReg(penalty='binarsity', solver='svrg', C=C_chosen,
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

        # # save feature names
        # features_names = []
        # for i, column_name in enumerate(original_feature_names):
        #     for n in range(1, binarizer.n_values[i] + 1):
        #         features_names.append(str(column_name) + '#%s' % n)
        # np.save('./results/beta/features_names_bina', np.array(features_names))

        # cvg check
        cvg_history = np.column_stack(
            (learner._solver_obj.get_history("n_iter"),
             learner._solver_obj.get_history("obj")))
        np.save('./results/cvg/5-%s' % model, cvg_history)

        coeffs = learner.coef_
        np.save('./results/beta/5-%s' % model, coeffs)

        # # LR bina support
        # model = "LR_bina"
        #
        # bins_boundaries = binarizer.bins_boundaries
        # coef = learner.coef_
        #
        # categorical_X = np.empty_like(X_cont)
        # categorical_X_test = np.empty_like(X_test_cont)
        #
        # id_col = 0
        # for i in range(X_cont.shape[1]):
        #     feature = X_cont.iloc[:, i]
        #     feature_test = X_test_cont.iloc[:, i]
        #
        #     boundaries = bins_boundaries[feature_names_cont[i]]
        #
        #     cut_points_idx = list()
        #     for idx, j in enumerate(
        #             range(blocks_start[i],
        #                   blocks_start[i] + blocks_length[i] - 1)):
        #         if coef[j + 1] != coef[j]:
        #             cut_points_idx.append(idx + 1)
        #
        #     if len(cut_points_idx) > 0:
        #         boundaries = list(boundaries[cut_points_idx])
        #         boundaries.insert(0, -np.inf)
        #         boundaries.append(np.inf)
        #
        #         binarized_feat = pd.cut(feature, boundaries, labels=False)
        #         binarized_feat_test = pd.cut(feature_test, boundaries,
        #                                      labels=False)
        #
        #         categorical_X[:, id_col] = binarized_feat
        #         categorical_X_test[:, id_col] = binarized_feat_test
        #         id_col += 1
        #
        # categorical_X = categorical_X[:, :id_col]
        # categorical_X_test = categorical_X_test[:, :id_col]
        #
        # # if with_categorical:
        # #     categorical_X = np.concatenate((categorical_X,
        # #                                     np.array(X_cat)),
        # #                                    axis=1)
        # #     categorical_X_test = np.concatenate((categorical_X_test,
        # #                                          np.array(X_test_cat)),
        # #                                    axis=1)
        #
        # one_hot_encoder = OneHotEncoder(sparse=True)
        # one_hot_encoder.fit(categorical_X)
        #
        # blocks_start = one_hot_encoder.feature_indices_[:-1, ]
        # np.save('./results/beta/blocks_start-%s' % model, blocks_start)
        #
        # X_bin_try = one_hot_encoder.transform(categorical_X)
        # X_test_bin_try = one_hot_encoder.transform(categorical_X_test)
        #
        # learner = LearnerLogReg(C=1e10, solver='svrg', step=1e-3)
        #
        # learner.fit(X_bin_try, y)
        # y_pred = learner.predict_proba(X_test_bin_try)[:, 1]
        #
        # np.save('./results/y_pred/1-%s' % model, y_pred)
        # auc = roc_auc_score(y_test, y_pred)
        # auc = max(auc, 1 - auc)
        #
        # result.append(
        #     [model.replace('_', ' '), "%g" % auc, "%.3f" % (time() - start)])
        # print("\n %s done, AUC: %.3f" % (model, auc))

    if model_ == 'svm_rbf':
        # svm RBF on raw features
        model = "svm_rbf"

        C_grid_size_ = 8
        gamma_grid_size = 8
        if test:
            C_grid_size_ = 2
            gamma_grid_size = 2
        C_grid_ = np.logspace(-2, 5, C_grid_size_)
        gamma_grid = np.logspace(-7, 1, gamma_grid_size)

        # use only 5000 examples for cv in svm
        n_restrict_svm_cv = 5000

        if with_categorical:
            X_final = pd.concat([X_std.iloc[:n_restrict_svm_cv, :],
                                 X_cat_bin.iloc[:n_restrict_svm_cv, :]],
                                axis=1)
        else:
            X_final = X_std.iloc[:n_restrict_svm_cv, :]

        param_grid = dict(gamma=gamma_grid, C=C_grid_)
        cv = StratifiedKFold(y[:n_restrict_svm_cv], n_folds=K, shuffle=True)
        grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv,
                            scoring="roc_auc", n_jobs=8, verbose=20)
        grid.fit(X_final, y[:n_restrict_svm_cv])

        # learning curves
        np.save('./results/learning_curves/6-%s' % model, grid.grid_scores_)

        start = time()
        learner = SVC(C=grid.best_params_['C'],
                      gamma=grid.best_params_['gamma'], probability=True)

        n_restrict_svm = 20000

        if with_categorical:
            X_final = pd.concat([X_std.iloc[:n_restrict_svm, :],
                                 X_cat_bin.iloc[:n_restrict_svm, :]],
                                axis=1)
            X_test_final = pd.concat([X_test_std.iloc[:n_restrict_svm, :],
                                      X_test_cat_bin.iloc[:n_restrict_svm, :]],
                                     axis=1)
        else:
            X_final = X_std.iloc[:n_restrict_svm, :]
            X_test_final = X_test_std.iloc[:n_restrict_svm, :]

        learner.fit(X_final, y[:n_restrict_svm])
        y_pred = learner.predict_proba(X_test_final)[:, 1]
        np.save('./results/y_pred/6-%s' % model, y_pred)
        auc = roc_auc_score(y_test[:n_restrict_svm], y_pred)
        auc = max(auc, 1 - auc)

        result = [model.replace('_', ' '), "%g" % auc,
                  "%.3f" % (time() - start)]
        print("\n %s done, AUC: %.3f" % (model, auc))

    if model_ == 'rf':
        # random forest on raw features
        model = "random_forest"

        learner = RandomForest(n_jobs=5)
        param_dist = {"max_depth": [None, 3, 5, 10, 20, 30, 50],
                      "min_samples_split": randint(2, X.shape[1]),
                      "min_samples_leaf": randint(2, 20),
                      "n_estimators": [200, 300, 500, 800],
                      "bootstrap": [True, False],
                      "criterion": ["entropy"]}

        cv = StratifiedKFold(y[:n_restrict_cv], n_folds=K, shuffle=True)
        n_iter_search = 40
        if test:
            n_iter_search = 3

        if with_categorical:
            X_final = pd.concat([X_cont.iloc[:n_restrict_cv, :],
                                 X_cat_bin.iloc[:n_restrict_cv, :]],
                                axis=1)
        else:
            X_final = X_cont.iloc[:n_restrict_cv, :]

        search = RandomizedSearchCV(learner,
                                    param_distributions=param_dist,
                                    n_iter=n_iter_search, scoring="roc_auc",
                                    verbose=10, n_jobs=10, cv=cv)
        search.fit(X_final, y[:n_restrict_cv])
        start = time()
        learner = RandomForest(max_depth=search.best_params_['max_depth'],
                               min_samples_split=
                               search.best_params_[
                                   'min_samples_split'],
                               min_samples_leaf=
                               search.best_params_[
                                   'min_samples_leaf'],
                               bootstrap=search.best_params_[
                                   'bootstrap'],
                               criterion=search.best_params_[
                                   'criterion'],
                               n_estimators=search.best_params_[
                                   'n_estimators'])

        infos = "max_depth: %s, min_samples_split: %s, min_samples_leaf: %s," \
                " bootstrap: %s, criterion: %s, n_estimators: %s" \
                % (search.best_params_['max_depth'],
                   search.best_params_['min_samples_split'],
                   search.best_params_['min_samples_leaf'],
                   search.best_params_['bootstrap'],
                   search.best_params_['criterion'],
                   search.best_params_['n_estimators'])

        print(infos)
        infos_rf = open("./results/infos_rf.txt", "w")
        infos_rf.write('%s' % infos)
        infos_rf.close()

        if with_categorical:
            X_final = pd.concat([X_cont, X_cat_bin], axis=1)
            X_test_final = pd.concat([X_test_cont, X_test_cat_bin], axis=1)
        else:
            X_final = X_cont
            X_test_final = X_test_cont

        learner.fit(X_final, y)
        y_pred = learner.predict_proba(X_test_final)[:, 1]
        np.save('./results/y_pred/7-%s' % model, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        auc = max(auc, 1 - auc)

        result = [model.replace('_', ' '), "%g" % auc,
                  "%.3f" % (time() - start)]
        print("\n %s done, AUC: %.3f" % (model, auc))

    if model_ == 'gb':
        # gradient boosting on raw features
        model = "gradient_boosting"

        learner = GradientBoosting()
        param_dist = {"max_depth": [None, 3, 5, 10, 20, 30, 50],
                      "min_samples_split": randint(2, X.shape[1]),
                      "min_samples_leaf": randint(2, 20),
                      "n_estimators": [200, 300, 500, 800]}

        cv = StratifiedKFold(y[:n_restrict_cv], n_folds=K, shuffle=True)
        n_iter_search = 40
        if test:
            n_iter_search = 3
        search = RandomizedSearchCV(learner,
                                    param_distributions=param_dist,
                                    n_iter=n_iter_search, scoring="roc_auc",
                                    verbose=10, n_jobs=10, cv=cv)

        if with_categorical:
            X_final = pd.concat([X_cont.iloc[:n_restrict_cv, :],
                                 X_cat_bin.iloc[:n_restrict_cv, :]],
                                axis=1)
        else:
            X_final = X_cont.iloc[:n_restrict_cv, :]

        search.fit(X_final, y[:n_restrict_cv])
        start = time()
        learner = GradientBoosting(max_depth=search.best_params_['max_depth'],
                                   min_samples_split=
                                   search.best_params_[
                                       'min_samples_split'],
                                   min_samples_leaf=
                                   search.best_params_[
                                       'min_samples_leaf'],
                                   n_estimators=search.best_params_[
                                       'n_estimators'])

        infos = "max_depth: %s, min_samples_split: %s, min_samples_leaf: %s," \
                " n_estimators: %s" \
                % (search.best_params_['max_depth'],
                   search.best_params_['min_samples_split'],
                   search.best_params_['min_samples_leaf'],
                   search.best_params_['n_estimators'])

        print(infos)
        infos_rf = open("./results/infos_gb.txt", "w")
        infos_rf.write('%s' % infos)
        infos_rf.close()

        if with_categorical:
            X_final = pd.concat([X_cont, X_cat_bin], axis=1)
            X_test_final = pd.concat([X_test_cont, X_test_cat_bin], axis=1)
        else:
            X_final = X_cont
            X_test_final = X_test_cont

        learner.fit(X_final, y)
        y_pred = learner.predict_proba(X_test_final)[:, 1]
        np.save('./results/y_pred/8-%s' % model, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        auc = max(auc, 1 - auc)

        result = [model.replace('_', ' '), "%g" % auc,
                  "%.3f" % (time() - start)]
        print("\n %s done, AUC: %.3f" % (model, auc))

    return result


t = PrettyTable(['Algos', 'AUC', 'time'])
start_init = time()

#models = ['quick_ones', 'l1_bin', 'l2_bin', 'bina']
models = ['quick_ones']
parallel = Parallel(n_jobs=4)
result = parallel(delayed(run_models)(model_) for model_ in models)

for res in result:
    if isinstance(res[0], list):
        for val in res:
            t.add_row(val)
    else:
        t.add_row(res)

K = 5
#result = run_models('svm_rbf')
#t.add_row(result)
#result = run_models('rf')
#t.add_row(result)
#result = run_models('gb')
#t.add_row(result)

# Final performances comparison
print("\n global time: %s s" % (time() - start_init))
print(t)
results = open("./results/results.txt", "w")
results.write('%s' % t)
results.write("\n global time: %s s" % (time() - start_init))
results.close()

# compress results and send it by email
os.system('say "computation finished"')
os.system('zip -r results.zip results')

send_from = 'simon.bussy@upmc.fr'
send_to = ['simon.bussy@gmail.com']
subject = "computation finished for %s" % filename
text = "results available \n"
files = "./results.zip"

msg = MIMEMultipart()
msg['From'] = send_from
msg['To'] = COMMASPACE.join(send_to)
msg['Subject'] = subject

msg.attach(MIMEText(text))

with open(files, "rb") as fil:
    part = MIMEApplication(
        fil.read(),
        Name="result_%s.zip" % filename
    )
    part[
        'Content-Disposition'] = 'attachment; filename=' \
                                 '"result_%s.zip"' % filename
    msg.attach(part)

try:
    smtp = smtplib.SMTP('smtp.upmc.fr')
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()
    print("Successfully sent email")
except smtplib.SMTPException:
    print("Error: unable to send email")

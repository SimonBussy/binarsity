import os
import smtplib
import numpy as np
import pandas as pd
import pylab as pl
from os.path import basename
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
from mlpp.inference import LearnerLogReg
from mlpp.preprocessing import FeaturesBinarizer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import indexable
from sklearn.model_selection import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.externals.joblib import Parallel, delayed
import multiprocessing as mp

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
    n_cuts_min = 4
    n_cuts_max = 50
    n_cuts_grid_size = 18

    try:
        n_cuts_min = int(argv[6])
        n_cuts_max = int(argv[7])
        n_cuts_grid_size = int(argv[8])
    except:
        pass

else:
    raise ValueError("at least 4 command-line arguments expected, %s given",
                     len(argv) - 1)


def cross_val_score(estimators, X, y=None, groups=None, scoring=None,
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
    scores = cross_val_score(clf, X, y, cv=K, verbose=0,
                             n_jobs=1, scoring="roc_auc",
                             fit_params=fit_params)
    score_mean = scores.mean()
    score_std = 2 * scores.std()
    if verbose:
        print("\n AUC: %0.3f (+/- %0.3f)" % (score_mean, score_std))
    return score_mean, score_std


if test:
    C_grid_size = 4
    n_cuts_grid_size = 3
else:
    C_grid_size = 20

n_cuts_grid = np.linspace(n_cuts_min, n_cuts_max, n_cuts_grid_size, dtype=int)
C_grid = np.logspace(1.5, 4, C_grid_size)

# drop lines with NaN values
df.dropna(axis=0, how='any', inplace=True)

# get label (have to be the last column!)
idx_label_column = -1
labels = df.iloc[:, idx_label_column]
labels = 2 * (labels.values != labels.values[0]) - 1
# drop it from df
df = df.drop(df.columns[[idx_label_column]], axis=1)

# continuous features only
to_be_dropped = []
for i in range(df.shape[1]):
    feature_type = FeaturesBinarizer._detect_feature_type(df.ix[:, i])
    if feature_type == 'discrete':
        to_be_dropped.append(i)
df = df.drop(df.columns[to_be_dropped], axis=1)

original_feature_names = df.columns

# shuffle and split training and test sets
X, X_test, y, y_test = train_test_split(
    df, labels, test_size=.33, random_state=0, stratify=labels)

del df

if test:
    n_restrict = 5000
    X = np.array(X)[:n_restrict, :]
    y = y[:n_restrict]
    X_test = np.array(X_test)[:n_restrict, :]
    y_test = y_test[:n_restrict]

print("Training:")
print(X.shape)
print("Test:")
print(X_test.shape)

# Center and reduce data
standardscaler = StandardScaler()
X_std = standardscaler.fit_transform(X)
X_test_std = standardscaler.transform(X_test)
print("data centered and reduced")

os.system('rm -rR ./results')
os.system('rm ./results.zip')
os.makedirs('./results/y_pred')
os.makedirs('./results/beta')
os.makedirs('./results/cvg')
os.makedirs('./results/learning_curves')


def run_models(model_):
    if model_ == 'quick_ones':

        # logistic regression on raw features, no penalization
        model = "no_pen_raw_feat"
        print("\n launch %s" % model)
        start = time()
        learner = LearnerLogReg(C=1e10, solver='svrg', step=1e-3)
        learner.fit(X_std, y)
        y_pred = learner.predict_proba(X_test_std)[:, 1]

        np.save('./results/y_test', y_test)
        np.save('./results/y_pred/1-%s' % model, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        auc = max(auc, 1 - auc)

        result = list()
        result.append(
            [model.replace('_', ' '), "%g" % auc, "%.3f" % (time() - start)])
        print("\n %s done, AUC: %.3f" % (model, auc))

        coeffs = learner.coef_
        np.save('./results/beta/1-%s' % model, coeffs)

        # cvg check
        cvg_history = np.column_stack(
            (learner._solver_obj.get_history("n_iter"),
             learner._solver_obj.get_history("obj")))
        np.save('./results/cvg/1-%s' % model, cvg_history)

        # logistic regression on raw features, l1 & l2 penalization
        penalties = ['l2', 'l1']
        for penalty in penalties:
            if penalty == 'l2':
                model = "l2_pen_raw_feat"
            else:
                model = "l1_pen_raw_feat"
            print("\n launch %s" % model)

            # cross validation on C
            avg_scores, score_test = np.empty(0), []
            for i, C_ in enumerate(C_grid):
                print("CV %s: %d%%" % (model, (i + 1) * 100 / C_grid_size))
                stdout.flush()

                learners = [LearnerLogReg(penalty=penalty, solver='svrg',
                                          C=C_, verbose=False, step=1e-3)
                            for _ in range(K)]
                auc = compute_score(learners, X_std, y, K, verbose=False)[0]

                avg_scores = np.append(avg_scores, max(auc, 1 - auc))
                learner = LearnerLogReg(penalty=penalty, solver='svrg',
                                        C=C_, verbose=False, step=1e-3)
                learner.fit(X_std, y)
                y_pred = learner.predict_proba(X_test_std)[:, 1]
                score_test.append(roc_auc_score(y_test, y_pred))

            idx_best = np.unravel_index(avg_scores.argmax(), avg_scores.shape)[
                0]
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
            learner.fit(X_std, y)
            y_pred = learner.predict_proba(X_test_std)[:, 1]
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

        # logistic regression on binarized features, no penalization
        model = "no_pen_bin_feat"
        print("\n launch %s" % model)
        # cross validation on n_cuts
        avg_scores, score_test = np.empty(0), []
        for i, n_cuts_ in enumerate(n_cuts_grid):
            print("CV %s: %d%%" % (model, i * 100 / n_cuts_grid_size))
            stdout.flush()

            binarizer = FeaturesBinarizer(n_cuts=n_cuts_)
            X_bin = binarizer.fit_transform(X)
            X_test_bin = binarizer.transform(X_test)
            learners = [
                LearnerLogReg(C=1e10, solver='svrg', verbose=False, step=1e-3)
                for _ in range(K)]
            auc = compute_score(learners, X_bin, y, K, verbose=False)[0]
            avg_scores = np.append(avg_scores, max(auc, 1 - auc))
            learner = LearnerLogReg(C=1e10, solver='svrg', verbose=False,
                                    step=1e-3)
            learner.fit(X_bin, y)
            y_pred = learner.predict_proba(X_test_bin)[:, 1]
            score_test.append(roc_auc_score(y_test, y_pred))

        idx_best = np.unravel_index(avg_scores.argmax(), avg_scores.shape)[0]
        n_cuts_chosen = n_cuts_grid[idx_best]

        # learning curves
        learning_curves = np.column_stack((n_cuts_grid, avg_scores, score_test))
        np.save('./results/learning_curves/3-%s' % model, learning_curves)

        binarizer = FeaturesBinarizer(n_cuts=n_cuts_chosen)
        X_bin = binarizer.fit_transform(X)
        X_test_bin = binarizer.transform(X_test)

        blocks_start = binarizer.feature_indices[:-1, ]
        np.save('./results/beta/blocks_start-%s' % model, blocks_start)

        start = time()
        learner = LearnerLogReg(C=1e10, solver='svrg', step=1e-3)
        learner.fit(X_bin, y)
        y_pred = learner.predict_proba(X_test_bin)[:, 1]
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
        np.save('./results/cvg/3-%s' % model, cvg_history)

        coeffs = learner.coef_
        np.save('./results/beta/3-%s' % model, coeffs)

    # logistic regression on binarized features, l1 & l2 penalization
    if model_ in ['l1_bin', 'l2_bin']:

        penalty = model_.split('_bin')[0]
        if penalty == 'l2':
            model = "l2_pen_bin_feat"
        else:
            model = "l1_pen_bin_feat"
        print("\n launch %s" % model)

        # cross validation on C and n_cuts
        avg_scores = np.empty((C_grid_size, n_cuts_grid_size))
        score_test = np.empty((C_grid_size, n_cuts_grid_size))
        tmp = 0
        for i, C_ in enumerate(reversed(C_grid)):
            for j, n_cuts_ in enumerate(n_cuts_grid):
                print("CV %s: %d%%" % (
                    model, tmp * 100 / (C_grid_size * n_cuts_grid_size)))
                stdout.flush()
                tmp += 1

                binarizer = FeaturesBinarizer(n_cuts=n_cuts_)
                X_bin = binarizer.fit_transform(X)
                X_test_bin = binarizer.transform(X_test)

                learners = [LearnerLogReg(penalty=penalty, solver='svrg',
                                          C=C_, verbose=False, step=1e-3)
                            for _ in range(K)]
                auc = compute_score(learners, X_bin, y, K, verbose=False)[0]
                avg_scores[i, j] = max(auc, 1 - auc)
                learner = LearnerLogReg(penalty=penalty, solver='svrg',
                                        C=C_, verbose=False, step=1e-3)
                learner.fit(X_bin, y)
                y_pred = learner.predict_proba(X_test_bin)[:, 1]
                score_test[i, j] = roc_auc_score(y_test, y_pred)

        # learning curves
        learning_curves = np.column_stack((avg_scores, score_test))
        np.save('./results/learning_curves/4-%s' % model, learning_curves)

        id_C, id_n_cuts = np.where(avg_scores == avg_scores.max())
        C_chosen = C_grid[len(C_grid) - id_C[0] - 1]
        n_cuts_chosen = n_cuts_grid[id_n_cuts[0]]

        binarizer = FeaturesBinarizer(n_cuts=n_cuts_chosen)
        X_bin = binarizer.fit_transform(X)
        X_test_bin = binarizer.transform(X_test)

        blocks_start = binarizer.feature_indices[:-1, ]
        np.save('./results/beta/blocks_start-%s' % model, blocks_start)

        start = time()
        learner = LearnerLogReg(penalty=penalty, solver='svrg', C=C_chosen,
                                verbose=False, step=1e-3)
        learner.fit(X_bin, y)
        y_pred = learner.predict_proba(X_test_bin)[:, 1]
        np.save('./results/y_pred/4-%s' % model, y_pred)

        auc = roc_auc_score(y_test, y_pred)
        auc = max(auc, 1 - auc)
        result = [model.replace('_', ' '), "%g" % auc,
                  "%.3f" % (time() - start)]
        print("\n %s done, AUC: %.3f" % (model, auc))

        # cvg check
        cvg_history = np.column_stack(
            (learner._solver_obj.get_history("n_iter"),
             learner._solver_obj.get_history("obj")))
        np.save('./results/cvg/4-%s' % model, cvg_history)

        coeffs = learner.coef_
        np.save('./results/beta/4-%s' % model, coeffs)

    if model_ == 'bina':

        # logistic regression on binarized features, binarsity penalization
        model = "bina_pen_bin_feat"
        print("\n launch %s" % model)

        # cross validation on C and n_cuts
        avg_scores = np.empty((C_grid_size, n_cuts_grid_size))
        score_test = np.empty((C_grid_size, n_cuts_grid_size))
        tmp = 0
        for i, C_ in enumerate(reversed(C_grid)):
            for j, n_cuts_ in enumerate(n_cuts_grid):
                
                tmp += 1
                print("CV %s: %d%%" % (
                    model, tmp * 100 / (C_grid_size * n_cuts_grid_size)))
                stdout.flush()

                binarizer = FeaturesBinarizer(n_cuts=n_cuts_)
                X_bin = binarizer.fit_transform(X)

                learners = [
                    LearnerLogReg(penalty='binarsity', solver='svrg', C=C_,
                                  verbose=False, step=1e-3,
                                  blocks_start=binarizer.feature_indices[:-1, ],
                                  blocks_length=binarizer.n_values)
                    for _ in range(K)]
                auc = compute_score(learners, X_bin, y, K, verbose=False)[0]
                avg_scores[i, j] = max(auc, 1 - auc)

                learner = LearnerLogReg(penalty='binarsity', solver='svrg',
                                        C=C_,
                                        verbose=False, step=1e-3,
                                        blocks_start=binarizer.feature_indices[
                                                     :-1, ],
                                        blocks_length=binarizer.n_values)
                learner.fit(X_bin, y)
                X_test_bin = binarizer.transform(X_test)
                y_pred = learner.predict_proba(X_test_bin)[:, 1]
                score_test[i, j] = roc_auc_score(y_test, y_pred)

        # learning curves
        learning_curves = np.column_stack((avg_scores, score_test))
        np.save('./results/learning_curves/5-%s' % model, learning_curves)

        id_C, id_n_cuts = np.where(avg_scores == avg_scores.max())
        C_chosen = C_grid[len(C_grid) - id_C[0] - 1]
        n_cuts_chosen = n_cuts_grid[id_n_cuts[0]]

        binarizer = FeaturesBinarizer(n_cuts=n_cuts_chosen)
        X_bin = binarizer.fit_transform(X)

        blocks_start = binarizer.feature_indices[:-1, ]
        blocks_length = binarizer.n_values
        np.save('./results/beta/blocks_start-%s' % model, blocks_start)

        learner = LearnerLogReg(penalty='binarsity', solver='svrg', C=C_chosen,
                                verbose=False, step=1e-3,
                                blocks_start=blocks_start,
                                blocks_length=blocks_length)
        start = time()
        learner.fit(X_bin, y)
        X_test_bin = binarizer.transform(X_test)
        y_pred = learner.predict_proba(X_test_bin)[:, 1]
        np.save('./results/y_pred/5-%s' % model, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        auc = max(auc, 1 - auc)
        result = [model.replace('_', ' '), "%g" % auc,
                  "%.3f" % (time() - start)]
        print("\n %s done, AUC: %.3f" % (model, auc))

        # save feature names
        features_names = []
        for i, column_name in enumerate(original_feature_names):
            for n in range(1, binarizer.n_values[i] + 1):
                features_names.append(str(column_name) + '#%s' % n)
        np.save('./results/beta/features_names_bina', np.array(features_names))

        # cvg check
        cvg_history = np.column_stack(
            (learner._solver_obj.get_history("n_iter"),
             learner._solver_obj.get_history("obj")))
        np.save('./results/cvg/5-%s' % model, cvg_history)

        coeffs = learner.coef_
        np.save('./results/beta/5-%s' % model, coeffs)

    if model_ == 'svm_rbf':

        svc = svm.SVC()
        # C_range = np.logspace(-2, 10, 13)
        # gamma_range = np.logspace(-9, 3, 13)
        #
        # scores = list()
        # scores_std = list()
        # for C in C_s:
        #     svc.C = C
        #     this_scores = cross_val_score(svc, X, y, n_jobs=1)
        #     scores.append(np.mean(this_scores))
        #     scores_std.append(np.std(this_scores))

    return result


models = ['quick_ones', 'l1_bin', 'l2_bin', 'bina']
t = PrettyTable(['Algos', 'AUC', 'time'])
start_init = time()

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

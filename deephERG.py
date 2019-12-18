import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))+'/mol2vec')
import mol2vec
from mol2vec.features import mol2alt_sentence
import deepchem as dc
from deepchem.utils.save import load_from_disk
from deepchem.utils.evaluate import Evaluator
from datetime import datetime
import numpy as np
np.set_printoptions(threshold=np.nan)


print("1.multi-Task-DNN")
print("2.single-Task-DNN")
print("3.all above")
method = input("Input the number of DNN methods and press ENTER to continue..")
method = str(method)

# input file format: *.sdf
# featurization
mol2vec.features.featurize('input_file/testset.sdf', 'output_file/testset_mol2vec.csv', 'input_file/model_300dim.pkl', 1, uncommon=None)
mol2vec.features.featurize('input_file/validationset.sdf', 'output_file/validationset_mol2vec.csv', 'input_file/model_300dim.pkl', 1, uncommon=None)
mol2vec.features.featurize('input_file/trainingset.sdf', 'output_file/trainingset_mol2vec.csv', 'input_file/model_300dim.pkl', 1, uncommon=None)
user_specified_features = ['mol2vec-000', 'mol2vec-001', 'mol2vec-002', 'mol2vec-003', 'mol2vec-004', 'mol2vec-005',
                           'mol2vec-006', 'mol2vec-007', 'mol2vec-008', 'mol2vec-009', 'mol2vec-010', 'mol2vec-011',
                           'mol2vec-012', 'mol2vec-013', 'mol2vec-014', 'mol2vec-015', 'mol2vec-016', 'mol2vec-017',
                           'mol2vec-018', 'mol2vec-019', 'mol2vec-020', 'mol2vec-021', 'mol2vec-022', 'mol2vec-023',
                           'mol2vec-024', 'mol2vec-025', 'mol2vec-026', 'mol2vec-027', 'mol2vec-028', 'mol2vec-029',
                           'mol2vec-030', 'mol2vec-031', 'mol2vec-032', 'mol2vec-033', 'mol2vec-034', 'mol2vec-035',
                           'mol2vec-036', 'mol2vec-037', 'mol2vec-038', 'mol2vec-039', 'mol2vec-040', 'mol2vec-041',
                           'mol2vec-042', 'mol2vec-043', 'mol2vec-044', 'mol2vec-045', 'mol2vec-046', 'mol2vec-047',
                           'mol2vec-048', 'mol2vec-049', 'mol2vec-050', 'mol2vec-051', 'mol2vec-052', 'mol2vec-053',
                           'mol2vec-054', 'mol2vec-055', 'mol2vec-056', 'mol2vec-057', 'mol2vec-058', 'mol2vec-059',
                           'mol2vec-060', 'mol2vec-061', 'mol2vec-062', 'mol2vec-063', 'mol2vec-064', 'mol2vec-065',
                           'mol2vec-066', 'mol2vec-067', 'mol2vec-068', 'mol2vec-069', 'mol2vec-070', 'mol2vec-071',
                           'mol2vec-072', 'mol2vec-073', 'mol2vec-074', 'mol2vec-075', 'mol2vec-076', 'mol2vec-077',
                           'mol2vec-078', 'mol2vec-079', 'mol2vec-080', 'mol2vec-081', 'mol2vec-082', 'mol2vec-083',
                           'mol2vec-084', 'mol2vec-085', 'mol2vec-086', 'mol2vec-087', 'mol2vec-088', 'mol2vec-089',
                           'mol2vec-090', 'mol2vec-091', 'mol2vec-092', 'mol2vec-093', 'mol2vec-094', 'mol2vec-095',
                           'mol2vec-096', 'mol2vec-097', 'mol2vec-098', 'mol2vec-099']
featurizer = dc.feat.UserDefinedFeaturizer(user_specified_features)

# Note: training sets for building models, validation sets for final evaluation, test sets  for tuning hyperparameters.
# Load trainingset
trainingset_file = 'output_file/trainingset_mol2vec.csv'
trainingset = dc.utils.save.load_from_disk(trainingset_file)
print("Columns of trainingset: %s" % str(trainingset.columns.values))
print("Number of examples in trainingset: %s" % str(trainingset.shape[0]))

# Load testset
testset_file = 'output_file/testset_mol2vec.csv'
testset = dc.utils.save.load_from_disk(testset_file)
print("Columns of testset: %s" % str(testset.columns.values))
print("Number of examples in testset: %s" % str(testset.shape[0]))

# Load validationset
validationset_file = 'output_file/validationset_mol2vec.csv'
validationset = dc.utils.save.load_from_disk(validationset_file)
print("Columns of validationset: %s" % str(validationset.columns.values))
print("Number of examples in validationset: %s" % str(validationset.shape[0]))

def remove_missing_entries(dataset):
  ##Remove missing entries.
  for i, (X, y, w, ids) in enumerate(dataset.itershards()):
    available_rows = X.any(axis=1)
    print("Shard %d has %d missing entries."
        % (i, np.count_nonzero(~available_rows)))
    X = X[available_rows]
    y = y[available_rows]
    w = w[available_rows]
    ids = ids[available_rows]
    dataset.set_shard(i, X, y, w, ids)

# metrics(multiTask)
roc_auc_metrics = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
accuracy_metrics = dc.metrics.Metric(dc.metrics.accuracy_score, np.mean)
matthews_metrics = dc.metrics.Metric(dc.metrics.matthews_corrcoef, np.mean)
recall_metrics = dc.metrics.Metric(dc.metrics.recall_score, np.mean)
precision_metrics = dc.metrics.Metric(dc.metrics.precision_score, np.mean)

# metrics(singleTask)
roc_auc_score = dc.metrics.Metric(dc.metrics.roc_auc_score)
accuracy_score = dc.metrics.Metric(dc.metrics.accuracy_score)
matthews_corrcoef = dc.metrics.Metric(dc.metrics.matthews_corrcoef)
recall_score = dc.metrics.Metric(dc.metrics.recall_score)
precision_score = dc.metrics.Metric(dc.metrics.precision_score)
all_metrics = [roc_auc_score, accuracy_score, matthews_corrcoef, recall_score, precision_score]

def transformer(dataset_train,dataset_test,dataset_validation):
    transformers_train = [dc.trans.BalancingTransformer(transform_w=True, dataset=dataset_train)]
    print("About to transform training data")
    for transformer in transformers_train:
        dataset_train = transformer.transform(dataset_train)

    transformers_test = [dc.trans.BalancingTransformer(transform_w=True, dataset=dataset_test)]
    print("About to transform test data")
    for transformer in transformers_test:
        dataset_test = transformer.transform(dataset_test)

    transformers_validation = [dc.trans.BalancingTransformer(transform_w=True, dataset=dataset_validation)]
    print("About to transform validation data")
    for transformer in transformers_validation:
        dataset_validation = transformer.transform(dataset_validation)
    return dataset_train,dataset_test,dataset_validation,transformers_train,transformers_test,transformers_validation;


def result(best_model, dataset_train, transformers_train,dataset_test,transformers_test):
    print("Evaluating models")
    train_roc_auc_score, train_pertask_roc_auc_score = best_model.evaluate(
        dataset_train, [roc_auc_metrics], transformers_train, per_task_metrics=True)
    validation_roc_auc_score, validation_pertask_roc_auc_score = best_model.evaluate(
        dataset_validation, [roc_auc_metrics], transformers_validation, per_task_metrics=True)

    train_accuracy_score, train_pertask_accuracy_score = best_model.evaluate(
        dataset_train, [accuracy_metrics], transformers_train, per_task_metrics=True)
    validation_accuracy_score, validation_pertask_accuracy_score = best_model.evaluate(
        dataset_validation, [accuracy_metrics], transformers_validation, per_task_metrics=True)

    train_matthews_score, train_pertask_matthews_score = best_model.evaluate(
        dataset_train, [matthews_metrics], transformers_train, per_task_metrics=True)
    validation_matthews_score, validation_pertask_matthews_score = best_model.evaluate(
        dataset_validation, [matthews_metrics], transformers_validation, per_task_metrics=True)

    train_recall_score, train_pertask_recall_score = best_model.evaluate(
        dataset_train, [recall_metrics], transformers_train, per_task_metrics=True)
    validation_recall_score, validation_pertask_recall_score = best_model.evaluate(
        dataset_validation, [recall_metrics], transformers_validation, per_task_metrics=True)

    train_precision_score, train_pertask_precision_score = best_model.evaluate(
        dataset_train, [precision_metrics], transformers_train, per_task_metrics=True)
    validation_precision_score, validation_pertask_precision_score = best_model.evaluate(
        dataset_validation, [precision_metrics], transformers_validation, per_task_metrics=True)

    print("----------------------------------------------------------------")
    print("Scores for trial %d" % trial)
    print("----------------------------------------------------------------")
    print("train_roc_auc_score: " + str(train_roc_auc_score))
    print("train_pertask_roc_auc_score: " + str(train_pertask_roc_auc_score))
    print("train_accuracy_score: " + str(train_accuracy_score))
    print("train_pertask_accuracy_score: " + str(train_pertask_accuracy_score))
    print("train_matthews_score: " + str(train_matthews_score))
    print("train_pertask_matthews_score: " + str(train_pertask_matthews_score))
    print("train_recall_score: " + str(train_recall_score))
    print("train_pertask_recall_score: " + str(train_pertask_recall_score))
    print("train_precision_score: " + str(train_precision_score))
    print("train_pertask_precision_score: " + str(train_pertask_precision_score))

    print("validation_roc_auc_score: " + str(validation_roc_auc_score))
    print("validation_pertask_roc_auc_score: " + str(validation_pertask_roc_auc_score))
    print("validation_accuracy_score: " + str(validation_accuracy_score))
    print("validation_pertask_accuracy_score: " + str(validation_pertask_accuracy_score))
    print("validation_matthews_score: " + str(validation_matthews_score))
    print("validation_pertask_matthews_score: " + str(validation_pertask_matthews_score))
    print("validation_recall_score: " + str(validation_recall_score))
    print("validation_pertask_recall_score: " + str(validation_pertask_recall_score))
    print("validation_precision_score: " + str(validation_precision_score))
    print("validation_pertask_precision_score: " + str(validation_pertask_precision_score))
    return

if method =="1" or method =="3":
    print("###################Start:multi-Task-DNN##########################")
    shard_size = 2000
    # num_trials means the repeat times of model construction
    num_trials = 1
    tasks = ['activity10', 'activity20', 'activity40', 'activity60', 'activity80', 'activity100']
    loader = dc.data.UserCSVLoader(tasks=tasks, smiles_field='Smiles', id_field="No.", featurizer=featurizer)
    dataset_train = loader.featurize(trainingset_file)
    dataset_test = loader.featurize(testset_file)
    dataset_validation = loader.featurize(validationset_file)
    dataset_train, dataset_test, dataset_validation, transformers_train, transformers_test, transformers_validation = transformer(
        dataset_train, dataset_test, dataset_validation)

    ###Create model###
    n_layers = 3
    n_features = 100
    all_results = []
    for trial in range(num_trials):
        print("Starting trial %d" % trial)
        params_dict = {"activation": ["relu"],
                       "momentum": [.9],
                       "init": ["glorot_uniform"],
                       "learning_rate": [1e-3],
                       "decay": [.0004],
                       "nb_epoch": [20],
                       "nesterov": [False],
                       "nb_layers": [3],
                       "batchnorm": [False],
                       "penalty": [0.],
                       }

        def model_builder(model_params, model_dir):
            model = dc.models.MultitaskClassifier(
                n_tasks=len(tasks),
                n_features=n_features,
                layer_sizes=[200, 100, 50],
                dropouts=[.25] * n_layers,
                weight_init_stddevs=[.02] * n_layers,
                bias_init_consts=[1.] * n_layers,
                batch_size=256,
                optimizer="adam",
                penalty_type="l2"
            )
            return model

        optimizer = dc.hyper.HyperparamOpt(model_builder)
        best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
            params_dict, dataset_train, dataset_test, [], roc_auc_metrics)
        timestamp = str(datetime.now().strftime('%m-%d-%H:%M:%S'))
        result(best_model, dataset_train, transformers_train, dataset_validation, transformers_validation);
        train_csv_out = "output_file/Mol2vec_multitask_train_classifier_"+timestamp+".csv"
        train_stats_out = "output_file/Mol2vec_multitask_train_classifier_stats_"+timestamp+".txt"
        best_train_evaluator = Evaluator(best_model, dataset_train, transformers_train)
        best_train_score = best_train_evaluator.compute_model_performance(
            [roc_auc_metrics], train_csv_out, train_stats_out, per_task_metrics=True)

        validation_csv_out = "output_file/Mol2vec_multitask_validation_classifier_"+timestamp+".csv"
        validation_stats_out = "output_file/Mol2vec_multitask_validation_classifier_stats_"+timestamp+".txt"
        best_validation_evaluator = Evaluator(best_model, dataset_validation, transformers_validation)
        best_validation_score = best_validation_evaluator.compute_model_performance(
            [roc_auc_metrics], validation_csv_out, validation_stats_out, per_task_metrics=True)
        print("##################End:multi-Task-DNN##########################")


if method =="2" or method =="3":
    print("###################Start:single-Task-DNN##########################")
    shard_size = 2000
    # num_trials means the repeat times of model construction
    num_trials = 1
    tasks = ['activity80']
    loader = dc.data.UserCSVLoader(tasks=tasks, featurizer=featurizer, smiles_field='Smiles', id_field="No.")

    dataset_train = loader.featurize(trainingset_file)
    dataset_test = loader.featurize(testset_file)
    dataset_validation = loader.featurize(validationset_file)

    dataset_train, dataset_test, dataset_validation, transformers_train, transformers_test, transformers_validation = transformer(
        dataset_train, dataset_test, dataset_validation)

    ###Create model###
    n_layers = 3
    n_features = 100

    all_results = []


    for trial in range(num_trials):
        print("Starting trial %d" % trial)
        params_dict = {"activation": ["relu"],
                       "momentum": [.9],
                       "init": ["glorot_uniform"],
                       "learning_rate": [1e-3],
                       "decay": [.0004],
                       "nb_epoch": [20],
                       "nesterov": [False],
                       "nb_layers": [3],
                       "batchnorm": [False],
                       "penalty": [0.],
                       }

        def model_builder(model_params, model_dir):
            model = dc.models.MultitaskClassifier(
                n_tasks=len(tasks),
                n_features=n_features,
                layer_sizes=[200, 100, 50],
                dropouts=[.25] * n_layers,
                weight_init_stddevs=[.02] * n_layers,
                bias_init_consts=[1.] * n_layers,
                batch_size=256,
                optimizer="adam",
                penalty_type="l2"
            )
            return model

        optimizer = dc.hyper.HyperparamOpt(model_builder)
        best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
            params_dict, dataset_train, dataset_test, [], roc_auc_metrics)

        timestamp = str(datetime.now().strftime('%m-%d-%H:%M:%S'))

        result(best_model, dataset_train, transformers_train, dataset_validation, transformers_validation);
        train_csv_out = "output_file/Mol2vec_singletask_train_classifier_"+timestamp+".csv"
        train_stats_out = "output_file/Mol2vec_singletask_train_classifier_stats_"+timestamp+".txt"
        best_train_evaluator = Evaluator(best_model, dataset_train, transformers_train)
        best_train_score = best_train_evaluator.compute_model_performance(
            [roc_auc_metrics], train_csv_out, train_stats_out, per_task_metrics=True)

        validation_csv_out = "output_file/Mol2vec_singletask_validation_classifier_"+timestamp+".csv"
        validation_stats_out = "output_file/Mol2vec_singletask_validation_classifier_stats_"+timestamp+".txt"
        best_validation_evaluator = Evaluator(best_model, dataset_validation, transformers_validation)
        best_validation_score = best_validation_evaluator.compute_model_performance(
            [roc_auc_metrics], validation_csv_out, validation_stats_out, per_task_metrics=True)

        print("##################End:single-Task-DNN##########################")

print("##############Finished!#####################")

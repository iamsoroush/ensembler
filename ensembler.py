import os
from itertools import combinations

import numpy as np
import pandas as pd


class Ensembler:
    """Ensembler object.

    Train the models with the same train and validation sets.
    """

    def __init__(self, y_val, val_predictions, test_predictions, model_names=None):
        """Instantiate an ensembler object.

        Args:
            y_val: True labels for validation set, 1D np.ndarray of shape (n_val,)
                Example: np.ndarray([1, 0, 9, 0, 1, 5, ... ])

            val_predictions: Models' predictions for validation data, a 2D np.ndarray of shape (n_models, n_val).
                Example: np.ndarray([[1, 2, 2, 1, 5, ...],
                                     [0, 1, 1, 8, 1, ...],
                                     ...])
            test_predictions: Models' predictions for test data, a 2D np.ndarray of shape (n_models, n_test)
                Example: np.ndarray([[1, 2, 2, 1, 5, ...],
                                     [0, 1, 1, 8, 1, ...],
                                     ...])
            model_names: List of strings for using in returned dataframs. If None, model_names=['model_1', ...]
        """

        self.y_val = y_val
        self.val_predictions = val_predictions
        self.test_predictions = test_predictions
        if model_names is None:
            self.model_names = np.array(['model_{}'.format(i + 1) for i in range(len(val_predictions))])
        else:
            self.model_names = np.array(model_names)

    def make_majority_vote(self):
        """Makes simple majority vote on given models and returns the results."""

        output_submission = pd.DataFrame()
        for i, test_pred in enumerate(self.test_predictions):
            output_submission[self.model_names[i]] = test_pred
        output_submission['Prediction'] = output_submission.mode(axis=1).T.iloc[0].astype(np.int)
        return output_submission

    def get_class_wise_fscores(self):
        f_scores = list()
        for val_pred in self.val_predictions:
            f_scores.append(self._calc_fscore(val_pred, self.y_val))
        return pd.DataFrame(f_scores, index=self.model_names)

    def weighted_ensembling(self, drop_correlated=False, corr_th=0.98, winner_mul=2):
        """Makes weighted ensembling.

        Ensembling is based on class-wise f1-scores obtained from predictions for validation set
         after merging the high correlated models.

        Args:
            drop_correlated: If True, drops the correlated models before doing ensembling.
            corr_th: Models that have higher correlation than this value on test predictions will be merged.
            winner_mul: For each class, the weight of winner model will multiplied by this coeff.
        """

        def weighted_prediction(row, f_scores):
            """Returns final prediction for each test sample.

            Final prediction is based on f_scores obtained for each class for each model.

            Args:
                row: Array of interest. When using on df.apply, pass axis=1
                f_scores: Array of f-scores of models ==> [[fscore_class1_model1, ...], [fscore_class1_model2], ...]
            """

            weights = np.zeros(f_scores.shape[1])
            for i, pred in enumerate(row):
                f_score = f_scores[i][pred]
                weights[pred] = weights[pred] + f_score
            return np.argmax(weights)

        if drop_correlated:
            val_preds, test_preds, model_names = self._merge_corr_models(corr_th)
            if len(model_names) < 2:
                print('All models are fully correlated for corr_th=', corr_th)
                return uncorr_models[0][1]
        else:
            val_preds, test_preds, model_names = self.val_predictions, self.test_predictions, self.model_names
        print('Calculating f-scores ...')
        f_scores = list()
        for val_pred in val_preds:
            f_scores.append(self._calc_fscore(val_pred, self.y_val))
        f_scores = np.array(f_scores)
        per_class_max_indices = (f_scores.argmax(axis=0), list(range(f_scores.shape[1])))
        multipliers = np.ones(f_scores.shape)
        multipliers[per_class_max_indices] = winner_mul
        f_scores = f_scores * multipliers
        print('Ensembling based on calculated f-scores ...')
        final_submission = pd.DataFrame()
        for i, test_pred in enumerate(test_preds):
            final_submission[model_names[i]] = test_pred
        predictions = final_submission.apply(weighted_prediction, axis=1, args=(f_scores,))
        final_submission['Prediction'] = predictions
        return final_submission

    def _merge_corr_models(self, corr_th):
        """Merge high-correlated models based on test-set correlations and return uncorrelated ones.

        Args:
            corr_th: Drop one of two models that have correlation higher than this value.
        """

        n_models = len(self.model_names)
        combs = list(combinations(range(n_models), 2))
        corrs = np.zeros((n_models, n_models))
        for ind1, ind2 in combs:
            corrs[ind1, ind2] = self._calc_corr(ind1, ind2)
        wheres = np.where(corrs > corr_th)
        correlated_indx = list(set(wheres[0]))
        print('Correlation matrix:')
        print(corrs)
        val_preds = np.delete(self.val_predictions, correlated_indx, axis=0)
        test_preds = np.delete(self.test_predictions, correlated_indx, axis=0)
        model_names = np.delete(self.model_names, correlated_indx, axis=0)
        return val_preds, test_preds, model_names

    def _calc_corr(self, model1_ind, model2_ind):
        """Calculates the correlation coefficient between given models on test set.

        Args:
            model1_ind: First model's index.
            model2_ind: Second model's index.
        """

        test_pred_1 = self.test_predictions[model1_ind]
        test_pred_2 = self.test_predictions[model2_ind]
        test_corr = np.abs(np.corrcoef(test_pred_1, test_pred_2)[0, 1])
        return test_corr

    def _calc_fscore(self, y_pred, y_true):
        """Calculates fscore based on given prediction and true values.

        Args:
            y_pred: Predicted values
            y_true: True values
        """

        classes = np.unique(y_true)
        f_scores = list()
        for c in classes:
            tp, fn, fp = self._calc_stats(y_pred, y_true, c=c)
            if tp == 0:
                f_scores.append(0)
                continue
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f_score = 2 * (recall * precision) / (recall + precision)
            f_scores.append(f_score)
        return f_scores

    @staticmethod
    def _calc_stats(y_pred, y_true, c):
        """Calculates true-positive, false-negative and false-positive for given class 'c' .

        Args:
            y_pred: Predicted values
            y_true: True values
            c: Class name
        """

        actual_class = (y_true == c)
        predicted_class = (y_pred == c)
        tp = np.count_nonzero(actual_class & predicted_class)
        fp = np.count_nonzero(~actual_class & predicted_class)
        fn = np.count_nonzero(actual_class & ~predicted_class)
        return tp, fn, fp
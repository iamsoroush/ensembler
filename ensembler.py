class Ensembler:
    def __init__(self, y_val, predictions=list()):
        """Ensembler object.

        Train the models with the same train and validation sets.

        Args:
            y_val: True labels for validation set, must be of shape (n_samples,)
                Example: [1, 0, 9, 0, 1, 5, ... ]
            predictions: A list of model predictions ==> (y_pred_val, y_pred_test)
                Example: [(y_pred_val_1, y_pred_test_1), ...]
                    y_pred_val: np.ndarray([1, 2, 5, 5, 0, 2, ...])
                    y_pred_test: np.ndarray([0, 0, 1, 9, 6, ... ])
        """

        self.y_val = y_val
        self.predictions = predictions

    def make_majority_vote(self):
        """Makes simple majority vote on given models and returns the results."""

        output_submission = pd.DataFrame()
        for i, prediction in enumerate(self.predictions):
            y_pred_test = prediction[1]
            output_submission['model_' + str(i)] = y_pred_test
        output_submission['Prediction'] = output_submission.mode(axis=1).T.iloc[0].astype(np.int)
        return output_submission

    def get_class_wise_fscores(self):
        f_scores = list()
        for model in self.predictions:
            y_pred_val = model[0]
            f_scores.append(self._calc_fscore(y_pred_val, self.y_val))
        return pd.DataFrame(f_scores, index=['model_{}'.format(i) for i in range(len(self.predictions))])

    def weighted_ensembling(self, drop_correlated=False, corr_th=0.98, winner_mul=2):
        """Makes weighted ensembling.

        Ensembling is based on class-wise f1-scores obtained from predictions for validation set
         after merging the high correlated models.

        Args:
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
            models = self._merge_corr_models(corr_th)

            if len(models) < 2:
                print('All models are fully correlated for corr_th=', corr_th)
                return uncorr_models[0][1]
        else:
            models = self.predictions

        print('Calculating f-scores ...')
        f_scores = list()
        for model in models:
            y_pred_val = model[0]
            f_scores.append(self._calc_fscore(y_pred_val, self.y_val))
        f_scores = np.array(f_scores)
        per_class_max_indices = (f_scores.argmax(axis=0), list(range(f_scores.shape[1])))
        multipliers = np.ones(f_scores.shape)
        multipliers[per_class_max_indices] = winner_mul
        f_scores = f_scores * multipliers

        print('Ensembling based on calculated f-scores ...')
        final_submission = pd.DataFrame()
        for i, model in enumerate(models):
            final_submission['model_' + str(i)] = model[1]
        predictions = final_submission.apply(weighted_prediction, axis=1, args=(f_scores,))
        final_submission['Prediction'] = predictions

        return final_submission

    def _merge_corr_models(self, corr_th):
        """Merge high-correlated models based on test-set correlations and return uncorrelated ones.

        Args:
            corr_th: Drop one of two models that have correlation higher than this value.
        """

        n_models = len(self.predictions)
        combs = list(combinations(range(n_models), 2))
        corrs = np.zeros((n_models, n_models))
        for ind1, ind2 in combs:
            corrs[ind1, ind2] = self._calc_corr(ind1, ind2)
        wheres = np.where(corrs > corr_th)
        correlated_indx = list(set(wheres[0]))
        print('Correlation matrix:')
        print(corrs)
        return np.delete(self.predictions, correlated_indx, axis=0)

    def _calc_corr(self, model1_ind, model2_ind):
        """Calculates the correlation coefficient between given models on test set.

        Args:
            model1_ind: First model's index in self.predictions
            model2_ind: Second model's index in self.predictions
        """

        model1 = self.predictions[model1_ind]
        model2 = self.predictions[model2_ind]
        test_corr = np.abs(np.corrcoef(model1[1], model2[1])[0, 1])
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
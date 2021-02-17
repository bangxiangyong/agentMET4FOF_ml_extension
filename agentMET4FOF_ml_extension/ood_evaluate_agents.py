from agentMET4FOF_ml_extension.ml_agents import ML_EvaluateAgent
import numpy as np

from agentMET4FOF_ml_extension.util.calc_auroc import calc_all_scores

class OOD_EvaluateAgent(ML_EvaluateAgent):
    """
    Last piece in the ML-Pipeline to evaluate the model's performance on the datastream.
    If ml_experiment_proxy is specified, this agent will save the results upon finishing.

    """
    parameter_choices = {}

    def init_parameters(self, evaluate_method=[], ml_experiment_proxy=None, **evaluate_params):
        self.ml_experiment_proxy = ml_experiment_proxy

    def on_received_message(self, message):

        if message["channel"] == "test":
            message_data_quantities = message["data"]["quantities"]
            nll_test_mu = message_data_quantities["test"]["nll_mu"]
            nll_test_var = message_data_quantities["test"]["nll_var"]
            y_test_var = message_data_quantities["test"]["y_var"]
            enc_test_var = message_data_quantities["test"]["enc_var"]

            nll_ood_mu = message_data_quantities["ood"]["nll_mu"]
            nll_ood_var = message_data_quantities["ood"]["nll_var"]
            y_ood_var = message_data_quantities["ood"]["y_var"]
            enc_ood_var = message_data_quantities["ood"]["enc_var"]

            max_magnitude = 1000000
            auroc_score_nllmu, gmean_nllmu, aps_nllmu, tpr_new_nllmu, fpr_new_nllmu = calc_all_scores(nll_test_mu.mean(-1), nll_ood_mu.mean(-1))
            auroc_score_nllvar, gmean_nllvar, aps_nllvar, tpr_new_nllvar, fpr_new_nllvar = calc_all_scores(
                np.clip(nll_test_var.mean(-1), -max_magnitude, max_magnitude),
                np.clip(nll_ood_var.mean(-1), -max_magnitude, max_magnitude))
            auroc_score_yvar, gmean_yvar, aps_yvar, tpr_new_yvar, fpr_new_yvar = calc_all_scores(
                np.clip(y_test_var.mean(-1), -max_magnitude, max_magnitude),
                np.clip(y_ood_var.mean(-1), -max_magnitude, max_magnitude))
            auroc_score_enc_var, gmean_enc_var, aps_enc_var, tpr_new_enc_var, fpr_new_enc_var = calc_all_scores(
                np.sum(enc_test_var[1], -1), np.sum(enc_ood_var[1], -1))

            score_nll_mu = {"auroc": auroc_score_nllmu, "gmean": gmean_nllmu, "aps": aps_nllmu,
                            "tpr": tpr_new_nllmu, "fpr": fpr_new_nllmu}
            score_nll_var = {"auroc": auroc_score_nllvar, "gmean": gmean_nllvar, "aps": aps_nllvar,
                             "tpr": tpr_new_nllvar, "fpr": fpr_new_nllvar}
            score_y_var = {"auroc": auroc_score_yvar, "gmean": gmean_yvar, "aps": aps_yvar, "tpr": tpr_new_yvar,
                           "fpr": fpr_new_yvar}
            score_enc_var = {"auroc": auroc_score_enc_var, "gmean": gmean_enc_var, "aps": aps_enc_var,
                             "tpr": tpr_new_enc_var, "fpr": fpr_new_enc_var}

            score_nll_mu = {key+"-nll-mu":val for key,val in score_nll_mu.items()}
            score_nll_var = {key + "-nll-var": val for key, val in score_nll_var.items()}
            score_y_var = {key + "-y-var": val for key, val in score_y_var.items()}
            score_enc_var = {key + "-enc-var": val for key, val in score_enc_var.items()}
            results ={}
            for result in [score_nll_mu,score_nll_var, score_y_var, score_enc_var]:
                results.update(result)

            self.log_info(str(results))
            self.upload_result(results)









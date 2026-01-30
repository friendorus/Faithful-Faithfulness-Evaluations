from mst_xai.evaluation_methods.perturbation_core_code.perturbation_core import perturbation_evaluation


def negative_perturbation_evaluation(**kwargs):
    return perturbation_evaluation(mode="negative", **kwargs)

from sklearn.mixture import GaussianMixture
from helper_functions.normalize import normalize_X
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def gaussian_mixture_model(pixel_seds, sed, *args, norm_method=''):
    X = pixel_seds.T

    X = normalize_X(X, norm_method=norm_method, sed=sed)
    
    range_n_clusters = [2, 4, 6, 8, 10, 12, 14]
    range_n_clusters = [4]

    # for n_clusters in range_n_clusters:
    #     gmm = GaussianMixture(n_components=n_clusters, random_state=0).fit(X) # requires (samples, features). In this case, each pixel is sample and each image is a feature

    # cluster_values = gmm.predict(X) + 1

    param_grid = {
        "n_components": range(1, 7),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    grid_search.fit(X)

    df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_components", "param_covariance_type", "mean_test_score"]
]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )
    df.sort_values(by="BIC score").head()


    sns.catplot(
        data=df,
        kind="bar",
        x="Number of components",
        y="BIC score",
        hue="Type of covariance",
    )
    plt.show()
    breakpoint()
    
    return cluster_values


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_pca(n_components=0.95):

    return PCA(n_components=n_components)


def get_lda():

    return LinearDiscriminantAnalysis()
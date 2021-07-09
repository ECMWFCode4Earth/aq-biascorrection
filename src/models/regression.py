import pickle
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


class ElasticNetRegr(ElasticNet):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        super(ElasticNetRegr, self).__init__(alpha=alpha, l1_ratio=l1_ratio)
        self.scale = StandardScaler()
        self.output_models = ROOT_DIR / "models" / "results" / "ElasticNet"
        self.output_predictions = ROOT_DIR / "data" / "predictions" / "ElasticNet"
        os.makedirs(self.output_models, exist_ok=True)
        os.makedirs(self.output_predictions, exist_ok=True)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, sample_weight=None):
        X = self.scale.fit_transform(X)
        return super(ElasticNetRegr, self).fit(X, y, sample_weight)

    def predict(
        self, 
        X: pd.DataFrame,
        filename: Union[str, path] = None
    ) -> pd.DataFrame:
        X = self.scale.transform(X)
        y_hat = pd.DataFrame(super(ElasticNetRegr, self).predict(X), index=X.index)
        if filename is not None:
            if isinstance(filename, str): filename = Path(filename)
            y_hat.to_csv(self.output_predictions / f"{filename}.csv")
        return y_hat

    def save(self, filename: str) -> NoReturn:
        with open(self.output_models / f"{filename}.pkl", 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self, filename: str) -> NoReturn:
        #  PENDING: Check 
        with open(self.output_models / f"{filename}.pkl", 'wb') as output:
            self = pickle.load(output)
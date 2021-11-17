import multiprocessing
import time
import warnings
from tempfile import mkdtemp

import category_encoders as ce
import joblib
import mlflow
import pandas as pd
from TaxiFareModel.data import get_data, clean_df, DIST_ARGS
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, AddGeohash, Direction, \
    DistanceToCenter
from TaxiFareModel.utils import compute_rmse, simple_time_tracker
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from termcolor import colored
from xgboost import XGBRegressor

MODEL_DIRECTY = "PipelineTest"  # must the same as PATH_TO_MODEL inside Makefile
MLFLOW_URI = "https://mlflow.lewagon.co/"


class Trainer(object):
    ESTIMATOR = "Linear"
    EXPERIMENT_NAME = "TaxifareModel"

    def __init__(self, X, y, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containig all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X:
        :param y:
        :param kwargs:
        """
        self.pipeline = None
        self.kwargs = kwargs
        self.local = kwargs.get("local",
                                False)  # if True training is done locally
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.experiment_name = kwargs.get("experiment_name",
                                          self.EXPERIMENT_NAME)  # cf doc above
        self.model_params = None  # for
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.15)
        self.nrows = self.X_train.shape[0]  # nb of rows to train on
        self.log_kwargs_params()
        self.log_machine_specs()

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "Lasso":
            model = Lasso()
        elif estimator == "Ridge":
            model = Ridge()
        elif estimator == "Linear":
            model = LinearRegression()
        elif estimator == "GBM":
            model = GradientBoostingRegressor()
        elif estimator == "RandomForest":
            model = RandomForestRegressor()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'max_features': ['auto', 'sqrt']
            }
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        elif estimator == "xgboost":
            model = XGBRegressor(objective='reg:squarederror',
                                 n_jobs=-1,
                                 max_depth=10,
                                 learning_rate=0.05,
                                 gamma=3)
            self.model_params = {
                'max_depth': range(10, 20, 2),
                'n_estimators': range(60, 220, 40),
                'learning_rate': [0.1, 0.01, 0.05]
            }
        else:
            model = Lasso()
        estimator_params = self.kwargs.get("estimator_params", {})
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model

    def set_pipeline(self):
        memory = self.kwargs.get("pipeline_memory", None)
        dist = self.kwargs.get("distance_type", "euclidian")
        feateng_steps = self.kwargs.get("feateng",
                                        ["distance", "time_features"])
        if memory:
            memory = mkdtemp()

        # Define feature engineering pipeline blocks here
        pipe_time_features = make_pipeline(
            TimeFeaturesEncoder(time_column='pickup_datetime'),
            OneHotEncoder(handle_unknown='ignore'))
        pipe_distance = make_pipeline(
            DistanceTransformer(distance_type=dist, **DIST_ARGS),
            StandardScaler())
        pipe_geohash = make_pipeline(AddGeohash(), ce.HashingEncoder())
        pipe_direction = make_pipeline(Direction(), StandardScaler())
        pipe_distance_to_center = make_pipeline(DistanceToCenter(),
                                                StandardScaler())

        # Define default feature engineering blocs
        feateng_blocks = [
            ('distance', pipe_distance, list(DIST_ARGS.values())),
            ('time_features', pipe_time_features, ['pickup_datetime']),
            ('geohash', pipe_geohash, list(DIST_ARGS.values())),
            ('direction', pipe_direction, list(DIST_ARGS.values())),
            ('distance_to_center', pipe_distance_to_center,
             list(DIST_ARGS.values())),
        ]
        # Filter out some bocks according to input parameters
        for bloc in feateng_blocks:
            if bloc[0] not in feateng_steps:
                feateng_blocks.remove(bloc)

        features_encoder = ColumnTransformer(feateng_blocks,
                                             n_jobs=None,
                                             remainder="drop")

        self.pipeline = Pipeline(steps=[('features', features_encoder),
                                        ('rgs', self.get_estimator())],
                                 memory=memory)

    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        # mlflow logs
        self.mlflow_log_metric("train_time", int(time.time() - tic))

    def evaluate(self):
        rmse_train = self.compute_rmse(self.X_train, self.y_train)
        self.mlflow_log_metric("rmse_train", rmse_train)
        if self.split:
            rmse_val = self.compute_rmse(self.X_val, self.y_val, show=True)
            self.mlflow_log_metric("rmse_val", rmse_val)
            print(
                colored(
                    "rmse train: {} || rmse val: {}".format(
                        rmse_train, rmse_val), "blue"))
        else:
            print(colored("rmse train: {}".format(rmse_train), "blue"))

    def compute_rmse(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(5), "blue"))
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 3)

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    ### MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key,
                                         value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key,
                                          value)

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    experiment = "taxifare_set_YOURNAME"
    if "YOURNAME" in experiment:
        print(
            colored(
                "Please define MlFlow experiment variable with your own name",
                "red"))
    params = dict(
        nrows=1000000,
        local=False,  # set to False to get data from GCP (Storage or BigQuery)
        optimize=True,
        estimator="xgboost",
        mlflow=True,  # set to True to log params to mlflow
        experiment_name=experiment,
        pipeline_memory=None,
        distance_type="manhattan",
        feateng=[
            "distance_to_center", "direction", "distance", "time_features",
            "geohash"
        ])
    print("############   Loading Data   ############")
    df = get_data(**params)
    df = clean_df(df)
    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)
    del df
    print("shape: {}".format(X_train.shape))
    print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
    # Train and save model, locally and
    t = Trainer(X=X_train, y=y_train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model()

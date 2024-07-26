from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import neps
import logging
from pathlib import Path
from math import log2, ceil

from src.automl.data import Dataset
from src.automl.data_utils import drop_outliers
from src.automl.utils import r2_acc_to_loss

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()

RESULTDIR = FILE.parent.parent.parent / "results"

METRICS = {"r2": r2_score}

ALGORITHMS = {
    #"SK_random_forest": "SK_random_forest_neps_pipeline_space.yaml",
    #"XGB_random_forest": "XGB_random_forest_neps_pipeline_space.yaml",
    "SK_gradient_boost" : "SK_gradient_boost_neps_pipeline_space.yaml",
    #"SK_ada_boost" : "SK_ada_boost_neps_pipeline_space.yaml",
    #"SK_MLP" : "SK_MLP_neps_pipeline_space.yaml",
    #"SK_gaussian_process": "SK_gaussian_process_neps_pipeline_space.yaml",
    #"SK_bayesian_ridge": "SK_bayesian_ridge_neps_pipeline_space.yaml",
    #"SK_elastic_net": "SK_elastic_net_neps_pipeline_space.yaml",
    #"XGB_dart_boost": "XGB_dart_boost_neps_pipeline_space.yaml"
    }

class AutoML:

    def __init__(
        self,
        task_id: str,
        output_path: Path,
        seed: int,
        dataset: Dataset,
        total_evaluations: int = 32,
        metric: str = "r2",
    ) -> None:
        self.task_id = task_id
        self.output_path=output_path,
        self.seed = seed
        self.dataset = dataset
        self.total_evaluations = total_evaluations
        self.selected_algos = list(ALGORITHMS.keys())
        self.metric = METRICS[metric]

    #####################
    # PIPELINES
    #####################
    def regression_pipeline_SK_random_forest(
        self,
        val_split: float,
        iqr_scale: float,
        n_estimators: int,
        criterion: str,
        max_depth : int
        ) -> float:

        run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
            self.dataset.X_train,
            self.dataset.y_train,
            random_state=self.seed,
            test_size=val_split,
        )

        run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, iqr_scale)

        scaler = StandardScaler()
        scaler.fit(run_X_train)

        run_X_train = scaler.transform(run_X_train)
        run_X_val = scaler.transform(run_X_val)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            random_state=self.seed
            )
        
        model.fit(run_X_train, run_y_train)

        val_preds = model.predict(run_X_val)
        val_score = self.metric(run_y_val, val_preds)
        loss = r2_acc_to_loss(val_score)
        logger.info(f"Validation score: {val_score:.4f} Loss: {loss: .4f}")
        return loss
    
    def regression_pipeline_XGB_random_forest(
        self,
        val_split: float,
        iqr_scale: float,
        learning_rate: float,
        max_depth: int,
        sampling_method: str,
        reg_lambda: float,
        reg_alpha: float
        ) -> float:

        run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
            self.dataset.X_train,
            self.dataset.y_train,
            random_state=self.seed,
            test_size=val_split,
        )

        run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, iqr_scale)

        scaler = StandardScaler()
        scaler.fit(run_X_train)

        run_X_train = scaler.transform(run_X_train)
        run_X_val = scaler.transform(run_X_val)

        model = XGBRFRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            sampling_method=sampling_method,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            random_state=self.seed
            )
        
        model.fit(run_X_train, run_y_train)

        val_preds = model.predict(run_X_val)
        val_score = self.metric(run_y_val, val_preds)
        loss = r2_acc_to_loss(val_score)
        logger.info(f"Validation score: {val_score:.4f} Loss: {loss: .4f}")
        return loss
    
    def regression_pipeline_SK_gradient_boost(
        self,
        val_split: float,
        iqr_scale: float,
        loss: str,
        learning_rate: float,
        n_estimators: int,
        subsample: float,
        criterion: str,
        max_depth : int
        ) -> float:

        run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
            self.dataset.X_train,
            self.dataset.y_train,
            random_state=self.seed,
            test_size=val_split,
        )

        run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, iqr_scale)

        scaler = StandardScaler()
        scaler.fit(run_X_train)

        run_X_train = scaler.transform(run_X_train)
        run_X_val = scaler.transform(run_X_val)

        model = GradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            max_depth=max_depth,
            random_state=self.seed
            )
        
        model.fit(run_X_train, run_y_train)

        val_preds = model.predict(run_X_val)
        val_score = self.metric(run_y_val, val_preds)
        loss = r2_acc_to_loss(val_score)
        logger.info(f"Validation score: {val_score:.4f} Loss: {loss: .4f}")
        return loss
    
    def regression_pipeline_SK_ada_boost(
        self,
        val_split: float,
        iqr_scale: float,
        n_estimators: int,
        learning_rate: float,
        loss: str
        ) -> float:

        run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
            self.dataset.X_train,
            self.dataset.y_train,
            random_state=self.seed,
            test_size=val_split,
        )

        run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, iqr_scale)

        scaler = StandardScaler()
        scaler.fit(run_X_train)

        run_X_train = scaler.transform(run_X_train)
        run_X_val = scaler.transform(run_X_val)

        model = AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=self.seed
            )
        
        model.fit(run_X_train, run_y_train)

        val_preds = model.predict(run_X_val)
        val_score = self.metric(run_y_val, val_preds)
        loss = r2_acc_to_loss(val_score)
        logger.info(f"Validation score: {val_score:.4f} Loss: {loss: .4f}")
        return loss
    
    def regression_pipeline_SK_MLP(
        self,
        val_split: float,
        iqr_scale: float,
        hidden_layer_sizes: int,
        hidden_layers: int,
        activation: str,
        solver: str,
        alpha: float,
        batch_size: int,
        learning_rate: str,
        learning_rate_init: float
        ) -> float:

        run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
            self.dataset.X_train,
            self.dataset.y_train,
            random_state=self.seed,
            test_size=val_split,
        )

        run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, iqr_scale)

        run_X_train = run_X_train.to_numpy(dtype=np.float32)
        run_y_train = run_y_train.to_numpy(dtype=np.float32)
        run_X_val = run_X_val.to_numpy(dtype=np.float32)
        run_y_val = run_y_val.to_numpy(dtype=np.float32)

        scaler = StandardScaler()
        scaler.fit(run_X_train)

        run_X_train = scaler.transform(run_X_train)
        run_X_val = scaler.transform(run_X_val)

        hidden_size_list = [hidden_layer_sizes for i in range(hidden_layers)]

        model = MLPRegressor(
            hidden_layer_sizes=hidden_size_list,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            random_state=self.seed
            )
        
        model.fit(run_X_train, run_y_train)

        val_preds = model.predict(run_X_val)
        val_score = self.metric(run_y_val, val_preds)
        loss = r2_acc_to_loss(val_score)
        logger.info(f"Validation score: {val_score:.4f} Loss: {loss: .4f}")
        return loss
    
    def regression_pipeline_SK_gaussian_process(
        self,
        val_split: float,
        iqr_scale: float,
        alpha: float
        ) -> float:

        run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
            self.dataset.X_train,
            self.dataset.y_train,
            random_state=self.seed,
            test_size=val_split,
        )

        run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, iqr_scale)

        scaler = StandardScaler()
        scaler.fit(run_X_train)

        run_X_train = scaler.transform(run_X_train)
        run_X_val = scaler.transform(run_X_val)

        model = GaussianProcessRegressor(
            alpha=alpha,
            random_state=self.seed
        )

        model.fit(run_X_train, run_y_train)

        val_preds = model.predict(X=run_X_val.astype(float))
        val_score = self.metric(run_y_val, val_preds)
        loss = r2_acc_to_loss(val_score)
        logger.info(f"Validation score: {val_score:.4f} Loss: {loss: .4f}")
        return loss
    
    def regression_pipeline_SK_bayesian_ridge(
        self,
        val_split: float,
        iqr_scale: float,
        alpha_1: float,
        alpha_2: float,
        lambda_1: float,
        lambda_2: float
        ) -> float:

        run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
            self.dataset.X_train,
            self.dataset.y_train,
            random_state=self.seed,
            test_size=val_split,
        )

        run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, iqr_scale)

        scaler = StandardScaler()
        scaler.fit(run_X_train)

        run_X_train = scaler.transform(run_X_train)
        run_X_val = scaler.transform(run_X_val)

        model = BayesianRidge(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2
        )

        model.fit(run_X_train, run_y_train)

        val_preds = model.predict(X=run_X_val.astype(float))
        val_score = self.metric(run_y_val, val_preds)
        loss = r2_acc_to_loss(val_score)
        logger.info(f"Validation score: {val_score:.4f} Loss: {loss: .4f}")
        return loss
    
    def regression_pipeline_SK_elastic_net(
        self,
        val_split: float,
        iqr_scale: float,
        alpha: float,
        l1_ratio: float
        ) -> float:

        run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
            self.dataset.X_train,
            self.dataset.y_train,
            random_state=self.seed,
            test_size=val_split,
        )

        run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, iqr_scale)

        scaler = StandardScaler()
        scaler.fit(run_X_train)

        run_X_train = scaler.transform(run_X_train)
        run_X_val = scaler.transform(run_X_val)

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=self.seed
        )

        model.fit(run_X_train, run_y_train)

        val_preds = model.predict(X=run_X_val.astype(float))
        val_score = self.metric(run_y_val, val_preds)
        loss = r2_acc_to_loss(val_score)
        logger.info(f"Validation score: {val_score:.4f} Loss: {loss: .4f}")
        return loss
    
    def regression_pipeline_XGB_dart_boost(
        self,
        val_split: float,
        iqr_scale: float,
        learning_rate: float,
        max_depth: float,
        sampling_method: str,
        rate_drop: float,
        skip_drop: float
        ) -> float:

        run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
            self.dataset.X_train,
            self.dataset.y_train,
            random_state=self.seed,
            test_size=val_split,
        )

        run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, iqr_scale)

        scaler = StandardScaler()
        scaler.fit(run_X_train)

        run_X_train = scaler.transform(run_X_train)
        run_X_val = scaler.transform(run_X_val)

        model = XGBRegressor(
            booster="dart",
            learning_rate=learning_rate,
            max_depth=max_depth,
            sampling_method=sampling_method,
            rate_drop=rate_drop,
            skip_drop=skip_drop,
            random_state=self.seed
        )

        model.fit(run_X_train, run_y_train)

        val_preds = model.predict(X=run_X_val.astype(float))
        val_score = self.metric(run_y_val, val_preds)
        loss = r2_acc_to_loss(val_score)
        logger.info(f"Validation score: {val_score:.4f} Loss: {loss: .4f}")
        return loss

    def run_neps_pipeline(
        self,
        task_id,
        total_evaluations,
        root_directory="results"
        ) -> None:

        curr_pipeline_space = ALGORITHMS[self.curr_algo]

        logger.info(f"Running: {self.curr_algo}")
        match self.curr_algo:
            case "SK_random_forest":
                neps.run(
                    run_pipeline=self.regression_pipeline_SK_random_forest,
                    root_directory=root_directory, 
                    pipeline_space="configs/" + curr_pipeline_space,
                    max_evaluations_total=total_evaluations,
                    searcher="bayesian_optimization",
                    initial_design_size=15,
                    acquisition="EI",
                    overwrite_working_directory=True,
                    task_id=str(task_id) + "_" + self.curr_algo,
                    post_run_summary=True,
                )
            case "XGB_random_forest":
                neps.run(
                    run_pipeline=self.regression_pipeline_XGB_random_forest,
                    root_directory=root_directory,
                    pipeline_space="configs/" + curr_pipeline_space,
                    max_evaluations_total=total_evaluations,
                    searcher="bayesian_optimization",
                    initial_design_size=15,
                    acquisition="EI",
                    overwrite_working_directory=True,
                    task_id=str(task_id) + "_" + self.curr_algo,
                    post_run_summary=True,
                )
            case "SK_gradient_boost":
                neps.run(
                    run_pipeline=self.regression_pipeline_SK_gradient_boost,
                    root_directory=root_directory,
                    pipeline_space="configs/" + curr_pipeline_space,
                    max_evaluations_total=total_evaluations,
                    searcher="bayesian_optimization",
                    initial_design_size=15,
                    acquisition="EI",
                    overwrite_working_directory=True,
                    task_id=str(task_id) + "_" + self.curr_algo,
                    post_run_summary=True,
                )
            case "SK_ada_boost":
                neps.run(
                    run_pipeline=self.regression_pipeline_SK_ada_boost,
                    root_directory=root_directory,
                    pipeline_space="configs/" + curr_pipeline_space,
                    max_evaluations_total=total_evaluations,
                    searcher="bayesian_optimization",
                    initial_design_size=15,
                    acquisition="EI",
                    overwrite_working_directory=True,
                    task_id=str(task_id) + "_" + self.curr_algo,
                    post_run_summary=True,
                )
            case "SK_MLP":
                neps.run(
                    run_pipeline=self.regression_pipeline_SK_MLP,
                    root_directory=root_directory,
                    pipeline_space="configs/" + curr_pipeline_space,
                    max_evaluations_total=total_evaluations,
                    searcher="bayesian_optimization",
                    initial_design_size=15,
                    acquisition="EI",
                    overwrite_working_directory=True,
                    task_id=str(task_id) + "_" + self.curr_algo,
                    post_run_summary=True,
                )
            case "SK_gaussian_process":
                neps.run(
                    run_pipeline=self.regression_pipeline_SK_gaussian_process,
                    root_directory=root_directory, 
                    pipeline_space="configs/" + curr_pipeline_space,
                    max_evaluations_total=total_evaluations,
                    searcher="bayesian_optimization",
                    initial_design_size=15,
                    acquisition="EI",
                    overwrite_working_directory=True,
                    task_id=str(task_id) + "_" + self.curr_algo,
                    post_run_summary=True,
                )
            case "SK_bayesian_ridge":
                neps.run(
                    run_pipeline=self.regression_pipeline_SK_bayesian_ridge,
                    root_directory=root_directory,
                    pipeline_space="configs/" + curr_pipeline_space,
                    max_evaluations_total=total_evaluations,
                    searcher="bayesian_optimization",
                    initial_design_size=15,
                    acquisition="EI",
                    overwrite_working_directory=True,
                    task_id=str(task_id) + "_" + self.curr_algo,
                    post_run_summary=True,
                )
            case "SK_elastic_net":
                neps.run(
                    run_pipeline=self.regression_pipeline_SK_elastic_net,
                    root_directory=root_directory,
                    pipeline_space="configs/" + curr_pipeline_space,
                    max_evaluations_total=total_evaluations,
                    searcher="bayesian_optimization",
                    initial_design_size=15,
                    acquisition="EI",
                    overwrite_working_directory=True,
                    task_id=str(task_id) + "_" + self.curr_algo,
                    post_run_summary=True,
                )
            case "XGB_dart_boost":
                neps.run(
                    run_pipeline=self.regression_pipeline_XGB_dart_boost,
                    root_directory=root_directory,
                    pipeline_space="configs/" + curr_pipeline_space,
                    max_evaluations_total=total_evaluations,
                    searcher="bayesian_optimization",
                    initial_design_size=15,
                    acquisition="EI",
                    overwrite_working_directory=True,
                    task_id=str(task_id) + "_" + self.curr_algo,
                    post_run_summary=True,
                )

        """ neps.run(
            run_pipeline=self.regression_pipeline,
            root_directory=root_directory,
            pipeline_space="configs/SK_random_forest_neps_pipeline_space.yaml",
            overwrite_working_directory=True,
            post_run_summary=True,
            development_stage_id=None,
            task_id=task_id,
            max_evaluations_total=max_evals,
            total_evaluations=max_evals,
            continue_until_max_evaluation_completed=True,
            max_cost_total=None,
            ignore_errors=False,
            loss_value_on_error=None,
            cost_value_on_error=None,
            pre_load_hooks=None,
            searcher="bayesian_optimization",
            initial_design_size=15,
            acquisition="EI"
        ) """

    def run(
        self
    ) -> None:
        
        self.divisions = ceil(log2(len(self.selected_algos))) + 1
        self.curr_total_epoch = ceil(self.total_evaluations / self.divisions)
        self.curr_per_model_epoch = ceil(self.curr_total_epoch / len(self.selected_algos))
        
        for i in range(self.divisions - 1):
            for algo in self.selected_algos:
                self.curr_algo = algo
                self.run_neps_pipeline(task_id=self.task_id, total_evaluations=self.curr_per_model_epoch, root_directory=RESULTDIR)
            
            algo_losses = {}
            # load losses to algo losses
            for algo in self.selected_algos:
                CSVDIR = FILE.parent.parent.parent / "results" / ("task_" + self.task_id + "_" + algo) / "summary_csv"
                csv = pd.read_csv(CSVDIR / "config_data.csv")

                col_names = csv.columns.values
                for col_name in col_names:
                    if("loss" in col_name):
                        loss = csv.iloc[0][col_name]
                        algo_losses.update({algo: loss})
            
            algos_sorted_by_losses = list(dict(sorted(algo_losses.items(), key=lambda key_val: key_val[1])).keys())
            for i in range(int(len(self.selected_algos) / 2)):
                self.selected_algos.remove(algos_sorted_by_losses[-(i+1)])
            
            logger.info(f"Selected algorithm(s) to pass the halving: {self.selected_algos}")
            self.curr_per_model_epoch = self.curr_per_model_epoch * 2
        
        # so many parents
        TXTDIR = FILE.parent.parent.parent
        best_algo_file = open(TXTDIR / "selected_algorithm.txt", "w")
        best_algo_file.write(self.selected_algos[0])
        best_algo_file.close()
        self.curr_algo = self.selected_algos[0]
        self.run_neps_pipeline(task_id=self.task_id, total_evaluations=self.curr_per_model_epoch, root_directory=RESULTDIR)

    def load_and_test_best_model(
        self
    ) -> None:
        
        TXTDIR = FILE.parent.parent.parent
        best_algo_file = open(TXTDIR / "selected_algorithm.txt", "r")
        algo = best_algo_file.read()
        
        CSVDIR = FILE.parent.parent.parent / "results" / ("task_" + self.task_id + "_" + algo) / "summary_csv"

        csv = pd.read_csv(CSVDIR / "config_data.csv")

        param_dict = {}
        col_names = csv.columns.values
        for col_name in col_names:
            if("config" in col_name and "config_id" not in col_name):
                param_dict.update({col_name.replace("config.", ""): csv.iloc[0][col_name]})
        
        match algo:
            case "SK_random_forest":
                run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
                    self.dataset.X_train,
                    self.dataset.y_train,
                    random_state=self.seed,
                    test_size=param_dict["val_split"],
                )

                run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, param_dict["iqr_scale"])

                scaler = StandardScaler()
                scaler.fit(run_X_train)

                run_X_train = scaler.transform(run_X_train)
                run_X_val = scaler.transform(run_X_val)
                run_X_test = scaler.transform(self.dataset.X_test)
                run_y_test = self.dataset.y_test

                model = RandomForestRegressor(
                    n_estimators=int(param_dict["n_estimators"]),
                    criterion=param_dict["criterion"],
                    max_depth=int(param_dict["max_depth"]),
                    random_state=self.seed
                    )
                
                model.fit(run_X_train, run_y_train)

            case "XGB_random_forest":
                run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
                    self.dataset.X_train,
                    self.dataset.y_train,
                    random_state=self.seed,
                    test_size=param_dict["val_split"],
                )

                run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, param_dict["iqr_scale"])

                scaler = StandardScaler()
                scaler.fit(run_X_train)

                run_X_train = scaler.transform(run_X_train)
                run_X_val = scaler.transform(run_X_val)
                run_X_test = scaler.transform(self.dataset.X_test)
                run_y_test = self.dataset.y_test

                model = XGBRFRegressor(
                    learning_rate=param_dict["learning_rate"],
                    max_depth=int(param_dict["max_depth"]),
                    sampling_method=param_dict["sampling_method"],
                    reg_lambda=param_dict["reg_lambda"],
                    reg_alpha=param_dict["reg_alpha"],
                    random_state=self.seed
                    )

                model.fit(run_X_train, run_y_train)

            case "SK_gradient_boost":
                run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
                    self.dataset.X_train,
                    self.dataset.y_train,
                    random_state=self.seed,
                    test_size=param_dict["val_split"],
                )

                run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, param_dict["iqr_scale"])

                scaler = StandardScaler()
                scaler.fit(run_X_train)

                run_X_train = scaler.transform(run_X_train)
                run_X_val = scaler.transform(run_X_val)
                run_X_test = scaler.transform(self.dataset.X_test)
                run_y_test = self.dataset.y_test

                model = GradientBoostingRegressor(
                    loss=param_dict["loss"],
                    learning_rate=param_dict["learning_rate"],
                    n_estimators=int(param_dict["n_estimators"]),
                    subsample=param_dict["subsample"],
                    criterion=param_dict["criterion"],
                    max_depth=int(param_dict["max_depth"]),
                    random_state=self.seed
                    )

                model.fit(run_X_train, run_y_train)

            case "SK_ada_boost":
                run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
                    self.dataset.X_train,
                    self.dataset.y_train,
                    random_state=self.seed,
                    test_size=param_dict["val_split"],
                )

                run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, param_dict["iqr_scale"])

                scaler = StandardScaler()
                scaler.fit(run_X_train)

                run_X_train = scaler.transform(run_X_train)
                run_X_val = scaler.transform(run_X_val)
                run_X_test = scaler.transform(self.dataset.X_test)
                run_y_test = self.dataset.y_test

                model = AdaBoostRegressor(
                    n_estimators=int(param_dict["n_estimators"]),
                    learning_rate=param_dict["learning_rate"],
                    loss=param_dict["loss"],
                    random_state=self.seed
                    )

                model.fit(run_X_train, run_y_train)

            case "SK_MLP":
                run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
                    self.dataset.X_train,
                    self.dataset.y_train,
                    random_state=self.seed,
                    test_size=param_dict["val_split"],
                )

                run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, param_dict["iqr_scale"])

                run_X_train = run_X_train.to_numpy(dtype=np.float32)
                run_y_train = run_y_train.to_numpy(dtype=np.float32)
                run_X_val = run_X_val.to_numpy(dtype=np.float32)
                run_y_val = run_y_val.to_numpy(dtype=np.float32)

                scaler = StandardScaler()
                scaler.fit(run_X_train)

                run_X_train = scaler.transform(run_X_train)
                run_X_val = scaler.transform(run_X_val)
                run_X_test = scaler.transform(self.dataset.X_test)
                run_y_test = self.dataset.y_test

                hidden_size_list = [int(param_dict["hidden_layer_sizes"]) for i in range(int(param_dict["hidden_layers"]))]

                model = MLPRegressor(
                    hidden_layer_sizes=hidden_size_list,
                    activation=param_dict["activation"],
                    solver=param_dict["solver"],
                    alpha=param_dict["alpha"],
                    batch_size=int(param_dict["batch_size"]),
                    learning_rate=param_dict["learning_rate"],
                    learning_rate_init=param_dict["learning_rate_init"],
                    random_state=self.seed
                    )

                model.fit(run_X_train, run_y_train)

            case "SK_gaussian_process":
                ## dropped
                pass
            case "SK_bayesian_ridge":
                run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
                    self.dataset.X_train,
                    self.dataset.y_train,
                    random_state=self.seed,
                    test_size=param_dict["val_split"],
                )

                run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, param_dict["iqr_scale"])

                scaler = StandardScaler()
                scaler.fit(run_X_train)

                run_X_train = scaler.transform(run_X_train)
                run_X_val = scaler.transform(run_X_val)
                run_X_test = scaler.transform(self.dataset.X_test)
                run_y_test = self.dataset.y_test

                model = BayesianRidge(
                    alpha_1=param_dict["alpha_1"],
                    alpha_2=param_dict["alpha_2"],
                    lambda_1=param_dict["lambda_1"],
                    lambda_2=param_dict["lambda_2"]
                )

                model.fit(run_X_train, run_y_train)

            case "SK_elastic_net":
                run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
                    self.dataset.X_train,
                    self.dataset.y_train,
                    random_state=self.seed,
                    test_size=param_dict["val_split"],
                )

                run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, param_dict["iqr_scale"])

                scaler = StandardScaler()
                scaler.fit(run_X_train)

                run_X_train = scaler.transform(run_X_train)
                run_X_val = scaler.transform(run_X_val)
                run_X_test = scaler.transform(self.dataset.X_test)
                run_y_test = self.dataset.y_test

                model = ElasticNet(
                    alpha=param_dict["alpha"],
                    l1_ratio=param_dict["l1_ratio"],
                    random_state=self.seed
                )

                model.fit(run_X_train, run_y_train)

            case "XGB_dart_boost":
                run_X_train, run_X_val, run_y_train, run_y_val = train_test_split(
                    self.dataset.X_train,
                    self.dataset.y_train,
                    random_state=self.seed,
                    test_size=param_dict["val_split"],
                )

                run_X_train, run_y_train = drop_outliers(run_X_train, run_y_train, param_dict["iqr_scale"])

                scaler = StandardScaler()
                scaler.fit(run_X_train)

                run_X_train = scaler.transform(run_X_train)
                run_X_val = scaler.transform(run_X_val)
                run_X_test = scaler.transform(self.dataset.X_test)
                run_y_test = self.dataset.y_test

                model = XGBRegressor(
                    booster="dart",
                    learning_rate=param_dict["learning_rate"],
                    max_depth=int(param_dict["max_depth"]),
                    sampling_method=param_dict["sampling_method"],
                    rate_drop=param_dict["rate_drop"],
                    skip_drop=param_dict["skip_drop"],
                    random_state=self.seed
                )

                model.fit(run_X_train, run_y_train)

        self.best_model = model

        val_preds = model.predict(run_X_val)
        val_score = self.metric(run_y_val, val_preds)

        loss = r2_acc_to_loss(val_score)
        logger.info(f"Final Validation score: {val_score:.4f} Loss: {loss: .4f}")

        test_preds = model.predict(run_X_test)

        logger.info("Writing predictions to disk")
        with self.output_path[0].open("wb") as f:
            np.save(f, test_preds)
        if(run_y_test is not None):
            test_score = self.metric(run_y_test, test_preds) 
            loss = r2_acc_to_loss(test_score)
            logger.info(f"Final Test score: {test_score:.4f} Loss: {loss: .4f}")
        else:
            logger.info(f"Final Test score failed: Test data Y doesnt exist.")

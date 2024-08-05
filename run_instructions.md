# AutoML Exam Dropouts - SS24 (Tabular Data) Submission

## Usage

To find the optimal model, the "run" command must be used first.

**Virtual Enviroment**

All the development was done in python 3.10 (3.10.12) and a virtual enviroment was used. After creating a virtual enviroment for python 3.10, the following command to install all dependicies can be run.

```bash
pip install -r requirements.txt
```

**Downloading Datasets**

You can download the required OpenML practice data using:

```bash
python download-openml.py --task <task_id>
```

**Running the AutoML**

To run the AutoML pipeline, the run command must be used to generate a fully trained selected model inside the "models/task_id" folder.
It also follows the same parsing on the example code so total epoch numbers can also be given as arguments. (for details check run.py)

Run command for the brazilian houses dataset:

```bash
python run.py --task brazilian_houses --seed 42
```

Run command for the exam dataset:

```bash
python run.py --task test exam --seed 42
```

## Running the Test and Getting Predictions

After the run command, the test command must be used to load the model from the models/task_id folder. The model is selected by reading from the selected_algorithm.txt file which is written after the run command. It also follows similar parsing to example code so the output path can be also given as arguments. (for details check test.py)

```bash
python test.py --task exam_dataset --fold 1 --seed 42
```

Depending on whether the dataset has y_test values, it will print a validation and test score or it will print validation score and a test prediction to the given output directory.

## Short Details on How It Works

the automl.py file is where all the functionality is written. The "ALGORITHMS" dictionary at the top can be used to modify the algorithm pool from which to perform the successive halving. (example the gaussian one is commented thus wont be in the algorithm pool) Only one algorithm can be left without comments to get a single hyperparameter optimization run. After the run, it creates a selected_algorithm.txt file where the name of the algorithm that survived the halving is written.

With NePS functionality, the results and the incumbent trajectory can be found in the results folder. The most recent run also keeps the results so that the hyperparameters can be read if required. (also in the summary_csv sub folder config_data.csv file)
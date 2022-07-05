# Consistency-preserving Visual Question Answering in Medical Imaging
This is the official repository of the paper "Consistency-preserving Visual Question Answering in Medical Imaging," published in the proceedings of the MICCAI2022.

Our method consists of a loss function and corresponding training method to improve the consistency. Evaluated on a medical dataset, we achieve improvements both in consistency and accuracy. For more details, please refer to our [paper](https://arxiv.org/abs/2206.13296).

<p align="center">
<img src="./assets/method.png" alt="method" width="500"/>
</p>

## Data
You can download our DME dataset from [here](https://zenodo.org/record/6784358). You can place the zip file in any location and then unzip it. We'll refer to the path to the unzipped folder as `<path_data>`. A more detailed description of the dataset can be found [here](DATA.md).


## Installing Requirements
After cloning the repo, create a new environment with Python 3.9, activate it, and then install the required packages by running:

    pip install -r requirements.txt

## Configuration file
In the folder `config/idrid_regions/single/` you can find different configuration files that correspond to different scenarios, as shown in Table 1 of our paper. More specifically, you can find the following configuration files:

<p align="center">

| Config file      | Consistency method |
| ----------- | ----------- |
| default_baseline.yaml      | None      |
| default_squint.yaml   | [SQuINT](https://openaccess.thecvf.com/content_CVPR_2020/papers/Selvaraju_SQuINTing_at_VQA_Models_Introspecting_VQA_Models_With_Sub-Questions_CVPR_2020_paper.pdf) by Selvaraju et al.     |
| default_consistency.yaml   | Ours        |

</p>

In order to use a configuration file to train, you must first change the fields `path_img`, `path_qa` and `path_masks` to match the path to the downloaded data `<path_data>`. Please note that with these configuration files you should obtain results that are similar to the ones reported in our paper. However, since we reported the average for 10 runs of each model, your results may deviate. 

If you have a [comet ml](https://www.comet.ml) account, you can set the parameter `comet_ml` to True in the configuration file to monitor the training process in real time. This requires the proper configuration to access your account from Python (see [this](https://www.comet.ml/docs/python-sdk/advanced/) for more info).

## Training
To train a model just run the following command:

    train.py --path_config <path_config>

Example:
    
    train.py --path_config config/idrid_regions/single/default_baseline.yaml

After training, the `logs` folder, as defined in the YAML file, will contain the results of the training. This includes the model weights for the best and last epoch, as well as the answers produced by the model for each epoch. Additionally, a JSON file named `logbook` will be generated, which contains the information from the config file and the values of the metrics (loss and performance) for each epoch.

## Inference for test set
In order to do inference on the test set, use the following command:

    inference.py --path_config <path_config>

The inference results are stored in the `logs` folder, as defined in the config file, in the sub-folder answers. In total 6 answer files are generated, as follows:

<p align="center">

| File name      | Meaning |
| ----------- | ----------- |
| answers_epoch_0.pt     | best model on test set      |
| answers_epoch_2000.pt   | best model on val set     |
| answers_epoch_1000.pt   | best model on train set        |
| answers_epoch_1000.pt   | best model on train set        |
| answers_epoch_2001.pt   | last model on val set        |
| answers_epoch_1001.pt   | last model on train set        |

</p>

Each of these files contains a matrix with two columns, the first one representing the question ID, and the second one corresponding to the answer provided by the model. The answer is an integer. To convert from integer to the textual answer, a dictionary is given in `<path_data>/processed/map_index_answer.pickle`

## Inference for a single sample
The following command allows you to do inference on a single sample using a previously trained model (as specified by the config file in `<path_config>`):

    inference_single.py --path_config <path_config> --path_image <path_image> --path_mask <path_mask> --question <question>



## Plotting metrics and learning curves
To plot learning curves and accuracy, use the following command after having trained and done inference:

    plotter.py --path_config <path_config>

The resulting plots are stored in the `logs` folder. 


## Computing consistency

After running the inference script, you can compute the consistency using:

    compute_consistency.py --path_config <path_config>

By default, this only computes the consistency C1 (see paper). To compute the consistency C2 as well, set the parameter `q3_too` to True when calling the function `compute_consistency`in the script `compute_consistency.py`.

<br />
<br />



This work was carried out at the [AIMI Lab](https://www.artorg.unibe.ch/research/aimi/index_eng.html) of the [ARTORG Center for Biomedical Engineering Research](https://www.artorg.unibe.ch) of the [University of Bern](https://www.unibe.ch/index_eng.html). Please cite this work as:

> citation pending

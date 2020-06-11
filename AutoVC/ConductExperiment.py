from conversion import *
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, help = "Which model to use")
    parser.add_argument('--train_size', default = None, type = str, help = "The number of minutes trained. Has to be the name of the experiment file")
    parser.add_argument('--test_data', type = str, help = "File location of test data folder")
    parser.add_argument('--name_list', type = str, help = "File location of name_list")
    parser.add_argument('--experiment_folder', type = str, help = "File location of experiment folder")


    config = parser.parse_args()

    Experiment(Model_path = config.model, train_length = config.train_size, test_data = config.test_data, name_list = config.name_list, experiment = config.experiment_folder)




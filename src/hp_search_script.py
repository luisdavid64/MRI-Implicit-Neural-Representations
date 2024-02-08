import os
import argparse
import shutil
import yaml
import torch.backends.cudnn as cudnn
from datetime import datetime
from models.utils import get_config, prepare_sub_folder, get_device
from parameter_search.find_best_config import grid_search, random_search
from src.utils import set_default_configs


def run(config, hp_config):
    """
    The script to run the hyperparameter search
    :param config: Main config for training a model
    :param hp_config: Configs that contains hyperparameters to be searched
    :return:
    """
    device = get_device(config["model"])
    print("Running on: ", device)
    cudnn.benchmark = True

    # Setup output folder
    output_folder = os.path.splitext(os.path.basename(opts.config))[0]
    model_name = os.path.join(output_folder, config['data'] + '/img_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}_hp_{}_search_' \
                              .format(config['model'], config['net']['network_input_size'],
                                      config['net']['network_width'], config['net']['network_depth'], config['loss'],
                                      config['lr'], config['encoder']['embedding'],
                                      hp_config["method"]))
    if not (config['encoder']['embedding'] == 'none'):
        model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
    print(model_name)

    output_directory = os.path.join(opts.output_path + "/outputs",
                                    model_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder


    ## adding useful additional_data to configs, which are popped before running the best configs.
    config["image_directory"] = image_directory
    config["output_directory"] = output_directory

    search_method = hp_config.pop("method")
    if search_method == "grid":
        print("** Running Grid Search **")
        best_configs = grid_search(model_configs=config, model_class=config["model"],
                                   epochs=hp_config.pop("max_epoch"), grid_search_spaces=hp_config.pop("search_space"),
                                   device=device)

    else:
        print("** Running Random Search **")
        best_configs = random_search(model_configs=config,
                                     model_class=config["model"],
                                     num_search=hp_config.pop("num_search"),
                                     epochs=hp_config.pop("max_epoch"),
                                     random_search_spaces=hp_config.pop("search_space"), device=device)

    with open(os.path.join(output_directory, "best_psnr_config.yaml"), "w") as yaml_file:
        yaml.dump(best_configs["PSNR"]["config"], yaml_file, default_flow_style=False)

    with open(os.path.join(output_directory, "best_ssim_config.yaml"), "w") as yaml_file:
        yaml.dump(best_configs["SSIM"]["config"], yaml_file, default_flow_style=False)

    with open(os.path.join(output_directory, "configs_and_results.txt"), "w") as tf:
        for item in best_configs["results"]:
            tf.write("{} -> {}\n".format(item[0], item[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config/config_image.yaml', help='Path to the config file.')
    parser.add_argument('--hp_config', type=str, default='src/hp_config/config_image.json',
                        help='Path to the HP config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    # Load experiment setting
    opts = parser.parse_args()
    config = get_config(opts.config)
    config = set_default_configs(config)
    hp_config = get_config(opts.hp_config)

    # Running the hyperparameter search algorithm
    run(config=config, hp_config=hp_config)

import random
from math import log10
from itertools import product
from src.parameter_search.hp_model_training import hp_training_function

"""
The file that implements Random and Grid hyperparameter search methods 
"""

# hyperparameter type
ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


def update_model_config(model_configs, hp_configs):
    """
    Updating the model's config from the search configs.
    """
    for k, v in hp_configs.items():
        k_split = k.split(".")
        if '.' in k:
            model_configs[k_split[0]][k_split[1]] = hp_configs[k]
        else:
            model_configs[k] = hp_configs[k]

    return model_configs


def findBestConfig(model_configs, hp_configs, epochs, device):
    """
    Get a list of hyperparameter configs for random search or grid search,
    trains a model on all configs and returns the one performing best
    on validation set
    """

    best_psnr = -999999
    best_config_psnr = None
    best_ssim = -1
    best_config_ssim = None
    results = []

    image_directory = model_configs.pop("image_directory")
    output_directory = model_configs.pop("output_directory")

    for i in range(len(hp_configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format(
            (i + 1), len(hp_configs)), hp_configs[i])

        # best_model_stats = train_function(model, train_loader, val_loader, **model_configs[i])
        model_configs = update_model_config(model_configs, hp_configs[i])
        model_configs["config_index"] = i+1

        import os, yaml
        with open(os.path.join(output_directory, "hp_search_config_{}.yaml".format(i+1)), "w") as hp_con:
            yaml.dump(hp_configs[i], hp_con, default_flow_style=False)

        from src.models.utils import get_data_loader
        dataset, data_loader, val_loader = get_data_loader(
            data=model_configs['data'],
            data_root=model_configs['data_root'],
            set=model_configs['set'],
            batch_size=model_configs['batch_size'],
            transform=model_configs['transform'],
            num_workers=0,
            sample=model_configs["sample"],
            slice=model_configs["slice"],
            shuffle=True,
            full_norm=model_configs["full_norm"],
            normalization=model_configs["normalization"],
            undersampling=model_configs["undersampling"],
            use_dists="no",
            per_coil=model_configs["per_coil"]
        )

        best_model_stats = hp_training_function(config=model_configs, max_epoch=epochs, image_directory=image_directory,
                                                device=device,
                                                dataset=dataset, data_loader=data_loader, val_loader=val_loader)

        # Best PSNR
        if best_model_stats["best_psnr"] > best_psnr:
            best_psnr, best_config_psnr = best_model_stats["best_psnr"], model_configs
            # best_model_psnr

        # Best SSIM
        if best_model_stats["best_ssim"] > best_ssim:
            best_ssim, best_config_ssim = best_model_stats["best_ssim"], model_configs
            # best_model_ssim

    print("\nSearch done. Best Psnr = {}".format(best_psnr))
    print("Best Config PSNR:", best_config_psnr)
    print("\nSearch done. Best SSIM = {}".format(best_ssim))
    print("Best Config SSIM:", best_config_ssim)

    return {"PSNR": {"config": best_config_psnr},
            "SSIM": {"config": best_config_ssim},
            "results": list(zip(hp_configs, results))}


def grid_search(model_class, model_configs, device, dataset=None, image=None, train_loader=None,
                val_loader=None, epochs=20, grid_search_spaces = {
                    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    "reg": [1e-4, 1e-5, 1e-6]
                }):
    """
    A simple grid search based on nested loops to tune learning rate and
    regularization strengths.
    Keep in mind that you should not use grid search for higher-dimensional
    parameter tuning, as the search space explodes quickly.

    Required arguments:
        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data

    Optional arguments:
        - grid_search_spaces: a dictionary where every key corresponds to a
        to-tune-hyperparameter and every value contains a list of possible
        values. Our function will test all value combinations which can take
        quite a long time. If we don't specify a value here, we will use the
        default values of both our chosen model as well as our solver
        - model: our selected model for this exercise
        - epochs: number of epochs we are training each model
        - patience: if we should stop early in our solver

    Returns:
        - The best performing model
        - A list of all configurations and results
    """
    configs = []

    print("Running Grid Search method on model: ", model_class)
    # converting search space into the format below
    # {
    #     "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    #     "reg": [1e-4, 1e-5, 1e-6]
    # }

    for k, v in grid_search_spaces.items():
        grid_search_spaces[k] = v.get("values")

    """
    # Simple implementation with nested loops
    for lr in grid_search_spaces["learning_rate"]:
        for reg in grid_search_spaces["reg"]:
            configs.append({"learning_rate": lr, "reg": reg})
    """

    # More general implementation using itertools
    for instance in product(*grid_search_spaces.values()):
        configs.append(dict(zip(grid_search_spaces.keys(), instance)))

    return findBestConfig(model_configs=model_configs, hp_configs=configs, epochs=epochs, device=device)


def random_search(model_class, model_configs, device, num_search=20, epochs=20, random_search_spaces = {
                      "learning_rate": ([0.0001, 0.1], 'log'),
                      "hidden_size": ([100, 400], "int"),
                      #"activation": ([Sigmoid(), Relu()], "item"),
                  }):
    """
    Samples N_SEARCH hyper parameter sets within the provided search spaces
    and returns the best model.

    See the grid search documentation above.

    Additional/different optional arguments:
        - random_search_spaces: similar to grid search but values are of the
        form
        (<list of values>, <mode as specified in ALLOWED_RANDOM_SEARCH_PARAMS>)
        - num_search: number of times we sample in each int/float/log list
    """
    configs = []
    print("Running Random Search method on model: ", model_class)
    # converting search space in the format below
    # random_search_spaces = {
    #     "learning_rate": ([0.0001, 0.1], 'log'),
    #     "hidden_size": ([100, 400], "int"),
    #     "loss": ([tanh, L2], "item"),
    # }
    for k, v in random_search_spaces.items():
        random_search_spaces[k] = tuple([v.get("values"), v.get("type")])

    for _ in range(num_search):
        configs.append(random_search_spaces_to_config(random_search_spaces))

    return findBestConfig(model_configs=model_configs, hp_configs=configs, epochs=epochs, device=device)


def random_search_spaces_to_config(random_search_spaces):
    """"
    Takes search spaces for random search as input; samples accordingly
    from these spaces and returns the sampled hyper-params as a config-object,
    which will be used to construct solver & network
    """

    config = {}

    for key, (rng, mode) in random_search_spaces.items():
        if mode not in ALLOWED_RANDOM_SEARCH_PARAMS:
            print("'{}' is not a valid random sampling mode. "
                  "Ignoring hyper-param '{}'".format(mode, key))
        elif mode == "log":
            if rng[0] <= 0 or rng[-1] <= 0:
                print("Invalid value encountered for logarithmic sampling "
                      "of '{}'. Ignoring this hyper param.".format(key))
                continue
            sample = random.uniform(log10(rng[0]), log10(rng[-1]))
            config[key] = 10 ** (sample)
        elif mode == "int":
            config[key] = random.randint(rng[0], rng[-1])
        elif mode == "float":
            config[key] = random.uniform(rng[0], rng[-1])
        elif mode == "item":
            config[key] = random.choice(rng)

    return config

def set_default_configs(config):
    """

    This method sets missing configurations.

    """
    if "per_coil" not in config:
        config["per_coil"] = False
    if "use_tv" not in config:
        config["use_tv"] = False
    if "regularization" not in config:
        config["regularization"] = dict()
        config["regularization"]["type"] = "none" 
    if "undersampling" not in config:
        config["undersampling"] = None

    return config
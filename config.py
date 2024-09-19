

def get_custom_dataset_folder():
    try:
        from config_secret import CUSTOM_DATASET_FOLDER
        return CUSTOM_DATASET_FOLDER
    except ImportError:
        return ''


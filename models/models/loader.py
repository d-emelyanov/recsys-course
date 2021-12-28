import importlib

def load_model(
    path,
    params,
    **kwargs
):
    module_name, model_class_name = (
        '.'.join(path.split('.')[:-1]),
        path.split('.')[-1]
    )
    module = importlib.import_module(f'models.{module_name}')
    model_class = getattr(module, model_class_name)
    model_ = model_class.from_args(args=params, **kwargs)
    return model_

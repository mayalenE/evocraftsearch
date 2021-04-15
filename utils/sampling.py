import numbers

import torch


def sample_value(config={}):
    """Samples scalar values depending on the provided properties."""

    val = None

    if isinstance(config, numbers.Number):  # works also for booleans
        val = config
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)

    elif config is None:
        # val = np.random.rand()
        val = torch.rand(())

    elif isinstance(config, tuple):

        if config[0] == 'continuous' or config[0] == 'continous':
            val = (config[1] - config[2]) * torch.rand(()) + config[2]

        elif config[0] == 'discrete':
            val = torch.randint(config[1], config[2] + 1, ())

        elif config[0] == 'function':
            val = config[1](*config[2:])  # call function and give the other elements in the tuple as paramters

        elif len(config) == 2:
            val = (config[0] - config[1]) * torch.rand(()) + config[1]

        else:
            raise ValueError('Unknown parameter type {!r} for sampling!', config[0])

    elif isinstance(config, list):
        val = config[torch.randint(len(config), ())]
        if isinstance(val, numbers.Number) and not isinstance(val, torch.Tensor):
            val = torch.tensor(val)

    elif isinstance(config, dict):
        if config['type'] == 'discrete':
            val = torch.randint(config['min'], config['max'] + 1, ())

        elif config['type'] == 'continuous':
            val = (config['min'] - config['max']) * torch.rand(()) + config['max']

        elif config['type'] == 'boolean':
            val = torch.rand(()) > 0.5

        elif config['type'] == 'function':
            function_call = config['callname']
            function_kwargs = config
            del function_kwargs['type']
            del function_kwargs['callname']
            val = function_call(**function_kwargs)

        else:
            raise ValueError('Unknown parameter type {!r} for sampling!', config['type'])

    return val

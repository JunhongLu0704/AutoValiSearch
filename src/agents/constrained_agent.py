class ConstrainedProposalAgent:
    """Tiny demo agent that emits legal configurations from a fixed search space."""

    def __init__(self, search_space):
        self.search_space = search_space

    def shared_anchor(self):
        return dict(self.search_space['shared_anchor'])

    def validate(self, config):
        for name, value in config.items():
            if value not in self.search_space['parameters'][name]:
                raise ValueError(f'illegal value for {name}: {value}')
        return True

class Config:
    def __init__(self, raw):
        if raw is None:
            raw = dict()

        self.print_time: bool = raw.get('print_time')
        self.use_subset_robot: bool = raw.get('use_subset_robot')
        self.seed: int = raw.get('seed')
        self.n_samples: int = raw.get('n_samples')
        self.default_starting_robot: int = raw.get('default_starting_robot')

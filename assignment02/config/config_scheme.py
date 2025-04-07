class ImprovementCentroid:
    def __init__(self, raw):
        if raw is None:
            raw = dict()

        self.epsilon: bool = raw.get('epsilon')
        self.skip_after_penalty_count: bool = raw.get('skip_after_penalty_count')


class ImprovementInterchange:
    def __init__(self, raw):
        if raw is None:
            raw = dict()

        self.cover_range: int = raw.get('cover_range')


class ParametersConfig:
    def __init__(self, raw):
        if raw is None:
            raw = dict()

        self.m: int = raw.get('m')
        self.q: int = raw.get('q')
        self.c_b: int = raw.get('c_b')
        self.c_h: int = raw.get('c_h')
        self.c_m: int = raw.get('c_m')
        self.c_c: int = raw.get('c_c')
        self.ld: int = raw.get('ld')
        self.r_min: int = raw.get('r_min')
        self.r_max: int = raw.get('r_max')


class Config:
    def __init__(self, raw):
        if raw is None:
            raw = dict()

        self.print_time: bool = raw.get('print_time')
        self.use_subset_robot: bool = raw.get('use_subset_robot')
        self.seed: int = raw.get('seed')
        self.n_samples: int = raw.get('n_samples')
        self.default_starting_robot: int = raw.get('default_starting_robot')
        self.improvement_centroid: ImprovementCentroid = ImprovementCentroid(raw.get('improvement_centroid'))
        self.improvement_interchange: ImprovementInterchange = ImprovementInterchange(
            raw.get('improvement_interchange'))
        self.parameters: ParametersConfig = ParametersConfig(raw.get("parameters"))

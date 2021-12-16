from .mmn import MMN
ARCHITECTURES = {"MMN": MMN}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)

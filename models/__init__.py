from .trans_vg import TransVG


def build_model(args, argsRWKV):
    return TransVG(args, argsRWKV)

FLO = 2613264


def get_flo(batch: int, seq: int) -> int:
    return FLO * batch * seq * seq

"""D1 dbservice scope; reuse the tested training/checkpoint implementation."""
from ts_benchmark.baselines.MindTSWeb import MindTSWeb


class MindTSD1(MindTSWeb):
    allowed_services = ("GAIA_dbservice1", "GAIA_dbservice2")

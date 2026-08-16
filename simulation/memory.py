"""
Memory module (Section III-B).

Maintains persistent observation data across epochs:
1. Observation history H(t): complete list of search events
   (UAV, cell, outcome, timestamp)
2. Confirmed finds d(t): discovered target records
"""


class MemoryModule:
    """Persistent memory across decision epochs."""

    def __init__(self):
        self.observation_history = []   # list of dicts
        self.confirmed_finds = {}       # target_id -> {cell, time}

    def add_observation(self, uav_id: int, cell_rc: tuple,
                        detection: int, timestamp: float):
        """Record a completed search event."""
        self.observation_history.append({
            "uav_id": uav_id,
            "cell": cell_rc,
            "detection": detection,   # 0 or 1
            "time": timestamp,
        })

    def record_find(self, target_id: int, cell_rc: tuple, timestamp: float):
        """Record a confirmed target discovery."""
        self.confirmed_finds[target_id] = {"cell": cell_rc, "time": timestamp}

    def get_history_text(self) -> str:
        """Full observation history for LLM prompt."""
        if not self.observation_history:
            return "No search observations yet."
        lines = []
        for obs in self.observation_history:
            result = "DETECTION" if obs["detection"] else "no detection"
            lines.append(
                f"  t={obs['time']:.0f}s: UAV {obs['uav_id']} searched "
                f"cell ({obs['cell'][0]},{obs['cell'][1]}) -> {result}"
            )
        return "\n".join(lines)

    def get_finds_text(self) -> str:
        """Summary of confirmed finds."""
        if not self.confirmed_finds:
            return "No targets found yet."
        lines = []
        for tid, info in self.confirmed_finds.items():
            lines.append(
                f"  Target {tid} found at cell ({info['cell'][0]},"
                f"{info['cell'][1]}) at t={info['time']:.0f}s"
            )
        return "\n".join(lines)

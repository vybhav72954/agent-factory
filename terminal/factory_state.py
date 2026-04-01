"""
Factory state management — single source of truth for all UI panes.

All 4 panes read from FactoryState. When the processing pipeline returns
a result, update_from_agent_result() updates everything in one atomic operation.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MachineState:
    """State of one factory machine."""
    id: int
    name: str
    status: str = "ONLINE"          # "ONLINE" | "DEGRADED" | "OFFLINE"
    rul: float = 999.0              # Remaining useful life (cycles)
    base_time: float = 8.0          # Max available hours per shift
    available_time: float = 8.0     # Current available hours


@dataclass
class FactoryState:
    """
    Global factory state shared across all 4 panes.

    5 Machines:
        1: CNC-Alpha       3: Press-Gamma
        2: CNC-Beta        4: Lathe-Delta
                           5: Mill-Epsilon
    """

    machines: dict = field(default_factory=lambda: {
        1: MachineState(1, "CNC-Alpha"),
        2: MachineState(2, "CNC-Beta"),
        3: MachineState(3, "Press-Gamma"),
        4: MachineState(4, "Lathe-Delta"),
        5: MachineState(5, "Mill-Epsilon"),
    })

    sensor_history: list = field(default_factory=lambda: [
        [] for _ in range(18)
    ])
    HISTORY_LENGTH: int = 60

    active_machine_id: int = 1

    capacity_pct: float = 100.0
    machine_req: float = 0.0
    breakeven_risk: bool = False

    comms_log: list = field(default_factory=list)
    MAX_LOG_ENTRIES: int = 100

    def update_from_agent_result(self, result: dict):
        """
        Update state from agent processing result.

        Args:
            result: dict with 'valid', 'machine_statuses', 'capacity_report' keys
        """
        if not result.get("valid", False):
            return

        for ms in result.get("machine_statuses", []):
            mid = ms["id"]
            if mid in self.machines:
                self.machines[mid].status = ms["status"]
                self.machines[mid].rul = ms["rul"]
                self.machines[mid].available_time = ms["available_time"]

        report = result.get("capacity_report", {})
        if report:
            self.capacity_pct = report.get("capacity_pct", self.capacity_pct)
            self.machine_req = report.get("machine_req", self.machine_req)
            self.breakeven_risk = report.get("breakeven_risk", self.breakeven_risk)

    def push_sensor_reading(self, sensor_values: np.ndarray):
        """
        Push one timestep of 18 sensor values into the ring buffer.

        Args:
            sensor_values: numpy array of shape (18,)
        """
        for i, val in enumerate(sensor_values):
            self.sensor_history[i].append(float(val))
            if len(self.sensor_history[i]) > self.HISTORY_LENGTH:
                self.sensor_history[i].pop(0)

    def add_log_entry(self, agent_name: str, message: str):
        """Add timestamped entry to comms log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.comms_log.append({
            "time": timestamp,
            "agent": agent_name,
            "message": message,
        })
        if len(self.comms_log) > self.MAX_LOG_ENTRIES:
            self.comms_log = self.comms_log[-self.MAX_LOG_ENTRIES:]

    def get_sensor_window(self) -> np.ndarray:
        """
        Build a (50, 18) sensor window from recent history.

        If insufficient history, pad with random baseline values [0.3, 0.7].

        Returns:
            numpy array of shape (50, 18) for predict_rul()
        """
        window = np.zeros((50, 18), dtype=np.float32)
        for sensor_idx in range(18):
            history = self.sensor_history[sensor_idx]
            if len(history) >= 50:
                window[:, sensor_idx] = history[-50:]
            elif len(history) > 0:
                padding = [history[0]] * (50 - len(history))
                window[:, sensor_idx] = padding + history
            else:
                window[:, sensor_idx] = np.random.uniform(0.3, 0.7, size=50)
        return window

    def reset_all(self):
        """Reset all machines to ONLINE."""
        for m in self.machines.values():
            m.status = "ONLINE"
            m.rul = 999.0
            m.available_time = m.base_time
        self.capacity_pct = 100.0
        self.machine_req = 0.0
        self.breakeven_risk = False

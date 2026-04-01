"""
ForgeMind Factory Floor Dashboard — Main Application

4-pane Textual app:
    Pane 1: Live sensor feed with RUL
    Pane 2: Capacity dashboard (all 5 machines)
    Pane 3: Agent comms log
    Pane 4: Chaos engine input

Run: python -m terminal.app
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog
from textual.binding import Binding
from textual import work

import re
import numpy as np

from .factory_state import FactoryState
from .layout import SensorFeedWidget, CapacityWidget, format_log_entry
from .dummy_oracle import predict_rul


class FactoryApp(App):
    """Predictive maintenance terminal dashboard."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: 1fr 1fr 3;
    }

    #sensor-pane {
        column-span: 1;
        row-span: 1;
        border: solid green;
        padding: 1;
    }

    #capacity-pane {
        column-span: 1;
        row-span: 1;
        border: solid blue;
        padding: 1;
    }

    #comms-pane {
        column-span: 2;
        row-span: 1;
        border: solid yellow;
        padding: 1;
    }

    #chaos-input {
        column-span: 2;
        row-span: 1;
        dock: bottom;
    }

    .status-online { color: green; }
    .status-degraded { color: yellow; }
    .status-offline { color: red; }

    .rul-healthy { color: green; }
    .rul-warning { color: yellow; }
    .rul-critical { color: red; }
    """

    BINDINGS = [
        Binding("ctrl+r", "reset_factory", "Reset All Machines"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.state = FactoryState()
        self._machine_cycle = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield SensorFeedWidget(id="sensor-pane")
        yield CapacityWidget(id="capacity-pane")
        yield RichLog(id="comms-pane", highlight=True, markup=True)
        yield Input(
            id="chaos-input",
            placeholder="CHAOS ENGINE > Type a fault description and press Enter...",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Initialize display on startup."""
        self._refresh_sensor_pane()
        self._refresh_capacity_pane()
        self._log("System", "Factory simulation online. All machines nominal.")
        self._log("System", "Type a fault description below to inject chaos.")
        self._log("System", "Keyboard: Ctrl+R = Reset | Ctrl+Q = Quit")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle chaos engine input — non-blocking."""
        user_text = event.value.strip()
        event.input.value = ""

        if not user_text:
            return

        self._log("Chaos Engine", f"Injecting: {user_text}")
        self._run_chaos(user_text)

    @work(thread=True)
    def _run_chaos(self, user_text: str) -> None:
        """
        Background worker — runs the processing pipeline.

        Flow:
            1. Extract machine ID from text
            2. Build sensor window (50 x 18)
            3. Predict RUL
            4. Build result → update state → refresh panes
        """
        machine_id = self._extract_machine_id(user_text)
        base_window = self.state.get_sensor_window()

        rul = predict_rul(base_window)

        if rul <= 15:
            new_status = "OFFLINE"
        elif rul <= 30:
            new_status = "DEGRADED"
        else:
            new_status = "ONLINE"

        result = {
            "valid": True,
            "rejection_reason": "",
            "spike": {
                "sensor_id": "Xs4",
                "spike_value": 0.95,
                "affected_window_positions": [45, 46, 47, 48, 49],
                "fault_severity": "HIGH",
                "plain_english_summary": "Sensor spike detected in final 5 timesteps.",
            },
            "rul": rul,
            "capacity_report": {
                "machine_id": machine_id,
                "machine_name": self.state.machines[machine_id].name,
                "status": new_status,
                "rul": rul,
                "capacity_pct": 80.0,
                "machine_req": 18.594,
                "breakeven_risk": True,
            },
            "dispatch_orders": f"[Floor Manager] Machine {machine_id} flagged for maintenance.",
            "machine_statuses": [
                {
                    "id": mid,
                    "name": m.name,
                    "status": new_status if mid == machine_id else m.status,
                    "rul": rul if mid == machine_id else m.rul,
                    "available_time": 0.0 if (mid == machine_id and new_status == "OFFLINE")
                        else (m.base_time * 0.5 if (mid == machine_id and new_status == "DEGRADED")
                        else m.available_time),
                    "base_time": m.base_time,
                }
                for mid, m in self.state.machines.items()
            ],
            "used_fallback": False,
        }

        self._process_result(result, machine_id)

    def _process_result(self, result: dict, machine_id: int) -> None:
        """Process pipeline result — log to comms, update state, refresh panes."""

        if not result["valid"]:
            self.call_from_thread(
                self._log, "Chaos Engine", f"REJECTED: {result['rejection_reason']}"
            )
            return

        spike = result["spike"]
        self.call_from_thread(
            self._log, "Diagnostic Agent",
            f"Sensor {spike['sensor_id']} spike to {spike['spike_value']:.2f} "
            f"({spike['fault_severity']}) — {spike['plain_english_summary']}"
        )

        old_rul = self.state.machines[machine_id].rul
        self.call_from_thread(
            self._log, "DL Oracle",
            f"RUL updated: {old_rul:.0f} → {result['rul']:.1f} cycles"
        )

        cap = result["capacity_report"]
        self.call_from_thread(
            self._log, "Capacity Agent",
            f"Machine {cap['machine_id']} {cap['status']}. "
            f"Capacity: {cap['capacity_pct']}%. ΣPD/T: {cap['machine_req']}"
        )

        self.call_from_thread(
            self._log, "Floor Manager", result["dispatch_orders"]
        )

        if result["used_fallback"]:
            self.call_from_thread(
                self._log, "System",
                "[bold yellow]⚠ Running in OFFLINE MODE — using cached responses[/bold yellow]"
            )

        self.state.update_from_agent_result(result)

        self.call_from_thread(self._refresh_sensor_pane)
        self.call_from_thread(self._refresh_capacity_pane)

    def _refresh_sensor_pane(self) -> None:
        """Update Pane 1: Live Sensor Feed."""
        pane = self.query_one("#sensor-pane", SensorFeedWidget)
        pane.refresh_content(self.state)

    def _refresh_capacity_pane(self) -> None:
        """Update Pane 2: Factory Capacity Dashboard."""
        pane = self.query_one("#capacity-pane", CapacityWidget)
        pane.refresh_content(self.state)

    def _log(self, agent: str, message: str) -> None:
        """Write to comms log pane (Pane 3)."""
        log_widget = self.query_one("#comms-pane", RichLog)
        formatted = format_log_entry(agent, message)
        log_widget.write(formatted)
        self.state.add_log_entry(agent, message)

    def _extract_machine_id(self, text: str) -> int:
        """
        Extract machine ID from user input.
        Looks for 'Machine N' or machine names.
        Falls back to cycling through machines 1-5.
        """
        match = re.search(r'[Mm]achine\s+(\d)', text)
        if match:
            mid = int(match.group(1))
            if 1 <= mid <= 5:
                return mid

        name_map = {
            "alpha": 1, "cnc-alpha": 1,
            "beta": 2, "cnc-beta": 2,
            "gamma": 3, "press-gamma": 3,
            "delta": 4, "lathe-delta": 4,
            "epsilon": 5, "mill-epsilon": 5,
        }
        text_lower = text.lower()
        for name, mid in name_map.items():
            if name in text_lower:
                return mid

        self._machine_cycle = (self._machine_cycle % 5) + 1
        return self._machine_cycle

    def action_reset_factory(self) -> None:
        """Reset all machines to ONLINE. Triggered by Ctrl+R."""
        self.state.reset_all()
        self._refresh_sensor_pane()
        self._refresh_capacity_pane()
        self._log("System", "Factory RESET — all machines restored to ONLINE.")


if __name__ == "__main__":
    app = FactoryApp()
    app.run()

"""
Widget classes for the 4-pane factory dashboard.

Pane 1: SensorFeedWidget   — Live sparklines + RUL
Pane 2: CapacityWidget      — All 5 machines + capacity metrics
Pane 3: CommsLogWidget      — Agent comms log (handled in app.py)
Pane 4: ChaosInputWidget    — Text input (handled in app.py)
"""

from textual.widgets import Static
from datetime import datetime


def mini_sparkline(values: list, width: int = 20) -> str:
    """Render a sparkline using Unicode block characters."""
    if not values:
        return "─" * width
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1.0
    vals = values[-width:]
    chars = []
    for v in vals:
        idx = int(((v - mn) / rng) * (len(blocks) - 1))
        idx = max(0, min(idx, len(blocks) - 1))
        chars.append(blocks[idx])
    return "".join(chars)


def status_bar(status: str) -> str:
    """Visual bar for machine status."""
    return {
        "ONLINE": "████████",
        "DEGRADED": "████░░░░",
        "OFFLINE": "░░░░░░░░",
    }.get(status, "????????")


def status_color(status: str) -> str:
    """Color for machine status."""
    return {"ONLINE": "green", "DEGRADED": "yellow", "OFFLINE": "red"}.get(status, "white")


def rul_color(rul: float) -> str:
    """Color based on RUL value."""
    if rul > 30:
        return "green"
    elif rul > 15:
        return "yellow"
    else:
        return "red"


def rul_label(rul: float) -> str:
    """Text label based on RUL value."""
    if rul > 30:
        return "HEALTHY"
    elif rul > 15:
        return "WARNING"
    else:
        return "CRITICAL"


class SensorFeedWidget(Static):
    """
    Live sensor feed with sparklines and RUL indicator.

    Displays 18 sensors (W0-W3 operating conditions, Xs0-Xs13 physical sensors)
    as Unicode sparklines for the active machine.
    """

    def refresh_content(self, state):
        """Rebuild sensor pane from factory state."""
        mid = state.active_machine_id
        machine = state.machines[mid]
        rc = rul_color(machine.rul)
        rl = rul_label(machine.rul)

        lines = [
            f"[bold]LIVE SENSOR FEED — {machine.name}[/bold]",
            "",
            f"  RUL: [{rc}]{machine.rul:.0f} cycles ▼ {rl}[/{rc}]",
            "",
        ]

        sensor_names = [f"W{i}" for i in range(4)] + [f"Xs{i}" for i in range(14)]
        for i, name in enumerate(sensor_names):
            history = state.sensor_history[i]
            if history:
                spark = mini_sparkline(history[-20:])
                lines.append(f"  {name:>4s} {spark}")
            else:
                lines.append(f"  {name:>4s} ────────────────────")

        self.update("\n".join(lines))


class CapacityWidget(Static):
    """
    Factory capacity dashboard showing all 5 machines and metrics.

    Displays:
    - Machine status (ONLINE/DEGRADED/OFFLINE) with visual bars
    - RUL for each machine
    - Overall capacity percentage
    - Break-even risk metric (ΣPD/T)
    """

    def refresh_content(self, state):
        """Rebuild capacity pane from factory state."""
        lines = [
            "[bold]FACTORY CAPACITY DASHBOARD[/bold]",
            "",
        ]

        for mid, machine in state.machines.items():
            bar = status_bar(machine.status)
            color = status_color(machine.status)
            lines.append(
                f"  Machine {mid}: [{color}]{bar} {machine.status:>8s}[/{color}]"
                f"  RUL: {machine.rul:.0f}"
            )

        lines.append("")
        risk_flag = " [bold red]⚠ CRITICAL[/bold red]" if state.breakeven_risk else ""
        lines.append(
            f"  Capacity: {state.capacity_pct:.0f}% | "
            f"ΣPD/T: {state.machine_req:.2f}{risk_flag}"
        )

        self.update("\n".join(lines))


AGENT_COLORS = {
    "System":           "dim",
    "Chaos Engine":     "bold magenta",
    "Diagnostic Agent": "cyan",
    "DL Oracle":        "bold green",
    "Capacity Agent":   "yellow",
    "Floor Manager":    "bold red",
}


def format_log_entry(agent: str, message: str) -> str:
    """Format a comms log entry with timestamp and color."""
    ts = datetime.now().strftime("%H:%M:%S")
    color = AGENT_COLORS.get(agent, "white")
    return f"[dim]{ts}[/dim] [{color}][{agent}][/{color}] {message}"

# TEAM TERMINAL UI — Technical Runbook

## Textual-Based Factory Floor Dashboard

**Owners:** 2 people · **Environment:** Fully local, no GPU, no API keys needed

**Your deliverable:** A Textual terminal app (`python -m terminal.app`) that displays a 4-pane factory dashboard, accepts chaos input from the professor, pipes it through the agent loop, and updates all panes live. You own `/terminal/*`.

---

## 0. What You're Building

```
┌─────────────────────────────────────┬──────────────────────────────┐
│  PANE 1: LIVE SENSOR FEED           │  PANE 2: CAPACITY DASHBOARD  │
│  Machine [N] — 18-sensor sparklines │  Machine 1: ████████ ONLINE  │
│  RUL: XX cycles ▼ [severity color]  │  Machine 2: ████████ ONLINE  │
│                                     │  Machine 3: ████▒▒▒▒ DEGRADED│
│                                     │  Machine 4: ░░░░░░░░ OFFLINE │
│                                     │  Machine 5: ████████ ONLINE  │
│                                     │  Capacity: 82% | ΣPD/T: 1.12│
├─────────────────────────────────────┴──────────────────────────────┤
│  PANE 3: AGENT COMMS LOG                                           │
│  [Chaos Engine] REJECTED: Unrecognized fault type. Try: ...        │
│  [Diagnostic Agent] Translating: "high-pressure temp surge"...     │
│  [DL Oracle]        RUL updated: 150 → 22 cycles                  │
│  [Capacity Agent]   Machine 3 DEGRADED. Capacity: 90%. ΣPD/T: 0.84│
│  [Floor Manager]    Reducing Machine 3 to 50% load...              │
├────────────────────────────────────────────────────────────────────┤
│  PANE 4: CHAOS ENGINE >  _                                         │
└────────────────────────────────────────────────────────────────────┘
```

You are the **only team the professor sees**. The DL model is invisible. The agents are invisible. All anyone experiences is your terminal. Make it look and feel good.

---

## 1. Tech Stack

```bash
pip install textual rich numpy
```

| Library | Version | Purpose |
|---|---|---|
| **Textual** | ≥0.40 | TUI framework — async, widgets, CSS-like styling |
| **Rich** | (bundled with Textual) | Color, markup, tables inside Textual widgets |
| **numpy** | ≥1.24 | Array handling for sensor windows |

**Why Textual, not raw Rich?** Rich's `console.input()` blocks the event loop. You cannot update the UI while waiting for input. Textual solves this with native async event handling — the `Input` widget fires events without blocking anything.

---

## 2. File Structure

```
/terminal/
├── app.py              # Main Textual app — entry point, event handlers
├── layout.py           # Widget classes for each of the 4 panes
├── factory_state.py    # Shared mutable state (machines, sensor history, RUL)
└── dummy_oracle.py     # Stub predict_rul() for days 1–9
```

---

## 3. Dummy Oracle — Your Day 1 Friend

You need this from the very first hour. It lets you build and test the entire UI + agent pipeline without waiting for Team DL.

```python
# terminal/dummy_oracle.py

import numpy as np

def predict_rul(sensor_tensor: np.ndarray) -> float:
    """
    Stub oracle. Returns a fixed RUL for integration testing.

    Same signature as dl_engine.inference.predict_rul:
        Input:  numpy array of shape (50, 18)
        Output: float RUL prediction

    On Day 10, this import gets swapped to the real model.
    """
    # Return 15.0 to test the OFFLINE threshold path
    # Change to 25.0 to test DEGRADED, or 50.0 to test ONLINE
    return 15.0
```

**Swap on Day 10:**

```python
# Before (Days 1-9):
from terminal.dummy_oracle import predict_rul

# After (Day 10+):
from dl_engine.inference import predict_rul
```

One import line. That's it.

---

## 4. Factory State (`factory_state.py`)

This is the shared mutable state that all panes read from. When the agent loop updates this, the UI reactively refreshes.

```python
# terminal/factory_state.py

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MachineState:
    """State of one factory machine."""
    id: int
    name: str
    status: str = "ONLINE"          # "ONLINE" | "DEGRADED" | "OFFLINE"
    rul: float = 999.0              # Current RUL prediction
    base_time: float = 8.0          # Max available hours
    available_time: float = 8.0     # Current available hours


@dataclass
class FactoryState:
    """Global factory state — single source of truth for all panes."""

    machines: dict = field(default_factory=lambda: {
        1: MachineState(1, "CNC-Alpha"),
        2: MachineState(2, "CNC-Beta"),
        3: MachineState(3, "Press-Gamma"),
        4: MachineState(4, "Lathe-Delta"),
        5: MachineState(5, "Mill-Epsilon"),
    })

    # Sensor history for sparklines — ring buffer of last N readings per sensor
    sensor_history: list = field(default_factory=lambda: [
        [] for _ in range(18)  # 18 sensors, each a list of recent values
    ])
    HISTORY_LENGTH: int = 60   # Keep last 60 readings for sparkline

    # Current selected machine for the sensor pane
    active_machine_id: int = 1

    # Capacity metrics (recomputed after each agent loop)
    capacity_pct: float = 100.0
    machine_req: float = 0.0
    breakeven_risk: bool = False

    # Comms log entries
    comms_log: list = field(default_factory=list)
    MAX_LOG_ENTRIES: int = 100

    def update_from_agent_result(self, result: dict):
        """
        Called after agent_loop returns. Updates all state in one shot.

        Args:
            result: dict from agents.agent_loop.run_agent_loop()
        """
        if not result.get("valid", False):
            return

        # Update machine statuses from capacity agent
        for ms in result.get("machine_statuses", []):
            mid = ms["id"]
            if mid in self.machines:
                self.machines[mid].status = ms["status"]
                self.machines[mid].rul = ms["rul"]
                self.machines[mid].available_time = ms["available_time"]

        # Update capacity metrics
        report = result.get("capacity_report", {})
        if report:
            self.capacity_pct = report.get("capacity_pct", self.capacity_pct)
            self.machine_req = report.get("machine_req", self.machine_req)
            self.breakeven_risk = report.get("breakeven_risk", self.breakeven_risk)

    def push_sensor_reading(self, sensor_values: np.ndarray):
        """
        Push one timestep of sensor data (18 values) into the ring buffer.
        Used for sparkline animation.
        """
        for i, val in enumerate(sensor_values):
            self.sensor_history[i].append(float(val))
            if len(self.sensor_history[i]) > self.HISTORY_LENGTH:
                self.sensor_history[i].pop(0)

    def add_log_entry(self, agent_name: str, message: str):
        """Add a timestamped entry to the comms log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.comms_log.append({
            "time": timestamp,
            "agent": agent_name,
            "message": message,
        })
        # Trim old entries
        if len(self.comms_log) > self.MAX_LOG_ENTRIES:
            self.comms_log = self.comms_log[-self.MAX_LOG_ENTRIES:]
```

---

## 5. Main App (`app.py`)

```python
# terminal/app.py

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static, ProgressBar
from textual.binding import Binding
from textual import work

import numpy as np

from .factory_state import FactoryState
from .dummy_oracle import predict_rul  # ← Swapped to dl_engine on Day 10

# Import agent loop — available after Team Agent delivers (Day 5+)
# from agents.agent_loop import run_agent_loop


class FactoryApp(App):
    """Agentic Predictive Maintenance — Terminal Simulation."""

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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # ── Pane 1: Sensor Feed ──
        yield Static(id="sensor-pane", markup=True)

        # ── Pane 2: Capacity Dashboard ──
        yield Static(id="capacity-pane", markup=True)

        # ── Pane 3: Agent Comms Log ──
        yield RichLog(id="comms-pane", highlight=True, markup=True)

        # ── Pane 4: Chaos Engine Input ──
        yield Input(
            id="chaos-input",
            placeholder="CHAOS ENGINE > Type a fault description and press Enter...",
        )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the display on startup."""
        self._refresh_sensor_pane()
        self._refresh_capacity_pane()
        self._log("System", "Factory simulation online. All machines nominal.")
        self._log("System", "Type a fault description below to inject chaos.")

    # ═══════════════════════════════════════════════════════
    # EVENT: Professor types a fault and presses Enter
    # ═══════════════════════════════════════════════════════

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle chaos engine input — non-blocking."""
        user_text = event.value.strip()
        event.input.value = ""  # Clear the input field immediately

        if not user_text:
            return

        self._log("Chaos Engine", f"Injecting: {user_text}")

        # Run agent loop in background worker (non-blocking)
        self._run_chaos(user_text)

    @work(thread=True)
    def _run_chaos(self, user_text: str) -> None:
        """
        Background worker — runs the full agent pipeline.
        Uses @work(thread=True) so Gemini API calls don't block the UI.
        """
        # Choose a machine (simple: cycle through, or parse from input)
        machine_id = self._extract_machine_id(user_text)

        # Create a base sensor window (50 timesteps × 18 sensors)
        # In production, this should be the last 50 readings from sensor_history
        base_window = self._get_current_sensor_window()

        # ── Run the full pipeline ──
        # UNCOMMENT when Team Agent delivers agent_loop.py:
        #
        # from agents.agent_loop import run_agent_loop
        # result = run_agent_loop(
        #     user_text=user_text,
        #     machine_id=machine_id,
        #     base_window=base_window,
        #     predict_rul_fn=predict_rul,
        # )

        # ── STUB for Days 1-4 (before agent_loop exists) ──
        rul = predict_rul(base_window)
        result = {
            "valid": True,
            "rejection_reason": "",
            "spike": {"sensor_id": "Xs4", "spike_value": 0.95,
                      "fault_severity": "HIGH",
                      "plain_english_summary": "Stub diagnostic result."},
            "rul": rul,
            "capacity_report": {
                "machine_id": machine_id,
                "machine_name": self.state.machines[machine_id].name,
                "status": "OFFLINE" if rul <= 15 else "ONLINE",
                "rul": rul,
                "capacity_pct": 80.0,
                "machine_req": 18.594,
                "breakeven_risk": True,
            },
            "dispatch_orders": f"[Floor Manager] Machine {machine_id} flagged. Stub dispatch.",
            "machine_statuses": [
                {"id": mid, "name": m.name, "status": m.status,
                 "rul": m.rul, "available_time": m.available_time, "base_time": m.base_time}
                for mid, m in self.state.machines.items()
            ],
            "used_fallback": False,
        }
        # ── END STUB ──

        # ── Process result ──
        if not result["valid"]:
            self.call_from_thread(
                self._log, "Chaos Engine", f"REJECTED: {result['rejection_reason']}"
            )
            return

        # Log each agent's output
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
                self._log, "System", "⚠ Running in OFFLINE MODE — using cached responses"
            )

        # Update factory state
        self.state.update_from_agent_result(result)

        # Refresh UI panes (must be called from main thread)
        self.call_from_thread(self._refresh_sensor_pane)
        self.call_from_thread(self._refresh_capacity_pane)

    # ═══════════════════════════════════════════════════════
    # PANE RENDERERS
    # ═══════════════════════════════════════════════════════

    def _refresh_sensor_pane(self) -> None:
        """Update Pane 1: Live Sensor Feed."""
        pane = self.query_one("#sensor-pane", Static)
        mid = self.state.active_machine_id
        machine = self.state.machines[mid]

        rul_color = self._rul_color(machine.rul)
        rul_label = self._rul_label(machine.rul)

        lines = [
            f"[bold]LIVE SENSOR FEED — {machine.name}[/bold]",
            "",
            f"  RUL: [{rul_color}]{machine.rul:.0f} cycles ▼ {rul_label}[/{rul_color}]",
            "",
        ]

        # Mini sparklines for each sensor (using recent history)
        sensor_names = [f"W{i}" for i in range(4)] + [f"Xs{i}" for i in range(14)]
        for i, name in enumerate(sensor_names):
            history = self.state.sensor_history[i]
            if history:
                spark = self._mini_sparkline(history[-20:])  # Last 20 values
                lines.append(f"  {name:>4s} {spark}")
            else:
                lines.append(f"  {name:>4s} ────────────────────")

        pane.update("\n".join(lines))

    def _refresh_capacity_pane(self) -> None:
        """Update Pane 2: Factory Capacity Dashboard."""
        pane = self.query_one("#capacity-pane", Static)

        lines = [
            "[bold]FACTORY CAPACITY DASHBOARD[/bold]",
            "",
        ]

        for mid, machine in self.state.machines.items():
            bar = self._status_bar(machine.status)
            color = self._status_color(machine.status)
            lines.append(
                f"  Machine {mid}: [{color}]{bar} {machine.status:>8s}[/{color}]"
                f"  RUL: {machine.rul:.0f}"
            )

        lines.append("")
        risk_flag = " [bold red]⚠ CRITICAL[/bold red]" if self.state.breakeven_risk else ""
        lines.append(
            f"  Capacity: {self.state.capacity_pct:.0f}% | "
            f"ΣPD/T: {self.state.machine_req:.2f}{risk_flag}"
        )

        pane.update("\n".join(lines))

    # ═══════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════

    def _log(self, agent: str, message: str) -> None:
        """Write to the comms log pane."""
        log_widget = self.query_one("#comms-pane", RichLog)
        color = {
            "System":           "dim",
            "Chaos Engine":     "bold magenta",
            "Diagnostic Agent": "cyan",
            "DL Oracle":        "bold green",
            "Capacity Agent":   "yellow",
            "Floor Manager":    "bold red",
        }.get(agent, "white")

        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        log_widget.write(f"[dim]{ts}[/dim] [{color}][{agent}][/{color}] {message}")

        # Also store in state
        self.state.add_log_entry(agent, message)

    def _extract_machine_id(self, text: str) -> int:
        """
        Try to extract a machine ID from the input text.
        Falls back to cycling through machines 1-5.
        """
        import re
        # Look for "Machine N" or "machine N"
        match = re.search(r'[Mm]achine\s+(\d)', text)
        if match:
            mid = int(match.group(1))
            if 1 <= mid <= 5:
                return mid

        # Look for machine names
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

        # Default: cycle through machines
        if not hasattr(self, '_machine_cycle'):
            self._machine_cycle = 0
        self._machine_cycle = (self._machine_cycle % 5) + 1
        return self._machine_cycle

    def _get_current_sensor_window(self) -> np.ndarray:
        """
        Build a (50, 18) base sensor window from recent history.
        If not enough history, pad with random normal values.
        """
        window = np.zeros((50, 18), dtype=np.float32)
        for sensor_idx in range(18):
            history = self.state.sensor_history[sensor_idx]
            if len(history) >= 50:
                window[:, sensor_idx] = history[-50:]
            elif len(history) > 0:
                # Pad front with the earliest known value
                padding = [history[0]] * (50 - len(history))
                window[:, sensor_idx] = padding + history
            else:
                # No history — use random baseline values in [0.3, 0.7]
                window[:, sensor_idx] = np.random.uniform(0.3, 0.7, size=50)
        return window

    @staticmethod
    def _mini_sparkline(values: list, width: int = 20) -> str:
        """Render a tiny sparkline string from a list of float values."""
        if not values:
            return "─" * width
        blocks = " ▁▂▃▄▅▆▇█"
        mn, mx = min(values), max(values)
        rng = mx - mn if mx > mn else 1.0
        # Take last `width` values
        vals = values[-width:]
        chars = []
        for v in vals:
            idx = int(((v - mn) / rng) * (len(blocks) - 1))
            idx = max(0, min(idx, len(blocks) - 1))
            chars.append(blocks[idx])
        return "".join(chars)

    @staticmethod
    def _status_bar(status: str) -> str:
        """Visual bar for machine status."""
        if status == "ONLINE":
            return "████████"
        elif status == "DEGRADED":
            return "████░░░░"
        else:
            return "░░░░░░░░"

    @staticmethod
    def _status_color(status: str) -> str:
        """Rich color name for machine status."""
        return {"ONLINE": "green", "DEGRADED": "yellow", "OFFLINE": "red"}.get(status, "white")

    @staticmethod
    def _rul_color(rul: float) -> str:
        if rul > 30:
            return "green"
        elif rul > 15:
            return "yellow"
        else:
            return "red"

    @staticmethod
    def _rul_label(rul: float) -> str:
        if rul > 30:
            return "HEALTHY"
        elif rul > 15:
            return "WARNING"
        else:
            return "CRITICAL"

    def action_reset_factory(self) -> None:
        """Reset all machines to ONLINE. Triggered by Ctrl+R."""
        for m in self.state.machines.values():
            m.status = "ONLINE"
            m.rul = 999.0
            m.available_time = m.base_time
        self.state.capacity_pct = 100.0
        self.state.machine_req = 0.0
        self.state.breakeven_risk = False
        self._refresh_sensor_pane()
        self._refresh_capacity_pane()
        self._log("System", "Factory RESET — all machines restored to ONLINE.")


# ── Entry point ──
if __name__ == "__main__":
    app = FactoryApp()
    app.run()
```

### 5.1 Running the App

```bash
# From repo root:
python -m terminal.app

# Or directly:
cd terminal && python app.py
```

### 5.2 Key Textual Concepts You Need to Know

| Concept | What it does | Where you use it |
|---|---|---|
| `ComposeResult` / `yield` | Declares the widget tree (like React's JSX) | `compose()` method |
| `CSS` class variable | Textual's CSS-like styling — grid layout, colors, borders | Layout positioning |
| `on_input_submitted` | Fires when user presses Enter in an `Input` widget | Chaos Engine handler |
| `@work(thread=True)` | Runs a method in a background thread (non-blocking) | Agent loop (Gemini calls) |
| `call_from_thread()` | Safely update UI from a background thread | Logging + pane refresh |
| `Static.update()` | Replace the content of a `Static` widget | Sensor pane, capacity pane |
| `RichLog.write()` | Append a line to a scrolling log | Comms log |
| `Binding` | Keyboard shortcuts | Ctrl+R = reset, Ctrl+Q = quit |

### 5.3 The `@work` + `call_from_thread` Pattern

This is the single most important pattern in your codebase. The Gemini API calls in the agent loop take 1-3 seconds. If you run them on the main thread, the entire UI freezes.

```python
# WRONG — UI freezes for 2 seconds
async def on_input_submitted(self, event):
    result = run_agent_loop(...)  # blocks the entire app
    self._refresh_panes()

# RIGHT — UI stays responsive
@work(thread=True)
def _run_chaos(self, user_text: str):
    result = run_agent_loop(...)  # runs in background thread
    self.call_from_thread(self._refresh_sensor_pane)  # safely update UI
```

**Rule:** Anything that touches the network goes in a `@work(thread=True)` method. Anything that updates UI widgets uses `call_from_thread()`.

---

## 6. Widget Reference — What to Use Where

### Pane 1: Sensor Feed

**Widget:** `Static` with Rich markup, manually updated.

**Why not a real chart widget?** Textual doesn't have a built-in chart widget. You have two options:

1. **ASCII sparklines** (recommended, shown above) — simple, fast, looks good in terminal
2. **Custom widget** using `textual-plotext` — fancier but more work

The sparkline approach using Unicode block characters (`▁▂▃▄▅▆▇█`) gives you a clean visual with zero dependencies.

### Pane 2: Capacity Dashboard

**Widget:** `Static` with Rich markup.

For the progress bars, you can either use Rich markup strings (shown above with `████████`) or Textual's built-in `ProgressBar` widget:

```python
# Option A: Markup strings (simpler, more control)
"Machine 1: [green]████████[/green] ONLINE"

# Option B: Textual ProgressBar (fancier)
bar = ProgressBar(total=100, show_eta=False)
bar.update(progress=capacity_pct)
```

Option A is recommended because it's easier to color-code per-machine.

### Pane 3: Comms Log

**Widget:** `RichLog` — this is purpose-built for scrolling logs with Rich markup. It auto-scrolls, supports colors, and handles overflow.

```python
log = self.query_one("#comms-pane", RichLog)
log.write("[bold cyan][Diagnostic Agent][/bold cyan] Sensor Xs4 spike to 0.95")
```

### Pane 4: Chaos Engine

**Widget:** `Input` — native text input with placeholder.

```python
yield Input(
    id="chaos-input",
    placeholder="CHAOS ENGINE > Type a fault description...",
)
```

---

## 7. Color Scheme

Use these consistently across all panes:

| Element | Color | Rich markup |
|---|---|---|
| System messages | Dim gray | `[dim]...[/dim]` |
| Chaos Engine | Bold magenta | `[bold magenta]...[/bold magenta]` |
| Diagnostic Agent | Cyan | `[cyan]...[/cyan]` |
| DL Oracle | Bold green | `[bold green]...[/bold green]` |
| Capacity Agent | Yellow | `[yellow]...[/yellow]` |
| Floor Manager | Bold red | `[bold red]...[/bold red]` |
| ONLINE status | Green | `[green]...[/green]` |
| DEGRADED status | Yellow | `[yellow]...[/yellow]` |
| OFFLINE status | Red | `[red]...[/red]` |
| RUL > 30 | Green | `[green]...[/green]` |
| 15 < RUL ≤ 30 | Yellow | `[yellow]...[/yellow]` |
| RUL ≤ 15 | Red | `[red]...[/red]` |
| Break-even warning | Bold red | `[bold red]⚠ CRITICAL[/bold red]` |
| Fallback mode flag | Bold yellow | `[bold yellow]⚠ OFFLINE MODE[/bold yellow]` |

---

## 8. Integration with Team Agent

### 8.1 What You Call

```python
from agents.agent_loop import run_agent_loop

result = run_agent_loop(
    user_text="bearing temperature spike on Machine 4",
    machine_id=4,
    base_window=np.random.randn(50, 18).astype(np.float32),
    predict_rul_fn=predict_rul,  # from dummy_oracle or dl_engine.inference
)
```

### 8.2 What You Get Back

```python
result = {
    "valid":            True,       # Did input pass the guard?
    "rejection_reason": "",         # Why rejected (empty if valid)
    "spike": {                      # Agent 1 output
        "sensor_id": "Xs4",
        "spike_value": 0.95,
        "affected_window_positions": [45, 46, 47, 48, 49],
        "fault_severity": "HIGH",
        "plain_english_summary": "Bearing temp critical."
    },
    "rul":              12.0,       # Agent 2 input / DL Oracle output
    "capacity_report": {            # Agent 2 output
        "machine_id": 4,
        "machine_name": "Lathe-Delta",
        "status": "OFFLINE",
        "rul": 12.0,
        "total_T": 32.0,
        "total_PD": 595,
        "machine_req": 18.594,
        "capacity_pct": 80.0,
        "breakeven_risk": True,
    },
    "dispatch_orders":  "[Floor Manager] Machine 4 OFFLINE...",  # Agent 3 output
    "machine_statuses": [...],      # All 5 machines for dashboard refresh
    "used_fallback":    False,      # True if Gemini was unavailable
}
```

### 8.3 What to Display from Each Field

| Field | Pane | What to show |
|---|---|---|
| `valid == False` | Pane 3 (log) | `[Chaos Engine] REJECTED: {rejection_reason}` |
| `spike` | Pane 3 (log) | `[Diagnostic Agent] Sensor {id} spike to {value} — {summary}` |
| `rul` | Pane 1 + Pane 3 | Update RUL display + log `[DL Oracle] RUL: old → new` |
| `capacity_report` | Pane 2 + Pane 3 | Update dashboard bars + log capacity/breakeven |
| `dispatch_orders` | Pane 3 (log) | Display the floor manager text verbatim |
| `machine_statuses` | Pane 2 | Refresh all 5 machine bars |
| `used_fallback` | Pane 3 (log) | Show warning if in offline mode |

---

## 9. Your Day-by-Day Checklist

| Day | Task | Done? |
|---|---|---|
| 1 | `pip install textual`. Scaffold `app.py` with 4-pane grid layout. Verify it runs with `python -m terminal.app`. | ☐ |
| 2 | Wire `dummy_oracle.py` into the app. Type any text → see "RUL: 15.0" update in sensor pane. Capacity pane shows all 5 machines. | ☐ |
| 3 | Build comms log pane with `RichLog`. Color-code agent messages. Test with hardcoded log entries. | ☐ |
| 4 | Build CHAOS ENGINE input with async handler (`@work` + `call_from_thread`). Typing input should not freeze UI. | ☐ |
| 5 | Add sparkline sensor visualization. **Phase 1 integration with Team Agent** — wire `run_agent_loop()` into the app. | ☐ |
| 6 | Add color-coded RUL states (green/amber/red) in both sensor pane and capacity dashboard. | ☐ |
| 7 | Add 3-state machine bars (full/half/empty) in capacity dashboard. Add breakeven warning. | ☐ |
| 8 | UI polish — consistent colors, clean layout, smooth text animation timing. | ☐ |
| 9 | **UI FREEZE.** No more layout changes after this day. Begin integration testing with dummy oracle. | ☐ |
| **10** | **INTEGRATION: Swap `dummy_oracle` → `dl_engine.inference`. Test with real model.** | ☐ |
| 11 | Fix any rendering issues with real RUL values (numbers might be larger/smaller than expected). | ☐ |
| 12 | Full rehearsal — run all 8-10 demo scenarios. | ☐ |
| 13 | **Record 3-minute backup demo video** in case of catastrophic failure on presentation day. | ☐ |
| 14 | Final bug fixes only. | ☐ |

---

## 10. Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| UI freezes when typing chaos input | Agent loop running on main thread | Use `@work(thread=True)` decorator |
| `call_from_thread` crashes | Trying to update widget from background thread without wrapper | Always use `self.call_from_thread(method, args)` |
| Pane layout looks wrong | CSS grid not configured correctly | Double-check `grid-size`, `column-span`, `row-span` |
| RichLog doesn't auto-scroll | Default behavior issue in some Textual versions | Set `auto_scroll=True` on `RichLog` |
| Sparklines look flat | All sensor values are in [0, 1] after normalization | That's correct — adjust sparkline scaling to [0, 1] range |
| Input field disappears | CSS stacking issue | Use `dock: bottom` for the input widget |
| Agent loop not found on Day 1-4 | Team Agent hasn't delivered yet | Use the stub in `_run_chaos()` — that's why it's there |
| Real model returns unexpected RUL range | Model trained on different scale than expected | Your display code should handle any positive float — don't hardcode thresholds |

---

## 11. Demo Day Checklist

```
[ ] App launches cleanly with `python -m terminal.app`
[ ] All 5 machines show ONLINE on startup
[ ] Typing a fault description updates all 4 panes
[ ] Invalid input shows REJECTED in comms log
[ ] DEGRADED state shows yellow + half bar
[ ] OFFLINE state shows red + empty bar
[ ] Breakeven warning appears when ΣPD/T > 1.0
[ ] Ctrl+R resets all machines to ONLINE
[ ] UI never freezes during Gemini API calls
[ ] If wifi disconnects, app still works (offline fallback)
[ ] 3-minute backup video is recorded and saved
```

---

*You own `/terminal/*`. You are the face of the project. If the terminal looks broken, nobody cares how good the model is. Make it clean, make it responsive, make it impressive.*

"""
Position Manager Terminal UI - Monitor and manage wallet positions.

Features:
- DataTable showing all positions with PnL
- Filter toggle for current 15m market
- Close positions with confirmation modal
- Real-time updates via polling
"""

import asyncio
import logging
from typing import Dict, List, Optional

import pandas as pd
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Label, Static

from src.positions.config import PositionsConfig
from src.positions.core import Position, PositionsCore

# Configure logging (force=True to override existing config)
logging.basicConfig(
    filename='data/finger_blaster.log',
    level=logging.DEBUG,  # Use DEBUG to see detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger("FingerBlaster.Positions.GUI")
logger.setLevel(logging.DEBUG)  # Ensure this logger uses DEBUG level

# Color constants
COLOR_GREEN = "#10b981"
COLOR_RED = "#ef4444"
COLOR_YELLOW = "#f59e0b"
COLOR_MUTED = "#6b7280"


def _format_market_title_et(market: Dict) -> str:
    """
    Format market title with ET timestamps.

    Converts start/end dates from UTC to ET and formats them
    to match activetrader's display format.

    For 15-minute BTC markets, calculates the correct start time
    as end_time - 15 minutes (since API start_date may not reflect
    the scheduled market window).

    Args:
        market: Market data dictionary with start_date, end_date, title

    Returns:
        Formatted title string with ET timestamps
    """
    try:
        # Get title components
        base_title = market.get('title', '') or market.get('question', '')
        end_date = market.get('end_date')

        # If we have end date, format it in ET
        if end_date:
            try:
                # Parse end timestamp
                end_dt = pd.Timestamp(end_date)

                # Localize to UTC if naive
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize('UTC')

                # Convert to ET
                end_et = end_dt.tz_convert('America/New_York')

                # Calculate start time as 15 minutes before end
                # (for 15-minute BTC markets)
                start_et = end_et - pd.Timedelta(minutes=15)

                # Format: "BITCOIN UP OR DOWN - FEBRUARY 1, 4:30PM-4:45PM ET"
                # Extract market name from title (everything before the date)
                market_name = base_title.split(' - ')[0] if ' - ' in base_title else base_title

                # Format the timestamp range
                date_str = end_et.strftime('%B %d').upper()
                start_time = start_et.strftime('%I:%M%p').lstrip('0')
                end_time = end_et.strftime('%I:%M%p').lstrip('0')

                return f"{market_name.upper()} - {date_str}, {start_time}-{end_time} ET"
            except Exception as e:
                logger.debug(f"Error formatting timestamps: {e}")
                # Fall through to return base title

        # Return base title if we can't format timestamps
        return base_title

    except Exception as e:
        logger.error(f"Error formatting market title: {e}")
        return market.get('title', 'Unknown')


class ConfirmCloseScreen(ModalScreen[bool]):
    """Modal screen for confirming position close."""

    BINDINGS = [
        ("y", "confirm", "Yes"),
        ("enter", "confirm", "Yes"),
        ("n", "cancel", "No"),
        ("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ConfirmCloseScreen {
        align: center middle;
    }

    #confirm-dialog {
        width: 60;
        height: 15;
        border: thick $primary;
        background: $surface;
        padding: 2 3;
    }

    #confirm-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #confirm-pnl {
        width: 100%;
        text-align: center;
        margin-bottom: 2;
    }

    #confirm-hint {
        width: 100%;
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    #confirm-buttons {
        width: 100%;
        height: 3;
        align: center middle;
        layout: horizontal;
    }

    #confirm-buttons Button {
        margin: 0 2;
        min-width: 12;
    }
    """

    def __init__(
        self,
        position: Position,
        name: Optional[str] = None,
        id: Optional[str] = None,
    ):
        super().__init__(name=name, id=id)
        self.position = position

    def compose(self) -> ComposeResult:
        pnl_color = COLOR_GREEN if self.position.pnl_usd >= 0 else COLOR_RED
        pnl_sign = "+" if self.position.pnl_usd >= 0 else ""
        side_color = COLOR_GREEN if self.position.outcome == "Up" else COLOR_RED

        with Container(id="confirm-dialog"):
            yield Label(
                f"Close [{side_color}]{self.position.outcome}[/] Position?",
                id="confirm-title"
            )
            yield Label(
                f"{self.position.size:.2f} shares @ ${self.position.current_price:.3f}",
                id="confirm-hint"
            )
            yield Label(
                f"PnL: [{pnl_color}]{pnl_sign}${self.position.pnl_usd:.2f} "
                f"({pnl_sign}{self.position.pnl_pct:.1f}%)[/]",
                id="confirm-pnl"
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", id="yes", variant="success")
                yield Button("No", id="no", variant="error")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yes":
            self.dismiss(True)
        else:
            self.dismiss(False)


class StatusBar(Static):
    """Status bar showing current state."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filter_mode = "All Positions"
        self.market_info = "Loading..."
        self.position_count = 0
        self.total_pnl = 0.0
        self.filter_enabled = False  # Track if filtering to current market

    def update_status(
        self,
        filter_mode: Optional[str] = None,
        filter_enabled: Optional[bool] = None,
        market_info: Optional[str] = None,
        position_count: Optional[int] = None,
        total_pnl: Optional[float] = None,
    ) -> None:
        if filter_mode is not None:
            self.filter_mode = filter_mode
        if filter_enabled is not None:
            self.filter_enabled = filter_enabled
        if market_info is not None:
            self.market_info = market_info
        if position_count is not None:
            self.position_count = position_count
        if total_pnl is not None:
            self.total_pnl = total_pnl
        self.refresh()

    def render(self) -> str:
        pnl_color = COLOR_GREEN if self.total_pnl >= 0 else COLOR_RED
        pnl_sign = "+" if self.total_pnl >= 0 else ""

        # Only show market info when filtering to current market
        if self.filter_enabled:
            market_display = f"Market: {self.market_info[:60]} | "
        else:
            market_display = ""

        return (
            f"Filter: [{COLOR_YELLOW}]{self.filter_mode}[/] | "
            f"{market_display}"
            f"Positions: {self.position_count} | "
            f"Total PnL: [{pnl_color}]{pnl_sign}${self.total_pnl:.2f}[/]"
        )


class LogWidget(Static):
    """Log message display widget."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._messages: List[str] = []
        self._max_messages = 3

    def add_message(self, msg: str) -> None:
        self._messages.append(msg)
        if len(self._messages) > self._max_messages:
            self._messages.pop(0)
        self.refresh()

    def render(self) -> str:
        if not self._messages:
            return f"[{COLOR_MUTED}]Ready[/]"
        return " | ".join(self._messages[-self._max_messages:])


class PositionManagerApp(App):
    """Position Manager Terminal UI Application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-container {
        height: 100%;
    }

    #status-bar {
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    #positions-table {
        height: 1fr;
        border: solid $primary;
    }

    #log-bar {
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    DataTable {
        height: 100%;
    }

    DataTable > .datatable--cursor {
        background: $accent;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
        Binding("f", "toggle_filter", "Filter"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "close_position", "Close"),
        Binding("enter", "close_position", "Close"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self, config: Optional[PositionsConfig] = None):
        super().__init__()
        self.config = config or PositionsConfig()
        self.core: Optional[PositionsCore] = None
        self._positions: List[Position] = []
        self._selected_position: Optional[Position] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main-container"):
            yield StatusBar(id="status-bar")
            yield DataTable(id="positions-table", cursor_type="row")
            yield LogWidget(id="log-bar")
        yield Footer()

    async def on_mount(self) -> None:
        # Setup table columns
        table = self.query_one("#positions-table", DataTable)
        table.add_columns(
            "Side",
            "Size",
            "Entry",
            "Current",
            "PnL ($)",
            "PnL (%)",
            "Market",
        )

        # Initialize core
        self.core = PositionsCore(config=self.config)

        # Register callbacks
        self.core.on("positions_update", self._on_positions_update)
        self.core.on("market_update", self._on_market_update)
        self.core.on("filter_changed", self._on_filter_changed)
        self.core.on("position_closed", self._on_position_closed)
        self.core.on("log", self._on_log)

        # Set initial filter state (defaults to All Positions)
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_status(filter_mode="All Positions", filter_enabled=False)

        # Start core
        self.run_worker(self._start_core(), exclusive=True)

    async def _start_core(self) -> None:
        try:
            if self.core:
                await self.core.start()
        except Exception as e:
            logger.error(f"Error starting PositionsCore: {e}", exc_info=True)
            self._log(f"Error: {e}")

    async def _on_positions_update(self, positions: List[Position]) -> None:
        """Handle positions update from core."""
        # Store positions and schedule UI update in main event loop
        self._positions = positions

        def update_ui():
            self._update_table()
            self._update_status()

        self.call_later(update_ui)

    async def _on_market_update(self, market: Dict) -> None:
        """Handle market update from core."""
        try:
            logger.warning(f"GUI._on_market_update called with market: {market.get('market_id', 'Unknown')}")
            logger.warning(f"Market title from dict: {market.get('title', 'Unknown')}")

            # Format the market title
            title = _format_market_title_et(market)
            logger.warning(f"Formatted market title: {title}")

            # Update the status bar - use call_from_thread to ensure it runs in the Textual event loop
            def update_status_bar():
                try:
                    status_bar = self.query_one("#status-bar", StatusBar)
                    logger.warning(f"Current filter_enabled state: {status_bar.filter_enabled}")
                    status_bar.update_status(market_info=title)
                    logger.warning(f"Status bar updated with new market info: {title}")
                except Exception as e:
                    logger.error(f"Error updating status bar: {e}", exc_info=True)

            # Schedule the update in the main Textual loop
            self.call_later(update_status_bar)

        except Exception as e:
            logger.error(f"Error in _on_market_update: {e}", exc_info=True)

    async def _on_filter_changed(self, filter_enabled: bool) -> None:
        """Handle filter change."""
        status_bar = self.query_one("#status-bar", StatusBar)
        mode = "15m Market" if filter_enabled else "All Positions"
        status_bar.update_status(filter_mode=mode, filter_enabled=filter_enabled)

    async def _on_position_closed(self, token_id: str, success: bool) -> None:
        """Handle position close result."""
        if success:
            self._log(f"Position closed successfully")
        else:
            self._log(f"Failed to close position")

    async def _on_log(self, message: str) -> None:
        """Handle log message."""
        self._log(message)

    def _log(self, message: str) -> None:
        """Add log message to display."""
        try:
            log_widget = self.query_one("#log-bar", LogWidget)
            log_widget.add_message(message)
        except Exception:
            pass

    def _update_table(self) -> None:
        """Update the positions table."""
        table = self.query_one("#positions-table", DataTable)
        table.clear()

        for pos in self._positions:
            # Format values
            pnl_color = COLOR_GREEN if pos.pnl_usd >= 0 else COLOR_RED
            pnl_sign = "+" if pos.pnl_usd >= 0 else ""

            side_color = COLOR_GREEN if pos.outcome == "Up" else COLOR_RED
            side = f"[{side_color}]{pos.outcome}[/]"

            size = f"{pos.size:.2f}"
            entry = f"${pos.avg_entry_price:.3f}"
            current = f"${pos.current_price:.3f}"
            pnl_usd = f"[{pnl_color}]{pnl_sign}${pos.pnl_usd:.2f}[/]"
            pnl_pct = f"[{pnl_color}]{pnl_sign}{pos.pnl_pct:.1f}%[/]"

            # Market indicator
            market = pos.market_title[:20] if pos.market_title else ""
            if pos.is_current_market:
                market = f"[{COLOR_YELLOW}]*[/] {market}"

            table.add_row(side, size, entry, current, pnl_usd, pnl_pct, market)

    def _update_status(self) -> None:
        """Update status bar."""
        status_bar = self.query_one("#status-bar", StatusBar)
        total_pnl = sum(p.pnl_usd for p in self._positions)
        status_bar.update_status(
            position_count=len(self._positions),
            total_pnl=total_pnl,
        )

    def _get_selected_position(self) -> Optional[Position]:
        """Get the currently selected position."""
        table = self.query_one("#positions-table", DataTable)
        if table.cursor_row is not None and 0 <= table.cursor_row < len(self._positions):
            return self._positions[table.cursor_row]
        return None

    def action_toggle_filter(self) -> None:
        """Toggle position filter."""
        if self.core:
            self.core.toggle_filter()

    def action_refresh(self) -> None:
        """Refresh positions."""
        if self.core:
            self._log("Refreshing...")
            asyncio.create_task(self.core.refresh_positions())

    def action_close_position(self) -> None:
        """Close the selected position."""
        position = self._get_selected_position()
        if not position:
            self._log("No position selected")
            return

        # Show confirmation modal
        self.push_screen(
            ConfirmCloseScreen(position),
            self._handle_close_confirmation,
        )

    def _handle_close_confirmation(self, confirmed: bool) -> None:
        """Handle close confirmation result."""
        if confirmed:
            position = self._get_selected_position()
            if position and self.core:
                asyncio.create_task(self.core.close_position(position.token_id))

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        table = self.query_one("#positions-table", DataTable)
        table.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        table = self.query_one("#positions-table", DataTable)
        table.action_cursor_up()

    async def on_unmount(self) -> None:
        """Cleanup on exit."""
        if self.core:
            try:
                await asyncio.wait_for(self.core.stop(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("PositionsCore stop timed out")
            except Exception as e:
                logger.error(f"Error stopping PositionsCore: {e}")

    def action_quit(self) -> None:
        """Quit the application."""
        self.run_worker(self._shutdown())

    async def _shutdown(self) -> None:
        """Shutdown gracefully."""
        if self.core:
            try:
                await asyncio.wait_for(self.core.stop(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("PositionsCore stop timed out during shutdown")
            except Exception as e:
                logger.error(f"Error stopping PositionsCore: {e}")
        self.exit()


def run_positions_app(config: Optional[PositionsConfig] = None) -> None:
    """Run the Position Manager app."""
    try:
        app = PositionManagerApp(config=config)
        app.run()
    except Exception as e:
        logger.error(f"Position Manager crashed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_positions_app()

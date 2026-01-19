"""
PDF Export functionality for Ham Radio Olympics.

Provides offline-capable PDF exports of olympiad data, sport standings,
and competitor-specific reports for use in remote locations without internet.

The cached olympiad PDF is automatically regenerated when medals or records change.
"""

import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from fpdf import FPDF
from PIL import Image, ImageDraw

from database import get_db
from dxcc import get_country_name, get_continent_name

logger = logging.getLogger(__name__)

# Medal colors (RGB)
MEDAL_COLORS = {
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "bronze": (205, 127, 50),
}

# Ribbon colors for each medal
RIBBON_COLORS = {
    "gold": (220, 20, 60),      # Crimson red
    "silver": (70, 130, 180),   # Steel blue
    "bronze": (34, 139, 34),    # Forest green
}

# Cache for medal icon bytes
_medal_icon_cache: Dict[str, bytes] = {}


def get_medal_icon(medal_type: str, size: int = 64) -> bytes:
    """Generate a medal icon as PNG bytes (emoji-style with ribbon).

    Args:
        medal_type: 'gold', 'silver', or 'bronze'
        size: Icon size in pixels

    Returns:
        PNG image bytes
    """
    cache_key = f"{medal_type}_{size}"
    if cache_key in _medal_icon_cache:
        return _medal_icon_cache[cache_key]

    color = MEDAL_COLORS.get(medal_type, (128, 128, 128))
    ribbon_color = RIBBON_COLORS.get(medal_type, (128, 128, 128))

    # Create image with transparency
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Dimensions
    ribbon_width = size // 3
    ribbon_top = 0
    ribbon_bottom = size // 3
    medal_radius = size // 3
    medal_center_x = size // 2
    medal_center_y = size * 2 // 3

    # Draw ribbon (two stripes forming a V)
    ribbon_dark = tuple(max(0, c - 30) for c in ribbon_color)
    # Left ribbon stripe
    draw.polygon([
        (size // 2 - 2, ribbon_top),
        (size // 2 - ribbon_width // 2, ribbon_top),
        (medal_center_x - medal_radius // 2, medal_center_y - medal_radius // 2),
        (medal_center_x, medal_center_y - medal_radius // 3),
    ], fill=ribbon_color)
    # Right ribbon stripe
    draw.polygon([
        (size // 2 + 2, ribbon_top),
        (size // 2 + ribbon_width // 2, ribbon_top),
        (medal_center_x + medal_radius // 2, medal_center_y - medal_radius // 2),
        (medal_center_x, medal_center_y - medal_radius // 3),
    ], fill=ribbon_dark)

    # Draw medal (circle with border and shine)
    medal_left = medal_center_x - medal_radius
    medal_top = medal_center_y - medal_radius
    medal_right = medal_center_x + medal_radius
    medal_bottom = medal_center_y + medal_radius

    # Outer edge (darker)
    darker = tuple(max(0, c - 50) for c in color)
    draw.ellipse([medal_left, medal_top, medal_right, medal_bottom], fill=darker)

    # Main medal body
    border = size // 16
    draw.ellipse([
        medal_left + border, medal_top + border,
        medal_right - border, medal_bottom - border
    ], fill=color)

    # Inner circle detail
    inner_border = size // 8
    inner_darker = tuple(max(0, c - 25) for c in color)
    draw.ellipse([
        medal_left + inner_border, medal_top + inner_border,
        medal_right - inner_border, medal_bottom - inner_border
    ], fill=inner_darker, outline=darker, width=1)

    # Highlight (shine effect)
    lighter = tuple(min(255, c + 80) for c in color)
    highlight_size = medal_radius // 2
    highlight_x = medal_center_x - medal_radius // 3
    highlight_y = medal_center_y - medal_radius // 3
    draw.ellipse([
        highlight_x, highlight_y,
        highlight_x + highlight_size, highlight_y + highlight_size
    ], fill=lighter)

    # Number in center (1, 2, 3)
    number = {"gold": "1", "silver": "2", "bronze": "3"}.get(medal_type, "")
    if number:
        # Draw number - use a simple approach since we can't guarantee font availability
        num_color = tuple(max(0, c - 80) for c in color)
        # Draw a thicker number by drawing multiple offset copies
        num_size = medal_radius // 2
        num_x = medal_center_x
        num_y = medal_center_y + size // 32  # Slight offset down

        # Simple "1", "2", "3" shapes
        if number == "1":
            draw.line([(num_x, num_y - num_size//2), (num_x, num_y + num_size//2)], fill=num_color, width=max(2, size//16))
        elif number == "2":
            # Simplified 2
            draw.arc([num_x - num_size//2, num_y - num_size//2, num_x + num_size//2, num_y], 180, 0, fill=num_color, width=max(2, size//16))
            draw.line([(num_x + num_size//2, num_y), (num_x - num_size//2, num_y + num_size//2)], fill=num_color, width=max(2, size//16))
            draw.line([(num_x - num_size//2, num_y + num_size//2), (num_x + num_size//2, num_y + num_size//2)], fill=num_color, width=max(2, size//16))
        elif number == "3":
            # Simplified 3
            draw.arc([num_x - num_size//2, num_y - num_size//2, num_x + num_size//2, num_y + num_size//6], 270, 90, fill=num_color, width=max(2, size//16))
            draw.arc([num_x - num_size//2, num_y - num_size//6, num_x + num_size//2, num_y + num_size//2], 270, 90, fill=num_color, width=max(2, size//16))

    # Convert to bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    icon_bytes = buf.getvalue()

    _medal_icon_cache[cache_key] = icon_bytes
    return icon_bytes


def format_date_us(date_str: str) -> str:
    """Format date as mm/dd/yy."""
    if not date_str:
        return "-"
    try:
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(date_str)
        return dt.strftime("%m/%d/%y")
    except (ValueError, TypeError):
        return date_str[:10] if len(date_str) >= 10 else date_str


def format_datetime_us(datetime_str: str) -> str:
    """Format datetime as mm/dd/yy hh:mm am/pm."""
    if not datetime_str:
        return "-"
    try:
        if "T" in datetime_str:
            dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(datetime_str)
        return dt.strftime("%m/%d/%y %I:%M %p")
    except (ValueError, TypeError):
        return datetime_str[:16] if len(datetime_str) >= 16 else datetime_str


def format_header_datetime() -> str:
    """Format current UTC datetime for header as mm/dd/yy hh:mm am/pm."""
    return datetime.utcnow().strftime("%m/%d/%y %I:%M %p") + " UTC"


# PDF cache directory - uses same base as database for persistent storage
def _get_pdf_cache_dir() -> Path:
    """Get the PDF cache directory, creating it if necessary."""
    db_path = os.getenv("DATABASE_PATH", "ham_olympics.db")
    cache_dir = Path(db_path).parent / "pdf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_pdf_path(olympiad_id: int) -> Path:
    """Get the path for a cached olympiad PDF."""
    return _get_pdf_cache_dir() / f"olympiad_{olympiad_id}.pdf"


def get_cached_pdf(olympiad_id: int) -> Optional[bytes]:
    """
    Get the cached PDF for an olympiad if it exists.

    Returns:
        PDF bytes if cached file exists, None otherwise
    """
    pdf_path = get_cached_pdf_path(olympiad_id)
    if pdf_path.exists():
        return pdf_path.read_bytes()
    return None


def get_cached_pdf_info(olympiad_id: int) -> Optional[Dict[str, Any]]:
    """
    Get info about the cached PDF.

    Returns:
        Dict with 'exists', 'path', 'generated_at', 'size_bytes' or None
    """
    pdf_path = get_cached_pdf_path(olympiad_id)
    if pdf_path.exists():
        stat = pdf_path.stat()
        return {
            "exists": True,
            "path": str(pdf_path),
            "generated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size_bytes": stat.st_size
        }
    return {"exists": False, "path": str(pdf_path), "generated_at": None, "size_bytes": 0}


def regenerate_cached_pdf(olympiad_id: int) -> bool:
    """
    Regenerate the cached PDF for an olympiad.

    This should be called after medals or records are updated.

    Returns:
        True if PDF was generated successfully, False otherwise
    """
    try:
        pdf_bytes = generate_cached_olympiad_pdf(olympiad_id)
        pdf_path = get_cached_pdf_path(olympiad_id)
        pdf_path.write_bytes(pdf_bytes)
        logger.info(f"Regenerated cached PDF for olympiad {olympiad_id}: {len(pdf_bytes)} bytes")
        return True
    except Exception as e:
        logger.error(f"Failed to regenerate cached PDF for olympiad {olympiad_id}: {e}")
        return False


def regenerate_active_olympiad_pdf() -> bool:
    """
    Regenerate the cached PDF for the active olympiad.

    This is the main entry point called after medal/record updates.

    Returns:
        True if PDF was generated successfully, False if no active olympiad or error
    """
    with get_db() as conn:
        cursor = conn.execute("SELECT id FROM olympiads WHERE is_active = 1")
        row = cursor.fetchone()
        if not row:
            logger.debug("No active olympiad found, skipping PDF regeneration")
            return False

    return regenerate_cached_pdf(row["id"])


class OlympicsPDF(FPDF):
    """Custom PDF class with Ham Radio Olympics branding."""

    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.set_auto_page_break(auto=True, margin=15)
        self._has_unicode_font = False
        # Try to add Unicode font support for medal icons (Linux path)
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux/Docker
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    self.add_font("DejaVu", "", font_path)
                    self._has_unicode_font = True
                    break
                except Exception:
                    pass

    def _use_unicode_font(self):
        """Try to use DejaVu font for Unicode, fall back to Helvetica."""
        if self._has_unicode_font:
            try:
                self.set_font("DejaVu", "", 9)
                return True
            except Exception:
                pass
        self.set_font("Helvetica", "", 9)
        return False

    def header(self):
        """Add page header with title and branding."""
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "Ham Radio Olympics", border=0, align="L")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 10, format_header_datetime(), border=0, align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        """Add page footer with page number."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"{self.page_no()} of {{nb}}", border=0, align="C")

    def add_cover_page(self, olympiad_name: str, start_date: str, end_date: str):
        """Add cover page with olympiad info."""
        self.add_page()
        self.set_font("Helvetica", "B", 24)
        self.ln(40)
        self.cell(0, 20, self.title, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(10)
        self.set_font("Helvetica", "", 16)
        self.cell(0, 10, olympiad_name, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 12)
        date_range = f"{format_date_us(start_date)} to {format_date_us(end_date)}"
        self.cell(0, 10, date_range, align="C", new_x="LMARGIN", new_y="NEXT")

    def add_section_header(self, title: str):
        """Add a section header."""
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def add_subsection_header(self, title: str):
        """Add a subsection header."""
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def add_table(self, headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None, col_aligns: Optional[List[str]] = None):
        """Add a table with headers and data rows.

        Args:
            headers: Column header names
            rows: List of row data
            col_widths: Optional list of column widths
            col_aligns: Optional list of alignments per column ("L", "C", "R")
        """
        if not rows:
            return

        if col_widths is None:
            page_width = 190
            col_widths = [page_width // len(headers)] * len(headers)

        if col_aligns is None:
            col_aligns = ["C"] * len(headers)

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(200, 200, 200)
        for header, width in zip(headers, col_widths):
            self.cell(width, 7, str(header), border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 9)
        for i, row in enumerate(rows):
            if i % 2 == 0:
                self.set_fill_color(255, 255, 255)
            else:
                self.set_fill_color(248, 248, 248)

            for cell, width, align in zip(row, col_widths, col_aligns):
                self.cell(width, 6, str(cell)[:40], border=1, fill=True, align=align)
            self.ln()

    def add_podium_table(self, qso_race_medals: Dict[str, Dict], cool_factor_medals: Dict[str, Dict]):
        """Add podium table with medal icons.

        Args:
            qso_race_medals: Dict mapping medal type to winner info
            cool_factor_medals: Dict mapping medal type to winner info
        """
        row_height = 7
        icon_size = 5  # mm

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(200, 200, 200)
        col_widths = [15, 55, 55]  # Medal icon, QSO Race, Cool Factor
        headers = ["", "QSO Race", "Cool Factor"]
        for header, width in zip(headers, col_widths):
            self.cell(width, row_height, header, border=1, fill=True, align="C")
        self.ln()

        # Data rows with medal icons
        self.set_font("Helvetica", "", 9)
        for i, medal_type in enumerate(["gold", "silver", "bronze"]):
            if i % 2 == 0:
                self.set_fill_color(255, 255, 255)
            else:
                self.set_fill_color(248, 248, 248)

            # Medal icon cell
            x_start = self.get_x()
            y_start = self.get_y()
            self.cell(col_widths[0], row_height, "", border=1, fill=True)

            # Draw medal icon centered in first cell
            icon_bytes = get_medal_icon(medal_type)
            icon_x = x_start + (col_widths[0] - icon_size) / 2
            icon_y = y_start + (row_height - icon_size) / 2
            self.image(io.BytesIO(icon_bytes), x=icon_x, y=icon_y, w=icon_size, h=icon_size)

            # QSO Race winner
            qso = qso_race_medals.get(medal_type, {})
            qso_call = qso.get("callsign", "-")
            self.cell(col_widths[1], row_height, qso_call, border=1, fill=True, align="C")

            # Cool Factor winner
            cf = cool_factor_medals.get(medal_type, {})
            cf_call = cf.get("callsign", "-")
            self.cell(col_widths[2], row_height, cf_call, border=1, fill=True, align="C")

            self.ln()

    def _draw_medal_icons_in_cell(self, x: float, y: float, width: float, height: float,
                                     gold: int, silver: int, bronze: int, icon_size: float = 4):
        """Draw medal icons within a cell area.

        Args:
            x, y: Cell top-left position
            width, height: Cell dimensions
            gold, silver, bronze: Count of each medal type
            icon_size: Size of each icon in mm
        """
        total_medals = gold + silver + bronze
        if total_medals == 0:
            return

        # Calculate spacing - center the medals in the cell
        spacing = 1  # mm between icons
        total_width = total_medals * icon_size + (total_medals - 1) * spacing
        start_x = x + (width - total_width) / 2
        icon_y = y + (height - icon_size) / 2

        current_x = start_x
        for medal_type, count in [("gold", gold), ("silver", silver), ("bronze", bronze)]:
            for _ in range(count):
                icon_bytes = get_medal_icon(medal_type)
                self.image(io.BytesIO(icon_bytes), x=current_x, y=icon_y, w=icon_size, h=icon_size)
                current_x += icon_size + spacing

    def add_standings_table_with_medals(self, standings: List[Dict], show_rank: bool = True):
        """Add standings table with medal icons.

        Args:
            standings: List of standing entries with callsign, total_points, medals info
            show_rank: Whether to show rank column
        """
        if not standings:
            return

        row_height = 6
        icon_size = 4  # mm

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(200, 200, 200)
        headers = ["#", "Callsign", "Points", "QSO Race Medals", "Cool Factor Medals"]
        col_widths = [12, 40, 25, 55, 55]
        for header, width in zip(headers, col_widths):
            self.cell(width, 7, header, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 9)
        for i, s in enumerate(standings):
            rank = i + 1
            if i % 2 == 0:
                self.set_fill_color(255, 255, 255)
            else:
                self.set_fill_color(248, 248, 248)

            # Rank
            self.cell(col_widths[0], row_height, str(rank), border=1, fill=True, align="C")

            # Callsign
            self.cell(col_widths[1], row_height, s["callsign"], border=1, fill=True, align="C")

            # Points
            self.cell(col_widths[2], row_height, str(s["total_points"]), border=1, fill=True, align="C")

            # QSO Race medals - draw cell then add icons
            x_race = self.get_x()
            y_race = self.get_y()
            self.cell(col_widths[3], row_height, "", border=1, fill=True)
            self._draw_medal_icons_in_cell(x_race, y_race, col_widths[3], row_height,
                                           s['gold_count'], s['silver_count'], s['bronze_count'], icon_size)

            # Cool Factor medals - draw cell then add icons
            x_cf = self.get_x()
            y_cf = self.get_y()
            self.cell(col_widths[4], row_height, "", border=1, fill=True)
            self._draw_medal_icons_in_cell(x_cf, y_cf, col_widths[4], row_height,
                                           s['cf_gold'], s['cf_silver'], s['cf_bronze'], icon_size)

            self.ln()

    def add_match_standings_table(self, leaderboard: List[Dict]):
        """Add match standings table with medal icons in Race and CF columns.

        Args:
            leaderboard: List of entries with callsign, qso_race_medal, cool_factor_medal, etc.
        """
        if not leaderboard:
            return

        row_height = 6
        icon_size = 4  # mm

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(200, 200, 200)
        headers = ["#", "Callsign", "Race", "CF", "POTA", "Pts"]
        col_widths = [15, 45, 25, 25, 25, 25]
        for header, width in zip(headers, col_widths):
            self.cell(width, 7, header, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 9)
        for i, entry in enumerate(leaderboard):
            rank = i + 1
            if i % 2 == 0:
                self.set_fill_color(255, 255, 255)
            else:
                self.set_fill_color(248, 248, 248)

            # Rank
            self.cell(col_widths[0], row_height, str(rank), border=1, fill=True, align="C")

            # Callsign
            self.cell(col_widths[1], row_height, entry["callsign"], border=1, fill=True, align="C")

            # Race medal - draw cell then add icon if applicable
            x_race = self.get_x()
            y_race = self.get_y()
            self.cell(col_widths[2], row_height, "", border=1, fill=True)
            if entry.get("qso_race_medal"):
                icon_bytes = get_medal_icon(entry["qso_race_medal"])
                icon_x = x_race + (col_widths[2] - icon_size) / 2
                icon_y = y_race + (row_height - icon_size) / 2
                self.image(io.BytesIO(icon_bytes), x=icon_x, y=icon_y, w=icon_size, h=icon_size)

            # CF medal - draw cell then add icon if applicable
            x_cf = self.get_x()
            y_cf = self.get_y()
            self.cell(col_widths[3], row_height, "", border=1, fill=True)
            if entry.get("cool_factor_medal"):
                icon_bytes = get_medal_icon(entry["cool_factor_medal"])
                icon_x = x_cf + (col_widths[3] - icon_size) / 2
                icon_y = y_cf + (row_height - icon_size) / 2
                self.image(io.BytesIO(icon_bytes), x=icon_x, y=icon_y, w=icon_size, h=icon_size)

            # POTA
            pota = "+1" if entry.get("pota_bonus") else "-"
            self.cell(col_widths[4], row_height, pota, border=1, fill=True, align="C")

            # Points
            self.cell(col_widths[5], row_height, str(entry["total_points"]), border=1, fill=True, align="C")

            self.ln()

    def add_matches_table(self, matches: List[Dict], target_type: str, matches_with_qsos: set):
        """Add matches table with bold targets for matches that have QSOs.

        Args:
            matches: List of match dicts with target_value, start_date, end_date
            target_type: The target type for formatting
            matches_with_qsos: Set of match IDs that have QSOs
        """
        if not matches:
            return

        row_height = 6
        col_widths = [90, 50, 50]

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(200, 200, 200)
        for header, width in zip(["Target", "Start", "End"], col_widths):
            self.cell(width, 7, header, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        for i, match in enumerate(matches):
            if i % 2 == 0:
                self.set_fill_color(255, 255, 255)
            else:
                self.set_fill_color(248, 248, 248)

            has_qsos = match["id"] in matches_with_qsos
            target_display = format_target(target_type, match["target_value"])

            # Target - bold if has QSOs
            if has_qsos:
                self.set_font("Helvetica", "B", 9)
            else:
                self.set_font("Helvetica", "", 9)
            self.cell(col_widths[0], row_height, target_display[:50], border=1, fill=True, align="L")

            # Dates - always normal weight
            self.set_font("Helvetica", "", 9)
            self.cell(col_widths[1], row_height, format_date_us(match["start_date"]), border=1, fill=True, align="C")
            self.cell(col_widths[2], row_height, format_date_us(match["end_date"]), border=1, fill=True, align="C")
            self.ln()

    def add_legend(self):
        """Add legend explaining scoring rules."""
        self.add_page()
        self.add_section_header("Legend & Scoring Rules")
        self.ln(5)

        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, "Medal Points:", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, "  Gold = 3 points  |  Silver = 2 points  |  Bronze = 1 point", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, "POTA Bonus:", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, "  +1 point for valid POTA activation or working POTA station", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, "Cool Factor Formula:", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, "  Cool Factor = Distance (km) / TX Power (watts)", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5, "  Higher is better - rewards efficiency", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, "Maximum Points Per Match:", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, "  Single mode: 7 points (3+3+1 POTA)", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5, "  Dual mode (separate pools): 14 points (7 per role)", new_x="LMARGIN", new_y="NEXT")


def get_olympiad_data(olympiad_id: int) -> Optional[Dict[str, Any]]:
    """Get olympiad data including sports and matches."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT id, name, start_date, end_date, qualifying_qsos
            FROM olympiads WHERE id = ?
        """, (olympiad_id,))
        olympiad = cursor.fetchone()
        if not olympiad:
            return None

        cursor = conn.execute("""
            SELECT id, name, description, target_type, work_enabled, activate_enabled, separate_pools
            FROM sports WHERE olympiad_id = ? ORDER BY name
        """, (olympiad_id,))
        sports = [dict(row) for row in cursor.fetchall()]

        return {
            "olympiad": dict(olympiad),
            "sports": sports
        }


def get_sport_data(sport_id: int, top_n: int = 10) -> Optional[Dict[str, Any]]:
    """Get sport data including matches and standings."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT s.id, s.name, s.description, s.target_type, s.work_enabled,
                   s.activate_enabled, s.separate_pools, o.name as olympiad_name
            FROM sports s
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE s.id = ?
        """, (sport_id,))
        sport = cursor.fetchone()
        if not sport:
            return None

        cursor = conn.execute("""
            SELECT id, start_date, end_date, target_value
            FROM matches WHERE sport_id = ? ORDER BY start_date
        """, (sport_id,))
        matches = [dict(row) for row in cursor.fetchall()]

        cursor = conn.execute("""
            SELECT m.callsign,
                   SUM(m.total_points) as total_points,
                   SUM(CASE WHEN m.qso_race_medal = 'gold' THEN 1 ELSE 0 END) as gold_count,
                   SUM(CASE WHEN m.qso_race_medal = 'silver' THEN 1 ELSE 0 END) as silver_count,
                   SUM(CASE WHEN m.qso_race_medal = 'bronze' THEN 1 ELSE 0 END) as bronze_count,
                   SUM(CASE WHEN m.cool_factor_medal = 'gold' THEN 1 ELSE 0 END) as cf_gold,
                   SUM(CASE WHEN m.cool_factor_medal = 'silver' THEN 1 ELSE 0 END) as cf_silver,
                   SUM(CASE WHEN m.cool_factor_medal = 'bronze' THEN 1 ELSE 0 END) as cf_bronze
            FROM medals m
            JOIN matches ma ON m.match_id = ma.id
            WHERE ma.sport_id = ?
            GROUP BY m.callsign
            ORDER BY total_points DESC
            LIMIT ?
        """, (sport_id, top_n))
        standings = [dict(row) for row in cursor.fetchall()]

        return {
            "sport": dict(sport),
            "matches": matches,
            "standings": standings
        }


def get_match_data(match_id: int, top_n: int = 10) -> Optional[Dict[str, Any]]:
    """Get match data including leaderboard."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT m.id, m.start_date, m.end_date, m.target_value,
                   s.name as sport_name, s.target_type
            FROM matches m
            JOIN sports s ON m.sport_id = s.id
            WHERE m.id = ?
        """, (match_id,))
        match = cursor.fetchone()
        if not match:
            return None

        cursor = conn.execute("""
            SELECT callsign, role, qso_race_medal, cool_factor_medal,
                   total_points, qso_race_claim_time, cool_factor_value, pota_bonus
            FROM medals
            WHERE match_id = ? AND total_points > 0
            ORDER BY total_points DESC, qso_race_claim_time ASC
            LIMIT ?
        """, (match_id, top_n))
        leaderboard = [dict(row) for row in cursor.fetchall()]

        # Get QSO race medal holders
        qso_race_medals = {}
        for medal_type in ['gold', 'silver', 'bronze']:
            cursor = conn.execute("""
                SELECT callsign, qso_race_claim_time
                FROM medals
                WHERE match_id = ? AND qso_race_medal = ?
                LIMIT 1
            """, (match_id, medal_type))
            row = cursor.fetchone()
            if row:
                qso_race_medals[medal_type] = dict(row)

        # Get cool factor medal holders
        cool_factor_medals = {}
        for medal_type in ['gold', 'silver', 'bronze']:
            cursor = conn.execute("""
                SELECT callsign, cool_factor_value
                FROM medals
                WHERE match_id = ? AND cool_factor_medal = ?
                LIMIT 1
            """, (match_id, medal_type))
            row = cursor.fetchone()
            if row:
                cool_factor_medals[medal_type] = dict(row)

        return {
            "match": dict(match),
            "leaderboard": leaderboard,
            "qso_race_medals": qso_race_medals,
            "cool_factor_medals": cool_factor_medals
        }


def get_records_data_detailed() -> Dict[str, Any]:
    """Get detailed world records with QSO info."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT r.record_type, r.value, r.callsign, r.achieved_at, r.qso_id,
                   s.name as sport_name, m.target_value,
                   q.dx_callsign, q.mode, q.tx_power_w, q.distance_km, q.cool_factor
            FROM records r
            LEFT JOIN sports s ON r.sport_id = s.id
            LEFT JOIN matches m ON r.match_id = m.id
            LEFT JOIN qsos q ON r.qso_id = q.id
            WHERE r.callsign IS NOT NULL
            ORDER BY r.record_type, r.value DESC
        """)
        return {"world_records": [dict(row) for row in cursor.fetchall()]}


def get_records_data(sport_id: Optional[int] = None, callsign: Optional[str] = None) -> Dict[str, Any]:
    """Get world records and optionally personal bests."""
    with get_db() as conn:
        result = {"world_records": [], "personal_bests": []}

        if sport_id:
            cursor = conn.execute("""
                SELECT record_type, value, callsign, achieved_at
                FROM records
                WHERE sport_id = ? AND callsign IS NOT NULL
                ORDER BY record_type
            """, (sport_id,))
        else:
            cursor = conn.execute("""
                SELECT r.record_type, r.value, r.callsign, r.achieved_at, s.name as sport_name
                FROM records r
                LEFT JOIN sports s ON r.sport_id = s.id
                WHERE r.callsign IS NOT NULL
                ORDER BY r.record_type, r.value DESC
            """)
        result["world_records"] = [dict(row) for row in cursor.fetchall()]

        if callsign:
            cursor = conn.execute("""
                SELECT record_type, value, achieved_at, sport_id
                FROM records
                WHERE callsign = ?
                ORDER BY record_type
            """, (callsign,))
            result["personal_bests"] = [dict(row) for row in cursor.fetchall()]

        return result


def get_park_name(reference: str) -> Optional[str]:
    """Look up park name from pota_parks table, fetching from API if needed."""
    import httpx

    reference = reference.upper().strip()

    # Check cache first
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT name FROM pota_parks WHERE reference = ?",
            (reference,)
        )
        row = cursor.fetchone()
        if row:
            return row["name"]

    # Fetch from POTA API if not cached
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"https://api.pota.app/park/{reference}")
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, dict) and data.get("name"):
                    name = data["name"]
                    location = data.get("locationDesc", "")
                    grid = data.get("grid", "")

                    # Cache for future use
                    with get_db() as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO pota_parks (reference, name, location, grid, cached_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (reference, name, location, grid, datetime.utcnow().isoformat()))

                    return name
    except Exception as e:
        logger.warning(f"Failed to fetch park {reference} from POTA API: {e}")

    return None


def format_target(target_type: str, target_value: str) -> str:
    """Format target value for display with codes."""
    if target_type == "continent":
        name = get_continent_name(target_value)
        if name:
            return f"{target_value} - {name}"
        return target_value
    elif target_type == "country":
        if target_value.isdigit():
            name = get_country_name(int(target_value))
            if name:
                return f"{target_value} - {name}"
        return target_value
    elif target_type == "park" or target_type == "pota":
        name = get_park_name(target_value)
        if name:
            return f"{target_value} - {name}"
        return target_value
    elif target_type == "grid":
        return target_value
    else:
        return target_value


def format_medal_icon(medal: Optional[str]) -> str:
    """Format medal as icon text."""
    if medal == "gold":
        return "Gold"
    elif medal == "silver":
        return "Silver"
    elif medal == "bronze":
        return "Bronze"
    return "-"


def format_record_type(record_type: str) -> str:
    """Format record type for display."""
    return {
        "longest_distance": "Longest Distance",
        "highest_cool_factor": "Best Cool Factor",
        "lowest_power": "Lowest Power DX"
    }.get(record_type, record_type.replace("_", " ").title())


def build_world_records_section(pdf: OlympicsPDF, records: List[Dict[str, Any]]):
    """Build world records section with master/detail structure."""
    pdf.add_section_header("World Records")

    if not records:
        return

    for record in records:
        # Master row - record summary
        pdf.set_font("Helvetica", "B", 10)
        record_name = format_record_type(record["record_type"])
        pdf.cell(60, 7, record_name, border=0)

        pdf.set_font("Helvetica", "", 10)
        value_str = f"{record['value']:.1f}"
        if "distance" in record["record_type"]:
            value_str += " km"
        elif "cool_factor" in record["record_type"]:
            value_str += " km/W"
        elif "power" in record["record_type"]:
            value_str += " W"
        pdf.cell(40, 7, value_str, border=0)
        pdf.cell(40, 7, record["callsign"] or "-", border=0)
        pdf.cell(0, 7, format_datetime_us(record.get("achieved_at", "")), border=0, new_x="LMARGIN", new_y="NEXT")

        # Detail rows - QSO info
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 100, 100)

        details = []
        if record.get("dx_callsign"):
            details.append(f"QSO with: {record['dx_callsign']}")
        if record.get("mode"):
            details.append(f"Mode: {record['mode']}")
        if record.get("tx_power_w"):
            details.append(f"Power: {record['tx_power_w']}W")
        if record.get("distance_km"):
            details.append(f"Distance: {record['distance_km']:.0f} km")
        if record.get("sport_name"):
            target = record.get("target_value", "")
            details.append(f"Match: {record['sport_name']}: {target}")

        if details:
            pdf.cell(10, 5, "", border=0)  # indent
            pdf.cell(0, 5, "  |  ".join(details), border=0, new_x="LMARGIN", new_y="NEXT")

        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)


def build_sport_section(pdf: OlympicsPDF, sport_data: Dict[str, Any], include_qsos: bool, top_n: int):
    """Build sport section in PDF."""
    sport = sport_data["sport"]

    pdf.add_page()
    pdf.add_section_header(f"Sport: {sport['name']}")

    # Sport info
    pdf.set_font("Helvetica", "", 10)
    target_type = sport["target_type"].title()
    modes = []
    if sport["work_enabled"]:
        modes.append("Work")
    if sport["activate_enabled"]:
        modes.append("Activate")
    mode_str = " & ".join(modes) if modes else "N/A"
    pdf.cell(0, 6, f"Target Type: {target_type}  |  Modes: {mode_str}", new_x="LMARGIN", new_y="NEXT")
    if sport.get("description"):
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, sport["description"], new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
    pdf.ln(5)

    # Overall standings with medal icons for top 3
    if sport_data["standings"]:
        pdf.add_subsection_header(f"Overall Standings (Top {len(sport_data['standings'])})")
        pdf.add_standings_table_with_medals(sport_data["standings"])
        pdf.ln(5)

    # Pre-fetch match data to determine which have QSOs
    match_data_cache = {}
    matches_with_qsos = set()
    for match in sport_data["matches"]:
        match_data = get_match_data(match["id"], top_n)
        match_data_cache[match["id"]] = match_data
        if match_data and match_data["leaderboard"]:
            matches_with_qsos.add(match["id"])

    # Matches table with bold targets for those with QSOs
    if sport_data["matches"]:
        pdf.add_subsection_header("Matches")
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 4, "Bold targets have QSOs logged.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)
        pdf.add_matches_table(sport_data["matches"], sport["target_type"], matches_with_qsos)
        pdf.ln(5)

    # Match details with leaderboards
    for match in sport_data["matches"]:
        match_data = match_data_cache.get(match["id"])
        if match_data and match_data["leaderboard"]:
            build_match_section(pdf, match_data, sport["target_type"])


def build_match_section(pdf: OlympicsPDF, match_data: Dict[str, Any], target_type: str):
    """Build match section in PDF with master/detail structure."""
    match = match_data["match"]
    leaderboard = match_data["leaderboard"]

    if not leaderboard:
        return  # Skip matches with no medals

    target_display = format_target(target_type, match["target_value"])
    date_range = f"{format_date_us(match['start_date'])} - {format_date_us(match['end_date'])}"

    pdf.add_subsection_header(f"{target_display} ({date_range})")

    # Podium table showing QSO Race and Cool Factor winners with medal icons
    qso_medals = match_data.get("qso_race_medals", {})
    cf_medals = match_data.get("cool_factor_medals", {})

    if qso_medals or cf_medals:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Podium", new_x="LMARGIN", new_y="NEXT")
        pdf.add_podium_table(qso_medals, cf_medals)
        pdf.ln(3)

    # Standings table with medal icons
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Standings", new_x="LMARGIN", new_y="NEXT")
    pdf.add_match_standings_table(leaderboard)
    pdf.ln(5)


def generate_cached_olympiad_pdf(olympiad_id: int, top_n: int = 20) -> bytes:
    """
    Generate a cached (non-user-specific) PDF for an olympiad.
    """
    data = get_olympiad_data(olympiad_id)
    if not data:
        raise ValueError(f"Olympiad {olympiad_id} not found")

    olympiad = data["olympiad"]
    pdf = OlympicsPDF(title="Olympiad Standings")
    pdf.alias_nb_pages()

    # Cover page
    pdf.add_cover_page(olympiad["name"], olympiad["start_date"], olympiad["end_date"])

    # World records section with detail
    records = get_records_data_detailed()
    if records["world_records"]:
        pdf.add_page()
        build_world_records_section(pdf, records["world_records"])

    # Sports sections
    for sport in data["sports"]:
        sport_data = get_sport_data(sport["id"], top_n)
        if sport_data:
            build_sport_section(pdf, sport_data, include_qsos=True, top_n=top_n)

    # Legend
    pdf.add_legend()

    return bytes(pdf.output())


def generate_olympiad_pdf(
    olympiad_id: int,
    callsign: str,
    top_n: int = 10,
    include_qsos: bool = False,
    include_records: bool = False
) -> bytes:
    """Generate PDF for entire olympiad."""
    data = get_olympiad_data(olympiad_id)
    if not data:
        raise ValueError(f"Olympiad {olympiad_id} not found")

    olympiad = data["olympiad"]
    pdf = OlympicsPDF(title="Olympiad Report")
    pdf.alias_nb_pages()

    # Cover page
    pdf.add_cover_page(olympiad["name"], olympiad["start_date"], olympiad["end_date"])

    # Records section
    if include_records:
        records = get_records_data_detailed()
        if records["world_records"]:
            pdf.add_page()
            build_world_records_section(pdf, records["world_records"])

    # Sports sections
    for sport in data["sports"]:
        sport_data = get_sport_data(sport["id"], top_n)
        if sport_data:
            build_sport_section(pdf, sport_data, include_qsos, top_n)

    # Legend
    pdf.add_legend()

    return bytes(pdf.output())


def generate_sport_pdf(
    sport_id: int,
    callsign: str,
    top_n: int = 10,
    include_qsos: bool = False,
    include_records: bool = False
) -> bytes:
    """Generate PDF for single sport."""
    sport_data = get_sport_data(sport_id, top_n)
    if not sport_data:
        raise ValueError(f"Sport {sport_id} not found")

    sport = sport_data["sport"]
    pdf = OlympicsPDF(title=f"{sport['name']} Report")
    pdf.alias_nb_pages()

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(30)
    pdf.cell(0, 15, sport["name"], align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, sport.get("olympiad_name", ""), align="C", new_x="LMARGIN", new_y="NEXT")

    # Records section
    if include_records:
        records = get_records_data(sport_id=sport_id, callsign=callsign)
        if records["world_records"] or records["personal_bests"]:
            pdf.add_page()
            build_records_section(pdf, records)

    # Sport details
    build_sport_section(pdf, sport_data, include_qsos, top_n)

    # Legend
    pdf.add_legend()

    return bytes(pdf.output())


def build_records_section(pdf: OlympicsPDF, records: Dict[str, Any]):
    """Build simple records section in PDF."""
    pdf.add_section_header("Records")

    if records["world_records"]:
        pdf.add_subsection_header("World Records")
        headers = ["Record Type", "Value", "Holder", "Date"]
        rows = []
        for r in records["world_records"]:
            rows.append([
                format_record_type(r["record_type"]),
                f"{r['value']:.1f}",
                r["callsign"] or "N/A",
                format_date_us(r.get("achieved_at", ""))
            ])
        pdf.add_table(headers, rows, [60, 40, 45, 45], ["L", "C", "C", "C"])
        pdf.ln(5)

    if records.get("personal_bests"):
        pdf.add_subsection_header("Personal Bests")
        headers = ["Record Type", "Value", "Date"]
        rows = []
        for r in records["personal_bests"]:
            rows.append([
                format_record_type(r["record_type"]),
                f"{r['value']:.1f}",
                format_date_us(r.get("achieved_at", ""))
            ])
        pdf.add_table(headers, rows, [70, 60, 60], ["L", "C", "C"])
        pdf.ln(5)


def generate_my_sports_pdf(
    callsign: str,
    top_n: int = 10,
    include_qsos: bool = False,
    include_records: bool = False
) -> bytes:
    """Generate PDF for competitor's entered sports."""
    with get_db() as conn:
        cursor = conn.execute("SELECT id, name, start_date, end_date FROM olympiads WHERE is_active = 1")
        olympiad = cursor.fetchone()

        cursor = conn.execute("""
            SELECT s.id, s.name
            FROM sport_entries se
            JOIN sports s ON se.sport_id = s.id
            WHERE se.callsign = ?
            ORDER BY s.name
        """, (callsign,))
        entries = cursor.fetchall()

    pdf = OlympicsPDF(title=f"My Sports - {callsign}")
    pdf.alias_nb_pages()

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(30)
    pdf.cell(0, 15, "My Sports Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, callsign, align="C", new_x="LMARGIN", new_y="NEXT")
    if olympiad:
        pdf.cell(0, 10, dict(olympiad)["name"], align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    sports_count = len(entries) if entries else 0
    pdf.cell(0, 10, f"Sports Entered: {sports_count}", align="C", new_x="LMARGIN", new_y="NEXT")

    # Records section
    if include_records:
        records = get_records_data(callsign=callsign)
        if records["world_records"] or records["personal_bests"]:
            pdf.add_page()
            build_records_section(pdf, records)

    # Sports sections
    if entries:
        for entry in entries:
            sport_data = get_sport_data(entry["id"], top_n)
            if sport_data:
                build_sport_section(pdf, sport_data, include_qsos, top_n)
    else:
        pdf.add_page()
        pdf.add_section_header("Entered Sports")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 10, "No sports entered yet.", new_x="LMARGIN", new_y="NEXT")

    # Legend
    pdf.add_legend()

    return bytes(pdf.output())

"""dark theme: shared colors, qt palette, and stylesheet."""
from pathlib import Path

from PyQt6 import QtGui, QtWidgets

ASSETS = Path(__file__).resolve().parent / "assets"

# ---- ui colors ----------------------------------------------------------------
BG = "#0e1014"          # window background (gaps between panels)
SURFACE = "#15181e"     # panels and plots
SURFACE_2 = "#1c2027"   # inputs, hover rows
SURFACE_3 = "#252a33"   # pressed / selected
BORDER = "#2a303a"
TEXT = "#e4e7ec"
MUTED = "#8a93a3"
FAINT = "#5b6472"
ACCENT = "#5aa9ff"
DANGER = "#ff6b6b"
WARN = "#f2c14e"

# ---- plot colors --------------------------------------------------------------
PLOT_BG = SURFACE
AXIS = "#3a414c"
AXIS_TEXT = "#9aa3b2"
ISOTHERM = "#262c35"
ISOTHERM_0 = "#2f5f8f"
ISOBAR = "#232831"
DRY_ADIABAT = "#4a3a2a"
MOIST_ADIABAT = "#25402f"
MIXING_RATIO = "#3f3a5c"
MIXING_LABEL = "#8f86c9"

T_LINE = "#ff5d5d"
TD_LINE = "#3fd47f"
ORIG = "#6b7482"
ORIG_ALPHA = 90      # 0-255; the unedited profile is drawn in its own colors at this opacity
PARCEL = "#d9dde4"
MARKER = "#ffffff"
LCL = "#7cc4ff"
LFC = "#f2c14e"
EL = "#c792ea"

HODO_RINGS = "#262c35"
HODO_BANDS = ["#ff5d5d", "#3fd47f", "#f2c14e", "#5aa9ff", "#c792ea"]
# comparison soundings, one color each; kept away from the T (red) and Td (green) lines
COMPARE = ["#5aa9ff", "#ffa94d", "#da77f2", "#66d9e8", "#ffd43b", "#f783ac"]


def apply(app):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    c = QtGui.QColor
    role = QtGui.QPalette.ColorRole
    for r, col in ((role.Window, BG), (role.WindowText, TEXT), (role.Base, SURFACE_2),
                   (role.AlternateBase, SURFACE), (role.Text, TEXT), (role.Button, SURFACE_2),
                   (role.ButtonText, TEXT), (role.ToolTipBase, SURFACE_3), (role.ToolTipText, TEXT),
                   (role.PlaceholderText, FAINT), (role.Highlight, "#2b4f78"),
                   (role.HighlightedText, TEXT), (role.Link, ACCENT), (role.Mid, BORDER),
                   (role.Dark, BG), (role.Shadow, "#000000"), (role.Light, SURFACE_3)):
        pal.setColor(r, c(col))
    grp = QtGui.QPalette.ColorGroup.Disabled
    for r in (role.Text, role.WindowText, role.ButtonText):
        pal.setColor(grp, r, c(FAINT))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)


STYLESHEET = f"""
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER};
            border-radius: 6px; padding: 6px 8px; }}

/* cards in the control panel */
QFrame#card {{ background: {SURFACE}; border-radius: 10px; }}
QLabel#sectionTitle {{ color: {MUTED}; font-size: 11px; font-weight: 600;
                       letter-spacing: 1px; padding-bottom: 2px; }}
QLabel#hint {{ color: {FAINT}; font-size: 11px; }}
QLabel {{ background: transparent; }}

/* inputs */
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
    background: {SURFACE_2}; color: {TEXT}; border: 1px solid transparent;
    border-radius: 6px; padding: 4px 8px; min-height: 18px;
    selection-background-color: #2b4f78; }}
QComboBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {BORDER}; }}
QComboBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox:disabled, QDoubleSpinBox:disabled {{ color: {FAINT}; background: {SURFACE}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url("{(ASSETS / 'chevron.svg').as_posix()}"); width: 10px; height: 6px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 4px; selection-background-color: {SURFACE_3}; outline: none; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0px; border: none; }}

QPushButton {{ background: {SURFACE_2}; color: {TEXT}; border: none; border-radius: 6px;
               padding: 6px 12px; }}
QPushButton:hover {{ background: {SURFACE_3}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton#primary {{ background: {ACCENT}; color: #0b1320; font-weight: 600; }}
QPushButton#primary:hover {{ background: #78b9ff; }}

QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator, QListWidget::indicator {{ width: 14px; height: 14px; border-radius: 4px;
                        border: 1px solid {FAINT}; background: {SURFACE_2}; }}
QCheckBox::indicator:checked, QListWidget::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT};
                                image: url("{(ASSETS / 'check.svg').as_posix()}"); }}

/* tabs: underline style, no frame */
QTabWidget::pane {{ border: none; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 7px 14px 6px 14px;
               border: none; border-bottom: 2px solid transparent; margin-right: 4px; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}

/* tables and lists */
QTreeWidget, QTableWidget, QListWidget {{ background: {SURFACE}; alternate-background-color: #171b21;
    border: none; outline: none; gridline-color: transparent; }}
QTreeWidget::item, QTableWidget::item {{ padding: 2px 4px; border: none; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: {SURFACE_2}; }}
QTreeWidget::item:selected, QTableWidget::item:selected {{ background: {SURFACE_3}; color: {TEXT}; }}
QListWidget {{ background: transparent; color: {MUTED}; }}
QListWidget::item {{ padding: 3px 0px; }}
QListWidget::item:selected {{ background: {SURFACE_3}; color: {TEXT}; }}
QHeaderView {{ background: {SURFACE}; }}
QHeaderView::section {{ background: {SURFACE}; color: {MUTED}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 6px 6px; font-size: 11px; font-weight: 600; }}
QTableCornerButton::section {{ background: {SURFACE}; border: none; }}

/* splitters act as gaps between panels */
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:vertical {{ height: 8px; }}

/* scrollbars */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {BORDER}; border-radius: 3px; min-height: 30px; min-width: 30px; }}
QScrollBar::handle:hover {{ background: {FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QDockWidget {{ border: none; }}
QStatusBar {{ background: {BG}; color: {MUTED}; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 8px; font-family: "SF Mono", "Menlo", "Consolas", monospace;
                     font-size: 11px; }}
QStatusBar::item {{ border: none; }}
QMenuBar {{ background: {BG}; color: {TEXT}; }}
QMenuBar::item:selected {{ background: {SURFACE_2}; border-radius: 4px; }}
QMenu {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {SURFACE_3}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
"""
STYLESHEET += f"""
QPushButton#ghost {{ background: transparent; color: {MUTED}; border: 1px solid {BORDER}; }}
QPushButton#ghost:hover {{ color: {TEXT}; border-color: {FAINT}; }}
"""

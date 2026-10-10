"""Design tokens and stylesheet for the TrachSplitter desktop UI.

Principles (HIG-informed):
- Follow the system light/dark appearance; no in-app appearance switch.
- One accent colour, reserved for primary actions, focus and selection.
- Colour in the content layer belongs to the stems: every stem keeps one
  colour everywhere (list, mixer, waveform). That is the app's signature.
- System UI font, 13 px body, 11 px minimum, weight carries hierarchy.
- Controls at least 28 px tall, visible keyboard focus, hover feedback.

Contrast (WCAG, computed):
  light  content #1D1D1F on #FFFFFF 16.8:1 · secondary #6E6E73 5.1:1
         white on accent #0066CC 5.6:1
  dark   content #F2F2F7 on #232326 14.1:1 · secondary #A1A1A6 6.1:1
         white on accent #0A6CD6 5.1:1
  stem dots >= 4.2:1 on their surface in both appearances (3:1 needed)
"""
import sys
from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap, QPalette, QGuiApplication


LIGHT = {
    'window': '#F2F2F5',      # window background behind cards
    'sidebar': '#E8E8ED',
    'surface': '#FFFFFF',     # cards, lists, inputs
    'surface_alt': '#F7F7FA',
    'hover': '#E5E5EA',
    'separator': '#D2D2D7',
    'content': '#1D1D1F',
    'secondary': '#6E6E73',
    'tertiary': '#AEAEB2',
    'accent': '#0066CC',
    'accent_hover': '#0077ED',
    'accent_pressed': '#00519E',
    'on_accent': '#FFFFFF',
    'accent_soft': '#D6E6FA',  # selection background
    'focus': '#4D9CF0',
    'danger': '#C4302B',
    'success': '#1E8E3E',
    'banner': '#E6F0FB',
    'plot_bg': '#FFFFFF',
    'plot_grid': '#D2D2D7',
}

DARK = {
    'window': '#1A1A1C',
    'sidebar': '#202023',
    'surface': '#232326',
    'surface_alt': '#2A2A2E',
    'hover': '#323236',
    'separator': '#3A3A3F',
    'content': '#F2F2F7',
    'secondary': '#A1A1A6',
    'tertiary': '#636366',
    'accent': '#0A6CD6',
    'accent_hover': '#1A7DE8',
    'accent_pressed': '#0857AD',
    'on_accent': '#FFFFFF',
    'accent_soft': '#16365C',
    'focus': '#4DA3FF',
    'danger': '#FF6961',
    'success': '#30D158',
    'banner': '#1C2E45',
    'plot_bg': '#1E1E21',
    'plot_grid': '#3A3A3F',
}

# Stem identity colours: (light, dark)
STEM_COLORS = {
    'vocals': ('#D6204E', '#FF6482'),
    'drums':  ('#C25E00', '#FF9F0A'),
    'bass':   ('#5E5CE6', '#8E8CFF'),
    'other':  ('#1E8E3E', '#30D158'),
    'guitar': ('#9A7300', '#FFD60A'),
    'piano':  ('#0A7FB0', '#64D2FF'),
}
_FALLBACK_STEM = ('#6E6E73', '#A1A1A6')


def system_is_dark() -> bool:
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    except Exception:
        pass
    pal = QGuiApplication.palette()
    return pal.color(QPalette.ColorRole.Window).lightness() < 128


def tokens(dark: bool) -> Dict[str, str]:
    return DARK if dark else LIGHT


def stem_color(name: str, dark: bool) -> str:
    base = name[3:] if name.startswith('no_') else name
    pair = STEM_COLORS.get(base, _FALLBACK_STEM)
    if name.startswith('no_'):
        return _FALLBACK_STEM[1 if dark else 0]  # "everything else" stays neutral
    return pair[1 if dark else 0]


def stem_label(name: str) -> str:
    if name.startswith('no_'):
        return f"Everything but {name[3:]}"
    return name.capitalize()


def dot_icon(color: str, size: int = 12) -> QIcon:
    """Filled circle used as the stem's identity mark."""
    pm = QPixmap(size * 2, size * 2)  # 2x for high-DPI
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pm)


def ui_font() -> QFont:
    if sys.platform == 'darwin':
        family = '.AppleSystemUIFont'
    elif sys.platform == 'win32':
        family = 'Segoe UI Variable Text'
    else:
        family = 'Inter'
    f = QFont(family)
    f.setPixelSize(13)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return f


def apply_palette(app, t: Dict[str, str]):
    """Native widgets (menus, dialogs, tooltips) pick up the same tokens."""
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(t['window']))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(t['content']))
    pal.setColor(QPalette.ColorRole.Base, QColor(t['surface']))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(t['surface_alt']))
    pal.setColor(QPalette.ColorRole.Text, QColor(t['content']))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(t['secondary']))
    pal.setColor(QPalette.ColorRole.Button, QColor(t['surface']))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(t['content']))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(t['accent']))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(t['on_accent']))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(t['surface']))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(t['content']))
    pal.setColor(QPalette.ColorRole.Link, QColor(t['accent']))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(t['tertiary']))
    app.setPalette(pal)


def _asset_dir():
    from pathlib import Path
    d = Path.home() / '.trachsplitter' / 'ui'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_assets(t: Dict[str, str]) -> Dict[str, str]:
    """Tiny SVG glyphs (chevrons, checkmark) tinted for the current theme.

    Qt stylesheets drop native arrows/checkmarks once a control is styled,
    so these keep the affordances visible.
    """
    tag = 'dark' if t is DARK else 'light'
    glyphs = {
        'chevron_down': f'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
                        f'<path d="M2 3.5 L5 6.5 L8 3.5" fill="none" stroke="{t["secondary"]}" '
                        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        'chevron_up': f'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
                      f'<path d="M2 6.5 L5 3.5 L8 6.5" fill="none" stroke="{t["secondary"]}" '
                      f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        'check': f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
                 f'<path d="M2.5 6.2 L5 8.6 L9.5 3.6" fill="none" stroke="{t["on_accent"]}" '
                 f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    }
    paths = {}
    d = _asset_dir()
    for name, svg in glyphs.items():
        p = d / f'{name}_{tag}.svg'
        try:
            if not p.exists() or p.read_text(encoding='utf-8') != svg:
                p.write_text(svg, encoding='utf-8')
        except OSError:
            pass
        paths[name] = p.as_posix()
    return paths


def stylesheet(t: Dict[str, str]) -> str:
    a = _write_assets(t)
    return f"""
    QComboBox::down-arrow {{ image: url("{a['chevron_down']}"); width: 10px; height: 10px; }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
        subcontrol-origin: border; width: 18px; border: none; background: transparent;
    }}
    QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
    QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {t['hover']}; }}
    QAbstractSpinBox::up-arrow {{ image: url("{a['chevron_up']}"); width: 8px; height: 8px; }}
    QAbstractSpinBox::down-arrow {{ image: url("{a['chevron_down']}"); width: 8px; height: 8px; }}
    QCheckBox::indicator:checked {{ image: url("{a['check']}"); }}
"""  + _base_stylesheet(t)


def _base_stylesheet(t: Dict[str, str]) -> str:
    return f"""
    QMainWindow, QDialog {{ background: {t['window']}; }}
    QWidget {{ color: {t['content']}; font-size: 13px; }}
    QWidget#page, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {t['window']}; }}
    QLabel {{ background: transparent; }}
    QScrollArea {{ border: none; }}

    /* ── Type scale: 22 title · 15 section · 13 body · 11 caption ── */
    QLabel#title {{ font-size: 22px; font-weight: 700; }}
    QLabel#section {{ font-size: 15px; font-weight: 600; }}
    QLabel#secondary, QLabel#hint {{ color: {t['secondary']}; }}
    QLabel#caption {{ color: {t['secondary']}; font-size: 11px; }}
    QLabel#file_loaded {{ font-size: 15px; font-weight: 600; }}
    QLabel#analysis_result {{ color: {t['secondary']}; }}
    QLabel#formLabel {{ color: {t['secondary']}; }}

    /* ── Sidebar ── */
    QWidget#sidebar {{ background: {t['sidebar']}; border-right: 1px solid {t['separator']}; }}
    QLabel#sidebarHeader {{ color: {t['secondary']}; font-size: 11px; font-weight: 600;
                            padding: 12px 12px 4px 12px; }}
    QListWidget#nav, QListWidget#recent {{ background: transparent; border: none; outline: none; }}
    QListWidget#nav::item {{ padding: 6px 10px; margin: 1px 8px; border-radius: 6px; min-height: 20px; }}
    QListWidget#nav::item:hover {{ background: {t['hover']}; }}
    QListWidget#nav::item:selected {{ background: {t['accent_soft']}; color: {t['content']}; }}
    QListWidget#recent::item {{ padding: 4px 10px; margin: 0 8px; border-radius: 6px;
                                color: {t['secondary']}; }}
    QListWidget#recent::item:hover {{ background: {t['hover']}; color: {t['content']}; }}

    /* ── Cards (grouped content) ── */
    QFrame#card {{ background: {t['surface']}; border: 1px solid {t['separator']}; border-radius: 10px; }}
    QFrame#dropZone {{ background: {t['surface']}; border: 2px dashed {t['separator']}; border-radius: 12px; }}
    QFrame#dropZone[active="true"] {{ border-color: {t['accent']}; background: {t['accent_soft']}; }}
    QFrame#banner {{ background: {t['banner']}; border: 1px solid {t['separator']}; border-radius: 8px; }}
    QFrame#hline {{ background: {t['separator']}; max-height: 1px; min-height: 1px; border: none; }}

    /* ── Buttons: 28 px min height, visible focus ── */
    QPushButton, QToolButton {{
        background: {t['surface']}; color: {t['content']};
        border: 1px solid {t['separator']}; border-radius: 6px;
        padding: 4px 14px; min-height: 20px;
    }}
    QPushButton:hover, QToolButton:hover {{ background: {t['hover']}; }}
    QPushButton:pressed, QToolButton:pressed {{ background: {t['separator']}; }}
    QPushButton:focus, QToolButton:focus {{ border: 2px solid {t['focus']}; padding: 3px 13px; }}
    QPushButton:disabled, QToolButton:disabled {{ color: {t['tertiary']}; background: {t['surface_alt']}; }}
    QPushButton#primary_btn {{
        background: {t['accent']}; color: {t['on_accent']}; border: 1px solid {t['accent']};
        font-weight: 600; padding: 6px 18px;
    }}
    QPushButton#primary_btn:hover {{ background: {t['accent_hover']}; }}
    QPushButton#primary_btn:pressed {{ background: {t['accent_pressed']}; }}
    QPushButton#primary_btn:focus {{ border: 2px solid {t['focus']}; padding: 5px 17px; }}
    QPushButton#primary_btn:disabled {{ background: {t['surface_alt']}; color: {t['tertiary']};
                                       border-color: {t['separator']}; }}
    QPushButton#secondary_btn, QPushButton#preset_btn {{ }}
    QPushButton#link_btn {{ background: transparent; border: none; color: {t['accent']};
                            padding: 2px 4px; font-weight: 600; }}
    QPushButton#link_btn:hover {{ text-decoration: underline; }}
    QPushButton#danger_btn {{ color: {t['danger']}; }}

    /* Mute / solo: state shown by fill AND letter weight, not colour alone */
    QPushButton#mute_btn, QPushButton#solo_btn {{
        padding: 0; font-size: 11px; font-weight: 700; color: {t['secondary']};
        border-radius: 5px; min-height: 0;
    }}
    QPushButton#mute_btn:checked {{ background: {t['danger']}; color: #FFFFFF; border-color: {t['danger']}; }}
    QPushButton#solo_btn:checked {{ background: {t['accent']}; color: #FFFFFF; border-color: {t['accent']}; }}

    /* Segmented control */
    QPushButton#segment {{ border-radius: 0; margin: 0; padding: 4px 14px; }}
    QPushButton#segment[first="true"] {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
    QPushButton#segment[last="true"] {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; }}
    QPushButton#segment:checked {{ background: {t['accent']}; color: {t['on_accent']}; border-color: {t['accent']}; }}

    /* ── Inputs ── */
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background: {t['surface']}; color: {t['content']};
        border: 1px solid {t['separator']}; border-radius: 6px;
        padding: 3px 8px; min-height: 20px; min-width: 56px;
        selection-background-color: {t['accent']}; selection-color: {t['on_accent']};
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {t['tertiary']}; }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 2px solid {t['focus']}; padding: 2px 7px; }}
    QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {t['tertiary']}; }}
    QComboBox {{ padding-right: 24px; }}
    QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right;
                            border: none; width: 22px; }}
    QComboBox QAbstractItemView {{
        background: {t['surface']}; border: 1px solid {t['separator']};
        selection-background-color: {t['accent']}; selection-color: {t['on_accent']}; outline: none;
    }}
    QCheckBox {{ spacing: 8px; min-height: 22px; background: transparent; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px;
                           border: 1px solid {t['tertiary']}; background: {t['surface']}; }}
    QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; }}
    QCheckBox:focus {{ outline: none; }}
    QCheckBox::indicator:focus {{ border: 2px solid {t['focus']}; }}

    /* ── Sliders ── */
    QSlider {{ min-height: 24px; background: transparent; }}
    QSlider::groove:horizontal {{ height: 4px; background: {t['separator']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: #FFFFFF; border: 1px solid {t['tertiary']};
        width: 16px; height: 16px; margin: -7px 0; border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{ border-color: {t['accent']}; }}
    QSlider::handle:horizontal:focus {{ border: 2px solid {t['focus']}; }}
    QSlider#pan::sub-page:horizontal {{ background: {t['separator']}; }}

    QProgressBar {{ border: none; border-radius: 3px; background: {t['separator']};
                    max-height: 6px; min-height: 6px; text-align: center; color: transparent; }}
    QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}

    /* ── Lists ── */
    QListWidget {{ background: {t['surface']}; border: 1px solid {t['separator']};
                   border-radius: 8px; padding: 4px; outline: none; }}
    QListWidget::item {{ padding: 6px 8px; border-radius: 5px; min-height: 20px; }}
    QListWidget::item:hover {{ background: {t['hover']}; }}
    QListWidget::item:selected {{ background: {t['accent_soft']}; color: {t['content']}; }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {t['tertiary']}; border-radius: 3px; min-height: 30px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {t['tertiary']}; border-radius: 3px; min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:hover {{ background: {t['separator']}; }}

    QTabWidget::pane {{ border: 1px solid {t['separator']}; border-radius: 8px; top: -1px;
                        background: {t['surface']}; }}
    QTabBar::tab {{ background: transparent; color: {t['secondary']}; padding: 6px 16px;
                    border: none; border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ color: {t['content']}; border-bottom: 2px solid {t['accent']}; }}
    QTabBar::tab:hover {{ color: {t['content']}; }}

    QStatusBar {{ background: {t['window']}; color: {t['secondary']}; border-top: 1px solid {t['separator']}; }}
    QMenuBar {{ background: {t['window']}; }}
    QMenuBar::item:selected {{ background: {t['hover']}; border-radius: 4px; }}
    QMenu {{ background: {t['surface']}; border: 1px solid {t['separator']}; padding: 4px; }}
    QMenu::item {{ padding: 5px 24px 5px 20px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {t['accent']}; color: {t['on_accent']}; }}
    QMenu::item:disabled {{ color: {t['tertiary']}; }}
    QMenu::separator {{ height: 1px; background: {t['separator']}; margin: 4px 8px; }}
    QToolTip {{ background: {t['surface']}; color: {t['content']};
                border: 1px solid {t['separator']}; padding: 5px 8px; border-radius: 4px; }}
    """

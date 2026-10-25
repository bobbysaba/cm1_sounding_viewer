"""main window for the cm1 sounding editor."""
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from . import __version__, edits, params
from . import theme as th
from .plots import HodographWidget, SkewTWidget
from .sounding import Sounding

APP_NAME = "CM1 Sounding Editor"


def _fmt(fmt, x):
    if x is None or not np.isfinite(x):
        return "—"
    return fmt.format(x)


def _json_num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return x if np.isfinite(x) else None


@dataclass
class Comparison:
    """a read-only sounding drawn beside the edited one, with its own parameter column."""
    path: str
    snd: Sounding
    color: str
    prof: object = None
    results: object = None
    visible: bool = True


def _swatch(color, size=10):
    pix = QtGui.QPixmap(size, size)
    pix.fill(QtGui.QColor(color))
    return QtGui.QIcon(pix)


class MainWindow(QtWidgets.QMainWindow):
    TABLE_COLS = ["z (m)", "θ (K)", "qv (g/kg)", "u (m/s)", "v (m/s)",
                  "p (hPa)", "T (°C)", "Td (°C)", "RH (%)"]

    def __init__(self, path=None):
        super().__init__()
        self.path = None
        self.original = None
        self.current = None
        self.saved_state = None
        self.save_path = None
        self.history = []
        self.redo_stack = []
        self.log = []
        self.comparisons = []
        self.settings = params.Settings()
        self.orig_results = None
        self.results = None
        self._prof = None
        self._orig_prof = None
        self._drag_base = None
        self._drag_changed = False
        self._table_dirty = True

        self.param_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self.param_timer.timeout.connect(self._update_params)

        self._build_ui()
        self._build_menus()
        self.setAcceptDrops(True)
        self.resize(1500, 920)
        self._update_actions()
        self._update_title()
        if path:
            self.open_file(path)

    # ---- ui construction --------------------------------------------------------
    def _build_ui(self):
        self.skewt = SkewTWidget()
        self.hodo = HodographWidget()
        self.param_tree = QtWidgets.QTreeWidget()
        self.param_tree.setColumnCount(4)
        self.param_tree.setHeaderLabels(["Parameter", "Original", "Edited", "Δ"])
        self.param_tree.setRootIsDecorated(False)
        self.param_tree.setAlternatingRowColors(True)
        self._param_items = {}
        self.param_tree.setIndentation(0)
        self.param_tree.header().setStretchLastSection(False)
        self.param_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        head = QtGui.QFont()
        head.setPixelSize(11)
        head.setBold(True)
        head.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, 1.0)
        for group in dict.fromkeys(g for g, *_ in params.TABLE):
            gi = QtWidgets.QTreeWidgetItem([group.upper()])
            gi.setFont(0, head)
            gi.setForeground(0, QtGui.QColor(th.MUTED))
            gi.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            gi.setSizeHint(0, QtCore.QSize(0, 28))
            gi.setFirstColumnSpanned(True)
            self.param_tree.addTopLevelItem(gi)
            for g, key, label, unit, fmt in params.TABLE:
                if g != group:
                    continue
                it = QtWidgets.QTreeWidgetItem([f"{label} ({unit})" if unit else label, "", "", ""])
                for c in (1, 2, 3):
                    it.setTextAlignment(c, QtCore.Qt.AlignmentFlag.AlignRight
                                        | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.param_tree.addTopLevelItem(it)
                self._param_items[key] = (it, fmt)
        for c, w in ((1, 66), (2, 66), (3, 60)):
            self.param_tree.header().setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.param_tree.setColumnWidth(c, w)
        for c in (1, 2, 3):
            self.param_tree.headerItem().setTextAlignment(
                c, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right.addWidget(self.hodo)
        right.addWidget(self.param_tree)
        right.setHandleWidth(8)
        right.setSizes([420, 480])
        main_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_split.addWidget(self.skewt)
        main_split.addWidget(right)
        main_split.setHandleWidth(8)
        main_split.setSizes([820, 430])
        self.main_split = main_split

        self.table = QtWidgets.QTableWidget(0, len(self.TABLE_COLS))
        self.table.setHorizontalHeaderLabels(self.TABLE_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setDefaultAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.cellChanged.connect(self._table_edited)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(main_split, "Profile")
        self.tabs.addTab(self.table, "Data table")
        self.tabs.currentChanged.connect(lambda _: self._refresh_table())
        central = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(central)
        cl.setContentsMargins(0, 6, 10, 0)
        cl.addWidget(self.tabs)
        self.setCentralWidget(central)

        # control dock
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setTitleBarWidget(QtWidgets.QWidget())
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(10, 10, 4, 10)
        lay.setSpacing(10)
        lay.addWidget(self._editing_box())
        lay.addWidget(self._layer_box())
        lay.addWidget(self._diag_box())
        lay.addWidget(self._compare_box())
        lay.addWidget(self._checks_box())
        lay.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.controls = panel
        # spin boxes are edited by typing, arrow keys, or the scroll wheel
        for sb in panel.findChildren(QtWidgets.QAbstractSpinBox):
            sb.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.readout = QtWidgets.QLabel("Open a sounding (File ▸ Open) or drop a file on this window.")
        self.statusBar().addPermanentWidget(self.readout)

        self.skewt.dragStarted.connect(self._drag_start)
        self.skewt.dragMoved.connect(self._skew_move)
        self.skewt.dragFinished.connect(self._skew_finish)
        self.skewt.hovered.connect(self._show_readout)
        self.hodo.dragStarted.connect(lambda k: self._drag_start("wind", k))
        self.hodo.dragMoved.connect(self._hodo_move)
        self.hodo.dragFinished.connect(lambda k: self._skew_finish("wind", k))
        self.hodo.hovered.connect(self._show_readout)

    @staticmethod
    def _card(title, layout_cls=QtWidgets.QFormLayout):
        frame = QtWidgets.QFrame()
        frame.setObjectName("card")
        outer = QtWidgets.QVBoxLayout(frame)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(10)
        lab = QtWidgets.QLabel(title.upper())
        lab.setObjectName("sectionTitle")
        outer.addWidget(lab)
        inner = layout_cls()
        if isinstance(inner, QtWidgets.QFormLayout):
            inner.setHorizontalSpacing(12)
            inner.setVerticalSpacing(8)
            inner.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            inner.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            inner.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
        outer.addLayout(inner)
        return frame, inner

    def _editing_box(self):
        box, f = self._card("Editing")
        self.smooth_cb = QtWidgets.QCheckBox("Smooth edits")
        self.smooth_cb.setChecked(True)
        self.smooth_cb.setToolTip("Off: a drag moves exactly one level and layer operations stop "
                                  "sharply at the layer edges.")
        self.sigma_spin = QtWidgets.QDoubleSpinBox(minimum=10, maximum=5000, value=250,
                                                   singleStep=25, decimals=0, suffix=" m")
        self.sigma_spin.setToolTip("Gaussian half-width used to spread drags and blend layer edges.")
        self.smooth_cb.toggled.connect(self.sigma_spin.setEnabled)
        self.hold_combo = QtWidgets.QComboBox()
        self.hold_combo.addItems(["Mixing ratio (qv)", "Relative humidity"])
        f.addRow(self.smooth_cb)
        f.addRow("Smoothing half-width", self.sigma_spin)
        f.addRow("T drags keep", self.hold_combo)
        hint = QtWidgets.QLabel("Drag T, Td, or hodograph points to edit. Drag empty space to pan, "
                                "scroll to zoom.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        f.addRow(hint)
        return box

    def _layer_box(self):
        box, f = self._card("Layer operation")
        self.op_combo = QtWidgets.QComboBox()
        for key, (label, *_rest) in edits.LAYER_OPS.items():
            self.op_combo.addItem(label, key)
        self.zbot_spin = QtWidgets.QDoubleSpinBox(minimum=0, maximum=100000, value=0, singleStep=50,
                                                  decimals=0, suffix=" m")
        self.ztop_spin = QtWidgets.QDoubleSpinBox(minimum=0, maximum=100000, value=1000, singleStep=50,
                                                  decimals=0, suffix=" m")
        whole = QtWidgets.QPushButton("Whole profile")
        whole.setObjectName("ghost")
        whole.clicked.connect(self._whole_profile)
        rng = QtWidgets.QHBoxLayout()
        rng.setSpacing(6)
        rng.addWidget(self.zbot_spin)
        rng.addWidget(QtWidgets.QLabel("to"))
        rng.addWidget(self.ztop_spin)
        self.arg_labels = [QtWidgets.QLabel(), QtWidgets.QLabel()]
        self.arg_spins = [QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()]
        self.carry_cb = QtWidgets.QCheckBox("Carry wind change above the layer")
        self.carry_cb.setChecked(True)
        self.carry_cb.setToolTip("Keeps the shear above the layer unchanged by shifting those "
                                 "winds with the layer top.")
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setObjectName("primary")
        apply_btn.clicked.connect(self._apply_layer_op)
        f.addRow("Operation", self.op_combo)
        f.addRow("Layer", rng)
        f.addRow("", whole)
        for lab, spin in zip(self.arg_labels, self.arg_spins):
            f.addRow(lab, spin)
        f.addRow(self.carry_cb)
        self.op_note = QtWidgets.QLabel()
        self.op_note.setWordWrap(True)
        self.op_note.setObjectName("hint")
        f.addRow(self.op_note)
        f.addRow(apply_btn)
        self.op_combo.currentIndexChanged.connect(self._op_changed)
        self._op_changed()
        return box

    def _diag_box(self):
        box, f = self._card("Diagnostics")
        self.sfc_combo = QtWidgets.QComboBox()
        for label, key in (("Same as lowest level", "lowest"), ("Linear extrapolation", "extrapolate"),
                           ("Zero (calm)", "zero")):
            self.sfc_combo.addItem(label, key)
        self.sfc_combo.setToolTip("The file has no surface wind; this sets the 0 m wind used for "
                                  "shear, SRH and the hodograph. It is never written to the file.")
        self.storm_combo = QtWidgets.QComboBox()
        for label, key in (("Bunkers right mover", "rm"), ("Bunkers left mover", "lm"),
                           ("0–6 km mean wind", "mean"), ("Custom", "custom")):
            self.storm_combo.addItem(label, key)
        self.cu_spin = QtWidgets.QDoubleSpinBox(minimum=-80, maximum=80, decimals=1, suffix=" m/s")
        self.cv_spin = QtWidgets.QDoubleSpinBox(minimum=-80, maximum=80, decimals=1, suffix=" m/s")
        cus = QtWidgets.QHBoxLayout()
        cus.setSpacing(6)
        cus.addWidget(QtWidgets.QLabel("u"))
        cus.addWidget(self.cu_spin)
        cus.addWidget(QtWidgets.QLabel("v"))
        cus.addWidget(self.cv_spin)
        self.parcel_combo = QtWidgets.QComboBox()
        self.parcel_combo.addItems(["ML", "SB", "MU", "None"])
        self.orig_cb = QtWidgets.QCheckBox("Show original profile")
        self.orig_cb.setChecked(True)
        f.addRow("Surface wind", self.sfc_combo)
        f.addRow("Storm motion", self.storm_combo)
        f.addRow("Custom", cus)
        f.addRow("Parcel on Skew-T", self.parcel_combo)
        f.addRow(self.orig_cb)
        for w in (self.sfc_combo, self.storm_combo):
            w.currentIndexChanged.connect(self._settings_changed)
        for w in (self.cu_spin, self.cv_spin):
            w.valueChanged.connect(self._settings_changed)
        self.parcel_combo.currentIndexChanged.connect(lambda _: self._refresh_plots())
        self.orig_cb.toggled.connect(lambda _: self._refresh_plots())
        self._settings_changed()
        return box

    def _compare_box(self):
        box, lay = self._card("Compare", QtWidgets.QVBoxLayout)
        self.compare_list = QtWidgets.QListWidget()
        self.compare_list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.compare_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.compare_list.itemChanged.connect(self._comparison_toggled)
        self.compare_list.itemSelectionChanged.connect(
            lambda: self.compare_remove.setEnabled(bool(self.compare_list.selectedItems())))
        self.compare_hint = QtWidgets.QLabel("Add other input soundings to overlay them and compare "
                                             "their parameters. Unchecked ones are hidden from the plots.")
        self.compare_hint.setWordWrap(True)
        self.compare_hint.setObjectName("hint")
        add = QtWidgets.QPushButton("Add…")
        add.setObjectName("ghost")
        add.clicked.connect(self.add_comparison_dialog)
        self.compare_remove = QtWidgets.QPushButton("Remove")
        self.compare_remove.setObjectName("ghost")
        self.compare_remove.setEnabled(False)
        self.compare_remove.clicked.connect(self._remove_selected_comparisons)
        btns = QtWidgets.QHBoxLayout()
        btns.setSpacing(6)
        btns.addWidget(add)
        btns.addWidget(self.compare_remove)
        btns.addStretch(1)
        lay.addWidget(self.compare_list)
        lay.addWidget(self.compare_hint)
        lay.addLayout(btns)
        self.compare_list.hide()
        return box

    def _checks_box(self):
        box, lay = self._card("Checks", QtWidgets.QVBoxLayout)
        self.checks_list = QtWidgets.QListWidget()
        self.checks_list.setWordWrap(True)
        self.checks_list.setMinimumHeight(90)
        self.checks_list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.checks_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        lay.addWidget(self.checks_list)
        return box

    def _action(self, menu, text, slot, shortcut=None):
        act = QtGui.QAction(text, self)
        if shortcut is not None:
            act.setShortcut(QtGui.QKeySequence(shortcut))
        act.triggered.connect(lambda checked=False: slot())
        menu.addAction(act)
        return act

    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        self.act_open = self._action(m, "Open…", self.open_dialog, QtGui.QKeySequence.StandardKey.Open)
        self._action(m, "Add Comparison Sounding…", self.add_comparison_dialog, "Ctrl+Shift+O")
        self.act_save = self._action(m, "Save", self.save, QtGui.QKeySequence.StandardKey.Save)
        self.act_save_as = self._action(m, "Save As…", self.save_as, QtGui.QKeySequence.StandardKey.SaveAs)
        m.addSeparator()
        self.act_sidecar = m.addAction("Write edit log (.edits.json) when saving")
        self.act_sidecar.setCheckable(True)
        self.act_sidecar.setChecked(True)
        m.addSeparator()
        self._action(m, "Quit", self.close, QtGui.QKeySequence.StandardKey.Quit)
        e = self.menuBar().addMenu("&Edit")
        self.act_undo = self._action(e, "Undo", self.undo, QtGui.QKeySequence.StandardKey.Undo)
        self.act_redo = self._action(e, "Redo", self.redo, QtGui.QKeySequence.StandardKey.Redo)
        e.addSeparator()
        self.act_revert = self._action(e, "Revert to original", self.revert)
        v = self.menuBar().addMenu("&View")
        self._action(v, "Reset Skew-T view", self.skewt.reset_view)
        self._action(v, "Reset hodograph view", self.hodo.reset_view)

    # ---- file handling ----------------------------------------------------------
    def open_dialog(self):
        if not self._confirm_discard():
            return
        start = str(Path(self.path).parent) if self.path else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open input_sounding", start, "All files (*)")
        if path:
            self.open_file(path, confirm=False)

    def open_file(self, path, confirm=True):
        if confirm and not self._confirm_discard():
            return
        try:
            snd = Sounding.read(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not open file", f"{path}\n\n{exc}")
            return False
        self.path = str(path)
        self.original = snd
        self.current = snd.copy()
        self.saved_state = snd.copy()
        self.save_path = None
        self.history.clear()
        self.redo_stack.clear()
        self.log = []
        z = snd.z
        self.zbot_spin.setMaximum(z[-1])
        self.ztop_spin.setMaximum(z[-1])
        self.hodo._fit_done = False
        self.orig_results = params.compute(self.original, self.settings)
        self._orig_prof = self.original.derived()
        self.readout.setText("")
        self._rebuild_comparisons()
        self._after_change()
        self.skewt.reset_view()
        self.statusBar().showMessage(f"Opened {path} ({len(z) - 1} levels, top {z[-1]:.0f} m)", 6000)
        return True

    # ---- comparison soundings -----------------------------------------------------
    def add_comparison_dialog(self):
        start = str(Path(self.path).parent) if self.path else str(Path.home())
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add comparison soundings", start, "All files (*)")
        if paths:
            self.add_comparisons(paths)

    def add_comparisons(self, paths):
        """load read-only soundings to compare; with nothing open, the first becomes the edited one."""
        paths = [str(p) for p in paths]
        if self.current is None:
            if not paths or not self.open_file(paths.pop(0)):
                return
        have = {str(Path(c.path).resolve()) for c in self.comparisons}
        added, failed = 0, []
        for path in paths:
            key = str(Path(path).resolve())
            if key in have:
                continue
            try:
                snd = Sounding.read(path)
            except Exception as exc:
                failed.append(f"{path}: {exc}")
                continue
            used = {c.color for c in self.comparisons}
            free = [c for c in th.COMPARE if c not in used]
            color = free[0] if free else th.COMPARE[len(self.comparisons) % len(th.COMPARE)]
            self.comparisons.append(Comparison(path, snd, color, snd.derived(),
                                               params.compute(snd, self.settings)))
            have.add(key)
            added += 1
        if failed:
            QtWidgets.QMessageBox.warning(self, "Could not open some files", "\n\n".join(failed))
        if added:
            self._rebuild_comparisons()
            self._refresh_plots()
            self._update_params()
            self.statusBar().showMessage(
                f"Added {added} comparison sounding{'s' if added > 1 else ''}", 6000)

    def _remove_selected_comparisons(self):
        rows = {self.compare_list.row(it) for it in self.compare_list.selectedItems()}
        if not rows:
            return
        self.comparisons = [c for i, c in enumerate(self.comparisons) if i not in rows]
        self._rebuild_comparisons()
        self._refresh_plots()
        self._update_params()

    def _comparison_toggled(self, item):
        row = self.compare_list.row(item)
        if 0 <= row < len(self.comparisons):
            self.comparisons[row].visible = item.checkState() == QtCore.Qt.CheckState.Checked
            self._refresh_plots()

    def _comparison_names(self):
        """file names, with the parent folder added where names clash (CM1 files are often all
        called input_sounding)."""
        paths = ([self.path] if self.path else []) + [c.path for c in self.comparisons]
        names = [Path(p).name for p in paths]
        out = [f"{Path(p).parent.name}/{n}" if names.count(n) > 1 else n for p, n in zip(paths, names)]
        return out[len(out) - len(self.comparisons):]

    def _rebuild_comparisons(self):
        """sync the compare list and the extra parameter columns with self.comparisons."""
        names = self._comparison_names()
        self.compare_list.blockSignals(True)
        self.compare_list.clear()
        for c, name in zip(self.comparisons, names):
            it = QtWidgets.QListWidgetItem(_swatch(c.color), name)
            it.setToolTip(c.path)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if c.visible else QtCore.Qt.CheckState.Unchecked)
            self.compare_list.addItem(it)
        self.compare_list.blockSignals(False)
        rows = self.compare_list.sizeHintForRow(0) if self.comparisons else 0
        self.compare_list.setFixedHeight(min(len(self.comparisons), 5) * rows + 4)
        self.compare_list.setVisible(bool(self.comparisons))
        self.compare_hint.setVisible(not self.comparisons)
        self.compare_remove.setEnabled(False)

        tree = self.param_tree
        tree.setColumnCount(4 + len(self.comparisons))
        head = tree.headerItem()
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        font = QtGui.QFont()
        font.setPixelSize(11)    # matches the header stylesheet
        font.setBold(True)
        fm = QtGui.QFontMetrics(font)
        tree.header().setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        for i, (c, name) in enumerate(zip(self.comparisons, names)):
            col = 4 + i
            name = name.split("/")[0]    # just the folder when file names clash
            head.setText(col, name)
            head.setIcon(col, _swatch(c.color, 8))
            head.setToolTip(col, c.path)
            head.setTextAlignment(col, right)
            tree.header().setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Interactive)
            tree.setColumnWidth(col, min(110, max(72, fm.horizontalAdvance(name) + 36)))
            for it, _ in self._param_items.values():
                it.setTextAlignment(col, right)
        # widen the right-hand pane so the new columns are visible, up to about half the window
        need = 200 + sum(tree.columnWidth(c) for c in range(1, tree.columnCount())) + 24
        left, cur = self.main_split.sizes()
        want = min(need, (left + cur) // 2)
        if cur < want:
            self.main_split.setSizes([left + cur - want, want])

    def _is_modified(self):
        return self.current is not None and not self.current.equals(self.original)

    def _is_unsaved(self):
        return self.current is not None and not self.current.equals(self.saved_state)

    def _confirm_discard(self):
        if not (self._is_modified() and self._is_unsaved()):
            return True
        r = QtWidgets.QMessageBox.question(
            self, "Unsaved changes", "The edited sounding has not been saved. Save it first?",
            QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        if r == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save()
        return r == QtWidgets.QMessageBox.StandardButton.Discard

    def save(self):
        if self.save_path is None:
            return self.save_as()
        return self._write(self.save_path)

    def save_as(self):
        if self.current is None:
            return False
        src = Path(self.path)
        default = self.save_path or str(src.with_name(f"{src.name}_edited"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save sounding as", default, "All files (*)")
        if not path:
            return False
        if Path(path).resolve() == src.resolve():
            r = QtWidgets.QMessageBox.warning(
                self, "Overwrite original?",
                f"This replaces the original sounding:\n{src}\n\nOverwrite it?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if r != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        return self._write(path)

    def _write(self, path):
        issues = self.current.check()
        errors = [msg for sev, msg, _ in issues if sev == "error"]
        if errors:
            QtWidgets.QMessageBox.critical(self, "Cannot save",
                                           "The profile has problems CM1 cannot use:\n• " + "\n• ".join(errors))
            return False
        if any(sev == "warning" for sev, _, _ in issues):
            r = QtWidgets.QMessageBox.warning(
                self, "Supersaturated levels",
                "Some levels are supersaturated. Save anyway?\n"
                "(Layer operation ▸ Cap at saturation fixes this.)",
                QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Cancel)
            if r != QtWidgets.QMessageBox.StandardButton.Save:
                return False
        try:
            self.current.write(path)
            if self.act_sidecar.isChecked():
                self._write_sidecar(path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", f"{path}\n\n{exc}")
            return False
        self.save_path = str(path)
        self.saved_state = self.current.copy()
        self._update_actions()
        self._update_title()
        self.statusBar().showMessage(f"Saved {path}", 8000)
        return True

    def _write_sidecar(self, path):
        res = params.compute(self.current, self.settings)
        record = {
            "tool": f"cm1-sounding-editor {__version__}",
            "saved": datetime.now().isoformat(timespec="seconds"),
            "source": str(Path(self.path).resolve()),
            "output": str(Path(path).resolve()),
            "diagnostic_settings": asdict(self.settings),
            "operations": self.log,
            "parameters": {key: {"original": _json_num(self.orig_results.values.get(key)),
                                 "edited": _json_num(res.values.get(key))}
                           for _, key, *_ in params.TABLE},
        }
        Path(str(path) + ".edits.json").write_text(json.dumps(record, indent=2))

    def closeEvent(self, ev):
        if self._confirm_discard():
            ev.accept()
        else:
            ev.ignore()

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        # option/alt-drop adds every file as a comparison; a plain drop opens the first file for
        # editing and adds any others as comparisons
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            self.add_comparisons(paths)
        elif self.open_file(paths[0]):
            self.add_comparisons(paths[1:])

    # ---- edit plumbing ------------------------------------------------------------
    def _sigma(self):
        return self.sigma_spin.value() if self.smooth_cb.isChecked() else 0.0

    def _commit(self, previous, entry, new_log=None):
        self.history.append((previous, list(self.log)))
        if len(self.history) > 300:
            self.history.pop(0)
        self.log = list(self.log) + [entry] if new_log is None else new_log
        self.redo_stack.clear()
        self._after_change()

    def undo(self):
        if not self.history:
            return
        self.redo_stack.append((self.current, list(self.log)))
        self.current, self.log = self.history.pop()
        self._after_change()

    def redo(self):
        if not self.redo_stack:
            return
        self.history.append((self.current, list(self.log)))
        self.current, self.log = self.redo_stack.pop()
        self._after_change()

    def revert(self):
        if not self._is_modified():
            return
        prev = self.current
        self.current = self.original.copy()
        self._commit(prev, None, new_log=[])

    def _drag_start(self, kind, k):
        if self.current is None:
            return
        self._drag_base = self.current.copy()
        self._drag_changed = False

    def _skew_move(self, kind, k, t_new):
        if self._drag_base is None:
            return
        if kind == "t":
            hold = "rh" if self.hold_combo.currentIndex() == 1 else "qv"
            self.current = edits.drag_temperature(self._drag_base, k, t_new, self._sigma(), hold)
        else:
            self.current = edits.drag_dewpoint(self._drag_base, k, t_new, self._sigma())
        self._drag_changed = True
        self._refresh_plots()
        self.skewt.show_marker(kind, k)
        self.param_timer.start()

    def _hodo_move(self, k, u, v):
        if self._drag_base is None:
            return
        self.current = edits.drag_wind(self._drag_base, k, u, v, self._sigma())
        self._drag_changed = True
        self._refresh_plots()
        self.hodo.show_marker(k)
        self.param_timer.start()

    def _skew_finish(self, kind, k):
        base = self._drag_base
        self._drag_base = None
        self.skewt.show_marker(None, -1)
        self.hodo.show_marker(None)
        if base is None or not self._drag_changed or self.current.equals(base):
            self.current = base if base is not None else self.current
            self._after_change()
            return
        z = float(base.z[k])
        entry = {"op": f"drag_{'temperature' if kind == 't' else 'dewpoint' if kind == 'td' else 'wind'}",
                 "z_m": z, "smoothing_halfwidth_m": self._sigma()}
        if kind == "wind":
            entry.update(du=round(float(self.current.u[k] - base.u[k]), 3),
                         dv=round(float(self.current.v[k] - base.v[k]), 3))
        else:
            b, c = base.derived(), self.current.derived()
            arr = "t" if kind == "t" else "td"
            entry["delta_K"] = round(float(getattr(c, arr)[k] - getattr(b, arr)[k]), 3)
            if kind == "t":
                entry["moisture_held"] = "rh" if self.hold_combo.currentIndex() == 1 else "qv"
        self._commit(base, entry)

    def _whole_profile(self):
        if self.current is not None:
            self.zbot_spin.setValue(0)
            self.ztop_spin.setValue(self.current.z[-1])

    def _op_changed(self):
        key = self.op_combo.currentData()
        label, fn, specs, uses_taper, uses_carry = edits.LAYER_OPS[key]
        for i, (lab, spin) in enumerate(zip(self.arg_labels, self.arg_spins)):
            if i < len(specs):
                name, default, lo, hi, step, dec = specs[i]
                lab.setText(name)
                spin.setRange(lo, hi)
                spin.setDecimals(dec)
                spin.setSingleStep(step)
                spin.setValue(default)
                lab.show()
                spin.show()
            else:
                lab.hide()
                spin.hide()
        self.carry_cb.setVisible(uses_carry)
        notes = {
            True: "Layer edges blend over the smoothing half-width when Smooth edits is on.",
            False: "",
        }
        self.op_note.setText(notes[uses_taper])
        self.op_note.setVisible(uses_taper)

    def _apply_layer_op(self):
        if self.current is None:
            return
        key = self.op_combo.currentData()
        label, fn, specs, uses_taper, uses_carry = edits.LAYER_OPS[key]
        zbot, ztop = self.zbot_spin.value(), self.ztop_spin.value()
        if ztop <= zbot:
            QtWidgets.QMessageBox.information(self, "Layer", "The layer top must be above its bottom.")
            return
        args = [self.arg_spins[i].value() for i in range(len(specs))]
        new = edits.apply_layer_op(self.current, key, zbot, ztop, args, self._sigma(),
                                   self.carry_cb.isChecked())
        if new.equals(self.current):
            self.statusBar().showMessage(f"{label}: no change at file precision", 5000)
            return
        entry = {"op": key, "label": label, "zbot_m": zbot, "ztop_m": ztop,
                 "args": {specs[i][0]: args[i] for i in range(len(specs))}}
        if uses_taper:
            entry["edge_blend_halfwidth_m"] = self._sigma()
        if uses_carry:
            entry["carry_above"] = self.carry_cb.isChecked()
        prev = self.current
        self.current = new
        self._commit(prev, entry)

    def _table_edited(self, row, col):
        if self.current is None:
            return
        item = self.table.item(row, col)
        fields = {0: "z", 1: "theta", 2: "qv", 3: "u", 4: "v", 5: "psfc"}
        field = fields.get(col)
        try:
            value = float(item.text())
        except ValueError:
            self._table_dirty = True
            self._refresh_table()
            return
        prev = self.current
        self.current = edits.set_level(prev, row, field, value)
        if self.current.equals(prev):
            return
        entry = {"op": "set_value", "field": field, "z_m": float(prev.z[row]), "value": value}
        self._commit(prev, entry)

    def _settings_changed(self, *_):
        self.settings = params.Settings(
            surface_wind=self.sfc_combo.currentData(),
            storm_motion=self.storm_combo.currentData(),
            custom_u=self.cu_spin.value(),
            custom_v=self.cv_spin.value())
        custom = self.storm_combo.currentData() == "custom"
        self.cu_spin.setEnabled(custom)
        self.cv_spin.setEnabled(custom)
        for c in getattr(self, "comparisons", []):
            c.results = params.compute(c.snd, self.settings)
        if self.original is not None:
            self.orig_results = params.compute(self.original, self.settings)
            self._update_params()

    # ---- refresh ------------------------------------------------------------------
    def _after_change(self):
        self._table_dirty = True
        self._refresh_plots()
        self._refresh_table()
        self._refresh_checks()
        self._update_actions()
        self._update_title()
        self.param_timer.start()

    def _refresh_plots(self):
        if self.current is None:
            return
        self._prof = self.current.derived()
        show = self.orig_cb.isChecked()
        name = self.parcel_combo.currentText()
        parcel = None
        if self.results is not None and name in self.results.parcels and self._drag_base is None:
            parcel = self.results.parcels[name]
        self.skewt.set_data(self._prof, self._orig_prof, parcel, show_orig=show and self._is_modified())
        u, v = params.filled_winds(self.current, self.settings.surface_wind)
        ou, ov = params.filled_winds(self.original, self.settings.surface_wind)
        res = self.results
        storms = None
        sel = None
        if res is not None:
            storms = {"RM": res.bunkers_rm, "LM": res.bunkers_lm, "MW": res.mean_wind}
            sel = res.storm
        self.hodo.set_data(self.current.z, u, v, (self.original.z, ou, ov), storms, sel,
                           show_orig=show and self._is_modified())
        shown = [c for c in self.comparisons if c.visible]
        self.skewt.set_comparisons([(c.prof, c.color) for c in shown])
        self.hodo.set_comparisons([(*params.filled_winds(c.snd, self.settings.surface_wind), c.color)
                                   for c in shown])

    def _update_params(self):
        if self.current is None:
            return
        self.results = params.compute(self.current, self.settings)
        o, e = self.orig_results.values, self.results.values
        for key, (it, fmt) in self._param_items.items():
            a, b = o.get(key, np.nan), e.get(key, np.nan)
            it.setText(1, _fmt(fmt, a))
            it.setText(2, _fmt(fmt, b))
            if np.isfinite(a) and np.isfinite(b):
                d = b - a
                txt = _fmt("{:+" + fmt[2:], d)
                changed = txt.strip("+-0.") != ""
            else:
                changed = np.isfinite(a) != np.isfinite(b)
                txt = "changed" if changed else ""
            it.setText(3, txt if changed else "")
            color = QtGui.QColor(th.ACCENT) if changed else QtGui.QColor(th.TEXT)
            for c in (2, 3):
                it.setForeground(c, color)
            for i, comp in enumerate(self.comparisons):
                it.setText(4 + i, _fmt(fmt, comp.results.values.get(key, np.nan)))
        if self._drag_base is None:
            self._refresh_plots()

    def _refresh_table(self):
        if self.current is None or not self._table_dirty or self.tabs.currentIndex() != 1:
            return
        self._table_dirty = False
        snd, prof = self.current, self.current.derived()
        n = len(snd.z)
        self.table.blockSignals(True)
        self.table.setRowCount(n)
        ro = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        rw = ro | QtCore.Qt.ItemFlag.ItemIsEditable
        grey = QtGui.QColor(th.FAINT)
        for k in range(n):
            vals = [snd.z[k], snd.theta[k], snd.qv[k], snd.u[k], snd.v[k],
                    prof.p[k] / 100.0, prof.t[k] - 273.15, prof.td[k] - 273.15, prof.rh[k] * 100.0]
            for c, val in enumerate(vals):
                if k == 0 and c in (3, 4):
                    txt = "—"
                elif k == 0 and c == 0:
                    txt = "sfc (0)"
                else:
                    txt = f"{val:.3f}" if c < 6 else f"{val:.2f}" if c < 8 else f"{val:.1f}"
                it = QtWidgets.QTableWidgetItem(txt)
                editable = (c in (1, 2) or (c in (0, 3, 4) and k > 0) or (c == 5 and k == 0))
                it.setFlags(rw if editable else ro)
                if not editable:
                    it.setForeground(grey)
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(k, c, it)
        self.table.blockSignals(False)

    def _refresh_checks(self):
        self.checks_list.clear()
        if self.current is None:
            return
        issues = self.current.check()
        if not issues:
            self.checks_list.addItem("No issues found.")
            return
        icons = {"error": "✖", "warning": "⚠", "info": "ℹ"}
        for sev, msg, zs in issues:
            where = ""
            if zs:
                shown = ", ".join(f"{z:.0f}" for z in zs[:8])
                more = f" (+{len(zs) - 8} more)" if len(zs) > 8 else ""
                where = f" at {len(zs)} level(s): {shown}{more} m"
            item = QtWidgets.QListWidgetItem(f"{icons[sev]}  {msg}{where}")
            item.setForeground(QtGui.QColor({"error": th.DANGER, "warning": th.WARN}.get(sev, th.MUTED)))
            self.checks_list.addItem(item)

    def _show_readout(self, k):
        if self._prof is None or k < 0 or k >= len(self._prof.z):
            return
        p, s = self._prof, self.current
        wind = "—" if k == 0 else f"{s.u[k]:.1f}, {s.v[k]:.1f} m/s"
        self.readout.setText(
            f"z {p.z[k]:.0f} m   p {p.p[k] / 100:.1f} hPa   T {p.t[k] - 273.15:.2f} °C   "
            f"Td {p.td[k] - 273.15:.2f} °C   θ {s.theta[k]:.3f} K   qv {s.qv[k]:.3f} g/kg   "
            f"RH {p.rh[k] * 100:.0f}%   u,v {wind}")

    def _update_actions(self):
        loaded = self.current is not None
        modified = self._is_modified()
        self.act_save.setEnabled(modified and self._is_unsaved())
        self.act_save_as.setEnabled(modified)
        self.act_undo.setEnabled(bool(self.history))
        self.act_redo.setEnabled(bool(self.redo_stack))
        self.act_revert.setEnabled(modified)
        self.controls.setEnabled(loaded)

    def _update_title(self):
        if self.current is None:
            self.setWindowTitle(APP_NAME)
            return
        name = Path(self.save_path or self.path).name
        mark = " •" if self._is_modified() and self._is_unsaved() else ""
        self.setWindowTitle(f"{name}{mark} — {APP_NAME}")
        self.setWindowModified(False)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    pg.setConfigOptions(antialias=True)
    th.apply(app)
    path = argv[1] if len(argv) > 1 else None
    win = MainWindow(path)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

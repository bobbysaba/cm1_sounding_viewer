"""interactive skew-t and hodograph widgets (pyqtgraph, dark theme)."""
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui

from . import theme as th
from . import thermo

SKEW = 38.0          # x shift (deg c) per unit of ln(1000/p)
PICK_PX = 10         # pick radius in pixels
BANDS = [(0.0, 1000.0), (1000.0, 3000.0), (3000.0, 6000.0), (6000.0, 9000.0), (9000.0, 1e9)]
MIXING_RATIOS = (1, 2, 4, 7, 10, 16, 24)
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]


def skew_xy(t_k, p_pa):
    y = np.log(100000.0 / np.asarray(p_pa))
    return np.asarray(t_k) - 273.15 + SKEW * y, y


def _font(px, bold=False):
    f = QtGui.QFont()
    f.setPixelSize(px)
    f.setBold(bold)
    return f


def _style_axis(axis, label=None):
    axis.setPen(pg.mkPen(th.AXIS))
    axis.setTextPen(pg.mkPen(th.AXIS_TEXT))
    axis.setStyle(tickLength=-4, tickTextOffset=6, tickFont=_font(11))
    if label is not None:
        axis.setLabel(label, color=th.MUTED, **{"font-size": "11px"})


def _faded(color):
    c = pg.mkColor(color)
    c.setAlpha(th.ORIG_ALPHA)
    return c


def _band_indices(z):
    """level indices for each hodograph height band, including the next level so bands connect."""
    out = []
    for lo, hi in BANDS:
        idx = np.where((z >= lo) & (z <= hi))[0]
        if idx.size and idx[-1] + 1 < len(z):
            idx = np.append(idx, idx[-1] + 1)
        out.append(idx)
    return out


def _style_plot(widget):
    widget.setBackground(th.PLOT_BG)
    pi = widget.getPlotItem()
    pi.hideButtons()
    pi.setMenuEnabled(False)
    pi.layout.setContentsMargins(6, 10, 6, 6)


# ---- shared drag-aware viewbox --------------------------------------------------
class EditViewBox(pg.ViewBox):
    """left-drag near a point edits it; left-drag elsewhere pans as usual."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.picker = None
        self.on_start = None
        self.on_move = None
        self.on_finish = None
        self._handle = None

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton or self.picker is None:
            super().mouseDragEvent(ev, axis)
            return
        if ev.isStart():
            self._handle = self.picker(self.mapSceneToView(ev.buttonDownScenePos()))
            if self._handle is not None:
                self.on_start(self._handle)
        if self._handle is None:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        self.on_move(self._handle, self.mapSceneToView(ev.scenePos()))
        if ev.isFinish():
            handle, self._handle = self._handle, None
            self.on_finish(handle)

    def pixel_size(self):
        px, py = self.viewPixelSize()
        return max(px, 1e-12), max(py, 1e-12)


class PlotView(pg.PlotWidget):
    """plot widget that also zooms on trackpad pinch (macos native gesture), which pyqtgraph ignores."""

    def viewportEvent(self, ev):
        if (ev.type() == QtCore.QEvent.Type.NativeGesture
                and ev.gestureType() == QtCore.Qt.NativeGestureType.ZoomNativeGesture):
            vb = self.getPlotItem().getViewBox()
            # value() is the change in magnification; positive means fingers spreading (zoom in)
            factor = 1.0 / max(1.0 + ev.value(), 0.1)
            scene_pos = self.mapToScene(ev.position().toPoint())
            vb.scaleBy((factor, factor), center=vb.mapSceneToView(scene_pos))
            ev.accept()
            return True
        return super().viewportEvent(ev)


# ---- axes ---------------------------------------------------------------------
class PressureAxis(pg.AxisItem):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.enableAutoSIPrefix(False)

    def tickValues(self, minVal, maxVal, size):
        vals = [np.log(1000.0 / p) for p in PRESSURE_LEVELS]
        return [(1.0, [v for v in vals if minVal <= v <= maxVal])]

    def tickStrings(self, values, scale, spacing):
        return [f"{1000.0 * np.exp(-v):.0f}" for v in values]


class ParcelLevelAxis(pg.AxisItem):
    """right-hand axis that labels the parcel lcl, lfc and el in their own colors."""

    COLORS = {"LCL": th.LCL, "LFC": th.LFC, "EL": th.EL}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.enableAutoSIPrefix(False)
        self._levels = []   # (y, label, detail)

    def set_levels(self, levels):
        self._levels = levels
        self.picture = None
        self.update()

    def tickValues(self, minVal, maxVal, size):
        return [(1.0, [y for y, *_ in self._levels if minVal <= y <= maxVal])]

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            match = min(self._levels, key=lambda lv: abs(lv[0] - v))
            out.append(f"{match[1]}  {match[2]}")
        return out

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        # axis line and ticks in per-level colors
        super().drawPicture(p, axisSpec, [], [])
        for pen, p1, p2 in tickSpecs:
            # recover the level from the tick's scene position
            yv = self.linkedView().mapSceneToView(self.mapToScene(p1)).y() if self.linkedView() else None
            col = th.AXIS_TEXT
            if yv is not None and self._levels:
                match = min(self._levels, key=lambda lv: abs(lv[0] - yv))
                col = self.COLORS.get(match[1], th.AXIS_TEXT)
            p.setPen(pg.mkPen(col, width=2))
            p.drawLine(p1, p2)
        if self.style["tickFont"] is not None:
            p.setFont(self.style["tickFont"])
        p.setClipRect(self.boundingRect().toAlignedRect())
        # stack labels that would overlap (e.g. lcl and lfc close together)
        placed = []
        for rect, flags, text in sorted(textSpecs, key=lambda s: s[0].top()):
            r = QtCore.QRectF(rect)
            if placed and r.top() < placed[-1].bottom():
                r.moveTop(placed[-1].bottom())
            placed.append(r)
            p.setPen(pg.mkPen(self.COLORS.get(text.split()[0], th.AXIS_TEXT)))
            p.drawText(r, int(flags), text)


# ---- skew-t -------------------------------------------------------------------
class SkewTWidget(PlotView):
    dragStarted = QtCore.pyqtSignal(str, int)
    dragMoved = QtCore.pyqtSignal(str, int, float)      # kind, level, new temperature (k)
    dragFinished = QtCore.pyqtSignal(str, int)
    hovered = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        self.vb = EditViewBox()
        self.parcel_axis = ParcelLevelAxis("right")
        super().__init__(parent, viewBox=self.vb,
                         axisItems={"left": PressureAxis("left"), "right": self.parcel_axis})
        _style_plot(self)
        pi = self.getPlotItem()
        pi.showAxis("right")
        _style_axis(pi.getAxis("left"), "Pressure (hPa)")
        _style_axis(pi.getAxis("bottom"), "Temperature (°C)")
        _style_axis(self.parcel_axis)
        self.parcel_axis.setWidth(78)
        self._draw_background()

        self.orig_t = self.plot(pen=pg.mkPen(_faded(th.T_LINE), width=2.4))
        self.orig_td = self.plot(pen=pg.mkPen(_faded(th.TD_LINE), width=2.4))
        self.parcel = self.plot(pen=pg.mkPen(th.PARCEL, width=1.3, style=QtCore.Qt.PenStyle.DashLine))
        self.parcel_marks = pg.ScatterPlotItem(size=7, pen=None)
        self.addItem(self.parcel_marks)
        self.t_curve = self.plot(pen=pg.mkPen(th.T_LINE, width=2.4))
        self.td_curve = self.plot(pen=pg.mkPen(th.TD_LINE, width=2.4))
        self.marker = pg.ScatterPlotItem(size=12, pen=pg.mkPen(th.MARKER, width=1.5), brush=None)
        self.addItem(self.marker)
        self.hover_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#39404b", width=1))
        self.addItem(self.hover_line)
        self.hover_line.hide()
        # comparison curves are added later, so lift the edited profile above them
        for item in (self.orig_t, self.orig_td, self.parcel, self.parcel_marks,
                     self.t_curve, self.td_curve, self.marker):
            item.setZValue(10)
        self.comp_curves = []     # (t, td) curve pairs, reused between updates

        # overlays pinned to the view edges: height ticks (left) and mixing-ratio labels (bottom)
        self.height_ticks = pg.PlotCurveItem(pen=pg.mkPen(th.AXIS_TEXT, width=1.2), connect="pairs")
        self.addItem(self.height_ticks, ignoreBounds=True)
        self.height_labels = []
        self.mix_labels = []
        for w in MIXING_RATIOS:
            lab = pg.TextItem(str(w), color=th.MIXING_LABEL, anchor=(0.5, 1.0))
            lab.setFont(_font(10))
            self.addItem(lab, ignoreBounds=True)
            self.mix_labels.append((w, lab))
        self._height_ticks = []   # (y, text)

        self.vb.picker = self._pick
        self.vb.on_start = self._start
        self.vb.on_move = self._move
        self.vb.on_finish = lambda h: self.dragFinished.emit(*h)
        self.vb.sigRangeChanged.connect(self._update_overlays)
        self.vb.sigResized.connect(self._update_overlays)
        self.scene().sigMouseMoved.connect(self._mouse_moved)
        self._y = np.array([])
        self._xt = np.array([])
        self._xtd = np.array([])
        self._drag_y = 0.0
        self.reset_view()

    def reset_view(self):
        self.vb.setRange(xRange=(-40, 50), yRange=(np.log(1000 / 1050), np.log(1000 / 100)), padding=0)

    def _draw_background(self):
        ymin, ymax = np.log(1000 / 1100), np.log(1000 / 50)
        for p in PRESSURE_LEVELS:
            self.addItem(pg.InfiniteLine(np.log(1000.0 / p), angle=0, pen=pg.mkPen(th.ISOBAR, width=1)))
        for tc in range(-140, 61, 10):
            pen = pg.mkPen(th.ISOTHERM_0 if tc == 0 else th.ISOTHERM, width=1)
            self.plot([tc + SKEW * ymin, tc + SKEW * ymax], [ymin, ymax], pen=pen)
        pgrid = np.linspace(110000.0, 5000.0, 80)
        for theta in range(240, 481, 10):
            x, y = skew_xy(theta * thermo.exner(pgrid), pgrid)
            self.plot(x, y, pen=pg.mkPen(th.DRY_ADIABAT, width=1))
        pmoist = np.linspace(110000.0, 15000.0, 60)
        for t0 in range(-20, 37, 4):
            the = thermo.sat_theta_e(t0 + 273.15, 100000.0)
            x, y = skew_xy(thermo.temp_on_pseudoadiabat(the, pmoist), pmoist)
            self.plot(x, y, pen=pg.mkPen(th.MOIST_ADIABAT, width=1, style=QtCore.Qt.PenStyle.DashLine))
        pw = np.linspace(110000.0, 60000.0, 20)
        for w in MIXING_RATIOS:
            x, y = skew_xy(thermo.dewpoint(w / 1000.0, pw), pw)
            self.plot(x, y, pen=pg.mkPen(th.MIXING_RATIO, width=1, style=QtCore.Qt.PenStyle.DotLine))

    def _update_overlays(self, *_):
        (x0, x1), (y0, y1) = self.vb.viewRange()
        px, py = self.vb.pixel_size()
        # mixing-ratio labels sit just above the temperature axis
        yb = y0 + 3 * py
        pb = 100000.0 * np.exp(-yb)
        for w, lab in self.mix_labels:
            x = float(thermo.dewpoint(w / 1000.0, pb) - 273.15 + SKEW * yb)
            if x0 + 8 * px < x < x1 - 8 * px and pb > 60000.0:
                lab.setPos(x, yb)
                lab.show()
            else:
                lab.hide()
        # height ticks and labels along the inside of the left edge
        xs, ys = [], []
        for (y, text), lab in zip(self._height_ticks, self.height_labels):
            if y0 + 6 * py < y < y1 - 6 * py:
                xs += [x0, x0 + 7 * px]
                ys += [y, y]
                lab.setPos(x0 + 9 * px, y)
                lab.show()
            else:
                lab.hide()
        self.height_ticks.setData(np.array(xs), np.array(ys))

    def _set_height_ticks(self, prof, y):
        ticks = []
        for km in range(0, 19):
            h = km * 1000.0
            if h <= prof.z[-1]:
                ticks.append((float(np.interp(h, prof.z, y)), "SFC" if km == 0 else f"{km} km"))
        if [t for _, t in ticks] != [t for _, t in self._height_ticks]:
            for lab in self.height_labels:
                self.removeItem(lab)
            self.height_labels = []
            for _, text in ticks:
                lab = pg.TextItem(text, color=th.AXIS_TEXT, anchor=(0.0, 0.5))
                lab.setFont(_font(10))
                self.addItem(lab, ignoreBounds=True)
                self.height_labels.append(lab)
        self._height_ticks = ticks
        self._update_overlays()

    def set_data(self, prof, orig_prof=None, parcel=None, show_orig=True):
        xt, y = skew_xy(prof.t, prof.p)
        xtd, _ = skew_xy(prof.td, prof.p)
        self._xt, self._xtd, self._y = xt, xtd, y
        self.t_curve.setData(xt, y)
        self.td_curve.setData(xtd, y)
        if orig_prof is not None and show_orig:
            ox, oy = skew_xy(orig_prof.t, orig_prof.p)
            odx, _ = skew_xy(orig_prof.td, orig_prof.p)
            self.orig_t.setData(ox, oy)
            self.orig_td.setData(odx, oy)
        else:
            self.orig_t.setData([], [])
            self.orig_td.setData([], [])
        self._set_height_ticks(prof, y)
        self.set_parcel(parcel, prof)

    def set_comparisons(self, items):
        """items: list of (profile, color); T is drawn solid and Td dashed, both in that color."""
        while len(self.comp_curves) < len(items):
            self.comp_curves.append((self.plot(), self.plot()))
        for i, (t_item, td_item) in enumerate(self.comp_curves):
            if i >= len(items):
                t_item.setData([], [])
                td_item.setData([], [])
                continue
            prof, color = items[i]
            x, y = skew_xy(prof.t, prof.p)
            xd, _ = skew_xy(prof.td, prof.p)
            t_item.setPen(pg.mkPen(color, width=1.8))
            td_item.setPen(pg.mkPen(color, width=1.8, style=QtCore.Qt.PenStyle.DashLine))
            t_item.setData(x, y)
            td_item.setData(xd, y)

    def set_parcel(self, parcel, prof=None):
        if parcel is None:
            self.parcel.setData([], [])
            self.parcel_marks.setData([], [])
            self.parcel_axis.set_levels([])
            self.parcel_axis.setLabel("")
            return
        x, y = skew_xy(parcel.t, parcel.p)
        self.parcel.setData(x, y)
        levels, mx, my, brushes = [], [], [], []
        z_to_lnp = (prof.z, np.log(100000.0 / prof.p)) if prof is not None else None
        for label, z in (("LCL", parcel.z_lcl), ("LFC", parcel.lfc), ("EL", parcel.el)):
            if not np.isfinite(z):
                continue
            if label == "LCL":
                yy = float(np.log(100000.0 / parcel.p_lcl))
            elif z_to_lnp is not None:
                yy = float(np.interp(z, *z_to_lnp))
            else:
                continue
            p_hpa = 1000.0 * np.exp(-yy)
            levels.append((yy, label, f"{p_hpa:.0f}"))
            mx.append(float(np.interp(yy, y, x)))
            my.append(yy)
            brushes.append(pg.mkBrush(ParcelLevelAxis.COLORS[label]))
        self.parcel_marks.setData(mx, my, brush=brushes)
        self.parcel_axis.set_levels(levels)
        self.parcel_axis.setLabel(f"{parcel.name} parcel (hPa)", color=th.MUTED, **{"font-size": "11px"})

    def show_marker(self, kind, k):
        if kind is None or k < 0 or k >= len(self._y):
            self.marker.setData([], [])
            return
        x = self._xt[k] if kind == "t" else self._xtd[k]
        self.marker.setData([x], [self._y[k]])

    # interaction
    def _pick(self, pt):
        if self._y.size == 0:
            return None
        px, py = self.vb.pixel_size()
        dy = (self._y - pt.y()) / py
        dt = np.hypot((self._xt - pt.x()) / px, dy)
        dd = np.hypot((self._xtd - pt.x()) / px, dy)
        kt, kd = int(np.argmin(dt)), int(np.argmin(dd))
        if min(dt[kt], dd[kd]) > PICK_PX:
            return None
        return ("t", kt) if dt[kt] <= dd[kd] else ("td", kd)

    def _start(self, handle):
        self._drag_y = float(self._y[handle[1]])
        self.show_marker(*handle)
        self.dragStarted.emit(*handle)

    def _move(self, handle, pt):
        t_new = pt.x() - SKEW * self._drag_y + 273.15
        self.dragMoved.emit(handle[0], handle[1], float(t_new))

    def _mouse_moved(self, pos):
        if self._y.size == 0 or not self.vb.sceneBoundingRect().contains(pos):
            self.hover_line.hide()
            self.hovered.emit(-1)
            return
        y = self.vb.mapSceneToView(pos).y()
        k = int(np.argmin(np.abs(self._y - y)))
        self.hover_line.setPos(self._y[k])
        self.hover_line.show()
        self.hovered.emit(k)


# ---- hodograph ----------------------------------------------------------------
class HodographWidget(PlotView):
    dragStarted = QtCore.pyqtSignal(int)
    dragMoved = QtCore.pyqtSignal(int, float, float)
    dragFinished = QtCore.pyqtSignal(int)
    hovered = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        self.vb = EditViewBox()
        super().__init__(parent, viewBox=self.vb)
        _style_plot(self)
        pi = self.getPlotItem()
        _style_axis(pi.getAxis("bottom"), "u (m/s)")
        _style_axis(pi.getAxis("left"), "v (m/s)")
        self.vb.setAspectLocked(True)
        ang = np.linspace(0, 2 * np.pi, 181)
        for r in range(10, 81, 10):
            self.plot(r * np.cos(ang), r * np.sin(ang), pen=pg.mkPen(th.HODO_RINGS, width=1))
            lab = pg.TextItem(str(r), color=th.FAINT, anchor=(0.5, 0))
            lab.setFont(_font(10))
            lab.setPos(r * 0.7071, -r * 0.7071)
            self.addItem(lab)
        self.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(th.AXIS)))
        self.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(th.AXIS)))

        self.orig_bands = [self.plot(pen=pg.mkPen(_faded(c), width=2.6)) for c in th.HODO_BANDS]
        self.bands = [self.plot(pen=pg.mkPen(c, width=2.6)) for c in th.HODO_BANDS]
        self.inflow = self.plot(pen=pg.mkPen(th.MUTED, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.storms = pg.ScatterPlotItem(size=8, pen=pg.mkPen(th.MUTED, width=1.2), brush=None)
        self.addItem(self.storms)
        self.storm_sel = pg.ScatterPlotItem(size=12, symbol="x", pen=None, brush=th.TEXT)
        self.addItem(self.storm_sel)
        self.labels = []
        self.marker = pg.ScatterPlotItem(size=12, pen=pg.mkPen(th.MARKER, width=1.5), brush=None)
        self.addItem(self.marker)
        for item in (*self.orig_bands, *self.bands, self.inflow, self.storms, self.storm_sel, self.marker):
            item.setZValue(10)
        self.comp_curves = []

        self.vb.picker = self._pick
        self.vb.on_start = lambda k: (self.show_marker(k), self.dragStarted.emit(k))
        self.vb.on_move = lambda k, pt: self.dragMoved.emit(k, float(pt.x()), float(pt.y()))
        self.vb.on_finish = lambda k: self.dragFinished.emit(k)
        self.scene().sigMouseMoved.connect(self._mouse_moved)
        self._u = np.array([])
        self._v = np.array([])
        self._fit_done = False

    def reset_view(self):
        if self._u.size:
            uu = np.concatenate([self._u, [0.0]])
            vv = np.concatenate([self._v, [0.0]])
            self.vb.setRange(xRange=(np.nanmin(uu) - 8, np.nanmax(uu) + 8),
                             yRange=(np.nanmin(vv) - 8, np.nanmax(vv) + 8))

    def set_data(self, z, u, v, orig=None, storms=None, selected=None, show_orig=True):
        self._z, self._u, self._v = z, u, v
        for item, idx in zip(self.bands, _band_indices(z)):
            item.setData(u[idx], v[idx])
        if orig is not None and show_orig:
            oz, ou, ov = orig
            for item, idx in zip(self.orig_bands, _band_indices(oz)):
                item.setData(ou[idx], ov[idx])
        else:
            for item in self.orig_bands:
                item.setData([], [])
        for lab in self.labels:
            self.removeItem(lab)
        self.labels = []
        for km in (1, 3, 6, 9):
            if km * 1000.0 <= z[-1]:
                lu, lv = np.interp(km * 1000.0, z, u), np.interp(km * 1000.0, z, v)
                lab = pg.TextItem(str(km), color=th.AXIS_TEXT, anchor=(-0.2, 1.1))
                lab.setFont(_font(11))
                lab.setPos(float(lu), float(lv))
                self.addItem(lab)
                self.labels.append(lab)
        if storms:
            names = list(storms)
            self.storms.setData([storms[n][0] for n in names], [storms[n][1] for n in names])
            for n in names:
                lab = pg.TextItem(n, color=th.MUTED, anchor=(-0.25, -0.1))
                lab.setFont(_font(10))
                lab.setPos(float(storms[n][0]), float(storms[n][1]))
                self.addItem(lab)
                self.labels.append(lab)
        if selected is not None:
            self.storm_sel.setData([selected[0]], [selected[1]])
            self.inflow.setData([selected[0], u[0]], [selected[1], v[0]])
        if not self._fit_done:
            self.reset_view()
            self._fit_done = True

    def set_comparisons(self, items):
        """items: list of (u, v, color), one dashed line per comparison sounding."""
        while len(self.comp_curves) < len(items):
            self.comp_curves.append(self.plot())
        for i, item in enumerate(self.comp_curves):
            if i >= len(items):
                item.setData([], [])
                continue
            u, v, color = items[i]
            # dashed, so comparisons stand apart from the edited profile's height-band colors
            item.setPen(pg.mkPen(color, width=1.8, style=QtCore.Qt.PenStyle.DashLine))
            item.setData(u, v)

    def show_marker(self, k):
        if k is None or k < 1 or k >= len(self._u):
            self.marker.setData([], [])
            return
        self.marker.setData([self._u[k]], [self._v[k]])

    def _pick(self, pt):
        if self._u.size < 2:
            return None
        px, py = self.vb.pixel_size()
        d = np.hypot((self._u[1:] - pt.x()) / px, (self._v[1:] - pt.y()) / py)
        k = int(np.argmin(d))
        return k + 1 if d[k] <= PICK_PX else None

    def _mouse_moved(self, pos):
        if self._u.size < 2 or not self.vb.sceneBoundingRect().contains(pos):
            self.hovered.emit(-1)
            return
        k = self._pick(self.vb.mapSceneToView(pos))
        self.hovered.emit(-1 if k is None else k)

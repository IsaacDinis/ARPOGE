import sys
import numpy as np
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QRubberBand, QPushButton,QSpinBox,QTextEdit
from PyQt6.QtCharts import QChart, QChartView, QLineSeries,QLogValueAxis, QValueAxis, QBarSeries,QBarSet,QScatterSeries
from PyQt6.QtCore import Qt, QRectF , QTimer, QPointF,QThread, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from pyqtgraph import ImageView
import pyqtgraph as pg
import threading
import dao
import matplotlib.pyplot as plt
import time
import toml
from astropy.io import fits
import subprocess
import os
import signal
import control as ct
import ctypes
import datetime
from dd_utils import *
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0
this_script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_script_dir, "../data/")
def update_histogram_axis_to_log(view: pg.ImageView, original_data: np.ndarray, eps: float):
    hist_widget = view.getHistogramWidget()

    # Ensure the color gradient is visible
    hist_widget.show()

    # Manually set log ticks
    ticks = []
    log_min = np.log10(np.maximum(original_data.min(), eps))
    log_max = np.log10(np.maximum(original_data.max(), eps))

    # Choose positions and labels
    exponents = np.arange(np.floor(log_min), np.ceil(log_max) + 1)
    for exp in exponents:
        pos = exp  # this is in log10 scale
        label = f"1e{int(exp)}"
        ticks.append((pos, label))

    # Set ticks on the gradient's axis
    hist_widget.axis.setTicks([ticks])

# ps -fA | grep python
# /usr/lib/qt5/bin/designer

def matplotlib_cmap_to_lut(name: str, n: int = 256):
    cmap = plt.get_cmap(name)
    colors = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.ubyte)
    return colors

def create_image_view_in_placeholder(parent, name: str, cmap: pg.ColorMap) -> ImageView:
    placeholder = parent.findChild(QWidget, name)
    layout = QVBoxLayout(placeholder)
    layout.setContentsMargins(0, 0, 0, 0)

    view = ImageView()
    layout.addWidget(view)

    # Hide buttons and color bar, set colormap
    if hasattr(view, 'ui'):
        view.ui.roiBtn.hide()
        view.ui.menuBtn.hide()
        view.ui.histogram.gradient.hide()
        view.ui.histogram.hide() # removes scale
        view.getHistogramWidget().gradient.setColorMap(cmap)

    return view

class ProcessThread(QThread):
    output_received = pyqtSignal(str)  # Signal for stdout/stderr output
    process_finished = pyqtSignal(int)  # Signal when process ends
    process_started = pyqtSignal(int)  # Signal for process PID

    def __init__(self, script_path):
        super().__init__()
        self.script_path = script_path
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(
                ["python", "-u", self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setpgrp
            )

            self.process_started.emit(self.process.pid)  # Émettre PID

            for line in self.process.stdout:
                self.output_received.emit(line.strip())

            for line in self.process.stderr:
                self.output_received.emit(f"ERROR: {line.strip()}")

            self.process.wait()
            self.process_finished.emit(self.process.returncode)

        except Exception as e:
            self.output_received.emit(f"Thread Error: {str(e)}")

        finally:
            self.process = None  # Éviter les références fantômes

    def stop(self):
        if self.process:
            os.killpg(os.getpgid(self.process.pid), 15)  # Envoie SIGTERM
            self.process.wait(timeout=5)  # Attendre que le processus termine proprement
            if self.process is not None and self.process.poll() is None:  # Si encore actif, tuer brutalement
                os.killpg(os.getpgid(self.process.pid), 9)  # SIGKILL en dernier recour


class ProcessManager(QWidget):
    def __init__(self, script_path, start_button=None, stop_button=None, log_output=None):
        super().__init__()
        self.script_path = script_path
        self.process_thread = None

        self.start_button = start_button
        self.stop_button = stop_button
        self.log_output = log_output

        if self.start_button:
            self.start_button.clicked.connect(self.start_process)

        if self.stop_button:
            self.stop_button.clicked.connect(self.stop_process)
            self.stop_button.setEnabled(False)

    def start_process(self):
        if not self.process_thread:
            self.process_thread = ProcessThread(self.script_path)
            self.process_thread.output_received.connect(self.log_output.append)  # Connect stdout/stderr to UI
            self.process_thread.process_finished.connect(self.process_completed)
            self.process_thread.process_started.connect(
                lambda pid: self.log_output.append(f"Started process with PID: {pid}")
            )

            self.process_thread.start()

            self.start_button.setEnabled(False)
            if self.stop_button:
                self.stop_button.setEnabled(True)

    def process_completed(self, exit_code):
        self.log_output.append(f"Process finished with exit code {exit_code}.")
        if self.process_thread:
            self.process_thread.quit()
            self.process_thread.wait()
            self.process_thread = None
        self.start_button.setEnabled(True)
        if self.stop_button:
            self.stop_button.setEnabled(False)

    def stop_process(self):
        if self.process_thread:
            self.process_thread.stop()
            self.process_thread.wait()
            self.log_output.append("Process terminated.")
            self.process_thread = None
            self.start_button.setEnabled(True)
            if self.stop_button:
                self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        if self.process_thread and self.process_thread.isRunning():
            self.stop_process()
            self.process_thread.wait()
        event.accept()

class CustomChartView(QChartView):
    def __init__(self, parent=None, xlabel="x", ylabel="y", n_lines = 1, plot_type="line"):
        chart = QChart()
        super().__init__(chart, parent)
        self.setRenderHint(self.renderHints().Antialiasing)
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.series_list = []
        self.x = None
        self.y_ranges = []
        self.plot_type = plot_type
        self.y_scale_type = "linear"

        self.x_axis = QValueAxis()
        self.y_axis = QValueAxis()

        self.chart().addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)
        self.chart().addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)

        self.set_axis_labels(xlabel, ylabel)

        self.flag_reset_zoom = True

        for _ in range(n_lines):
            series = QLineSeries()
            self.chart().addSeries(series)
            series.attachAxis(self.x_axis)
            series.attachAxis(self.y_axis)
            self.series_list.append(series)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.lastMousePos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.origin = event.position().toPoint()
            self.rubberBand.setGeometry(self.origin.x(), self.origin.y(), 0, 0)
            self.rubberBand.setStyleSheet("background-color: rgba(200, 200, 200, 100);")
            self.rubberBand.show()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.position() - self.lastMousePos
            self.chart().scroll(-delta.x(), delta.y())
            self.lastMousePos = event.position()
            self.flag_reset_zoom = False
        elif event.buttons() == Qt.MouseButton.MiddleButton and self.origin:
            current_pos = event.position().toPoint()
            x = min(self.origin.x(), current_pos.x())
            y = min(self.origin.y(), current_pos.y())
            width = abs(current_pos.x() - self.origin.x())
            height = abs(current_pos.y() - self.origin.y())
            self.rubberBand.setGeometry(x, y, width, height)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.MiddleButton and self.origin:
            rect = self.rubberBand.geometry()
            if rect.width() > 10 and rect.height() > 10:
                self.chart().zoomIn(QRectF(rect))
                self.flag_reset_zoom = False
            self.rubberBand.hide()

    def wheelEvent(self, event: QWheelEvent):
        zoomFactor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.chart().zoom(zoomFactor)
        self.flag_reset_zoom = False

    def set_axis_labels(self, xlabel, ylabel):
        self.x_axis.setTitleText(xlabel)
        self.y_axis.setTitleText(ylabel)

    def draw(self, datasets):
        if not all(isinstance(ds, (list, tuple)) and len(ds) == 2 for ds in datasets):
            raise ValueError("Datasets must be a list of (x, y) tuples or lists.")

        self.x = datasets[0][0] if datasets else None
        self.y_ranges = []

        for i, (x, y) in enumerate(datasets):
            if self.plot_type == "stem":
                lines = []
                for px, py in zip(x.squeeze(), y.squeeze()):
                    lines.append(QPointF(px, 0))  # Start from zero
                    lines.append(QPointF(px, py))  # Go to actual y-value
                    lines.append(QPointF(float('nan'), float('nan')))

                self.series_list[i].replace(lines)  # Update line series

            else:
                points = [QPointF(px, py) for px, py in zip(x.squeeze(), y.squeeze())]
                self.series_list[i].replace(points)

            self.y_ranges.append((min(y), max(y)))

        if self.flag_reset_zoom:
            self.resetZoom()

    def set_log_scale(self, x_log=True, y_log=True):
        x_label = self.x_axis.titleText()
        y_label = self.y_axis.titleText()
        self.y_scale_type = "log"
        self.chart().removeAxis(self.x_axis)
        self.chart().removeAxis(self.y_axis)

        self.x_axis = QLogValueAxis() if x_log else QValueAxis()
        self.y_axis = QLogValueAxis() if y_log else QValueAxis()

        self.x_axis.setTitleText(x_label)
        self.y_axis.setTitleText(y_label)

        self.chart().addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)
        self.chart().addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)

        for series in self.series_list:
            series.attachAxis(self.x_axis)
            series.attachAxis(self.y_axis)

    def set_legend(self,legend):
        for i in range(len(self.series_list)):
            self.series_list[i].setName(legend[i])


    def resetZoom(self):
        axes = self.chart().axes()
        if axes:
            x_axis = axes[0]
            y_axis = axes[1]

            if self.x is not None and self.y_ranges:
                xRange = np.min(self.x), (np.max(self.x))
                yMin = min(y_range[0] for y_range in self.y_ranges)
                yMax = max(y_range[1] for y_range in self.y_ranges)
                y_abs_max = max([np.abs(yMin),yMax])
                x_axis.setRange(xRange[0], xRange[1])
                if self.y_scale_type == "linear":
                    y_axis.setRange(-y_abs_max, y_abs_max)
                else :
                    if yMin.ndim > 0:
                        yMin = yMin[0]
                    if yMax.ndim > 0:
                        yMax = yMax[0]
                    y_axis.setRange(yMin, yMax)

        self.flag_reset_zoom = True
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi("gui.ui", self)
        self.sem_nb = 9
        self.init_images()
        self.init_vector_plots()
        self.init_shm()
        self.init_process()
        self.init_spinboxes()
        self.view_update_timer = QTimer()
        self.view_update_timer.timeout.connect(self.update_images)
        self.view_update_timer.timeout.connect(self.update_fft_wiew)
        self.view_update_timer.timeout.connect(self.update_time_wiew)
        self.view_update_timer.timeout.connect(self.update_modes_amp_wiew)
        self.view_update_timer.start(100) # ms 

        print("init done")
    def init_spinboxes(self):
        self.gain_spinbox.valueChanged.connect(self.gain_changed)
        self.gain_changed(self.gain_spinbox.value())


        self.n_modes_spinbox.valueChanged.connect(self.n_modes_changed)
        self.n_modes_changed(self.n_modes_spinbox.value())

        self.record_time_spinbox.valueChanged.connect(self.record_time_changed)
        self.record_time_changed(self.record_time_spinbox.value())
        
        self.closed_loop_checkbox.stateChanged.connect(self.closed_loop_check)
        self.save_slopes_checkbox.stateChanged.connect(self.save_slopes_check)
        self.reset_state_mat_button.clicked.connect(self.reset_state_mat)

        self.save_flat_button.clicked.connect(self.save_flat)
        self.load_flat_button.clicked.connect(self.load_flat)
        self.reset_flat_button.clicked.connect(self.reset_flat)
        self.save_latency_button.clicked.connect(self.save_latency)
        self.load_latency_button.clicked.connect(self.load_latency)
        self.nico_button.clicked.connect(self.start_record_and_close_loop)

        self.controller_select_dial.valueChanged.connect(self.update_controller_select)
        self.update_controller_select(self.controller_select_dial.value())

        self.pyramid_select_dial.valueChanged.connect(self.update_pyramid_select)
        self.update_pyramid_select(self.pyramid_select_dial.value())

    def init_process(self):

        self.record_process = ProcessManager("recorder.py",self.start_record_button, None, self.record_output)
        self.pol_reconstructor_process = ProcessManager("pol_reconstructor.py",self.start_pol_reconstructor_button, self.stop_pol_reconstructor_button, self.pol_reconstructor_output)
        self.freq_mag_estimator_process = ProcessManager("freq_mag_estimator.py",self.start_freq_mag_estimator_button, self.stop_freq_mag_estimator_button, self.freq_mag_estimator_output)
        self.identify_latency_frequency_process = ProcessManager("identify_latency_frequency.py",self.start_latency_identification_button, None, self.latency_identification_output)

    def init_shm(self):

        
        with open(os.path.join(this_script_dir,'../config/config.toml'), 'r') as f:
            config = toml.load(f)
        with open(os.path.join(this_script_dir,'../config/shm_path.toml'), 'r') as f:
            shm_path = toml.load(f)

        self.pyr_3_shm                     = dao.shm(shm_path['HW']['pixels_3sided'])             
        self.pyr_3_masked_shm                     = dao.shm(shm_path['HW']['pixels_masked_3sided'])


        self.modes_in_fft_shm = dao.shm(shm_path['frequency_domain_buff']['modes_in_fft'])
        self.modes_out_fft_shm = dao.shm(shm_path['frequency_domain_buff']['modes_out_fft'])
        self.pol_fft_shm = dao.shm(shm_path['frequency_domain_buff']['pol_fft'])
        self.f_shm = dao.shm(shm_path['frequency_domain_buff']['f'])

        self.modes_in_buf_shm = dao.shm(shm_path['time_domain_buff']['modes_in_buf'])
        self.modes_out_buf_shm = dao.shm(shm_path['time_domain_buff']['modes_out_buf']) 
        self.pol_buf_shm = dao.shm(shm_path['time_domain_buff']['pol_buf'])
        self.t_shm = dao.shm(shm_path['time_domain_buff']['t'])    

        self.closed_loop_state_flag_shm = dao.shm(shm_path['settings']['closed_loop_state_flag']) 
        self.save_slopes_state_flag_shm = dao.shm(shm_path['settings']['save_slopes_state_flag']) 
        self.n_modes_dd_shm = dao.shm(shm_path['settings']['n_modes_dd']) 
        self.n_modes_controlled_shm = dao.shm(shm_path['settings']['n_modes_controlled'])
        self.dd_update_rate_shm = dao.shm(shm_path['settings']['dd_update_rate'])
        self.dd_order_shm = dao.shm(shm_path['settings']['dd_order'])  


        self.K_mat_int_shm = dao.shm(shm_path['K']['K_mat_int']) 
        self.K_mat_dd_shm = dao.shm(shm_path['K']['K_mat_dd']) 
        self.K_mat_omgi_shm = dao.shm(shm_path['K']['K_mat_omgi']) 

        

        self.latency_shm = dao.shm(shm_path['G']['latency']) 
        self.fs_shm = dao.shm(shm_path['G']['fs']) 
        self.delay_shm = dao.shm(shm_path['G']['delay']) 


        self.controller_select_shm = dao.shm(shm_path['settings']['controller_select']) 
        self.gain_margin_shm = dao.shm(shm_path['settings']['gain_margin']) 
        self.record_time_shm = dao.shm(shm_path['settings']['record_time'])
        self.n_fft_shm = dao.shm(shm_path['settings']['n_fft']) 

        
        self.S_dd_shm = dao.shm(shm_path['S']['S_dd']) 
        self.S_omgi_shm = dao.shm(shm_path['S']['S_omgi']) 
        self.S_int_shm = dao.shm(shm_path['S']['S_int']) 
        self.f_opti_shm = dao.shm(shm_path['S']['f_opti']) 

        self.reset_flag_shm = dao.shm(shm_path['event_flag']['reset_flag']) 
        self.K_mat_flag_shm = dao.shm(shm_path['event_flag']['K_mat_flag']) 
        self.pyramid_flag_shm = dao.shm(shm_path['event_flag']['pyramid_flag']) 

        self.dm_shm = dao.shm(shm_path['HW']['dm']) 
        self.flat_dm_shm = dao.shm(shm_path['HW']['flat_dm']) 
        self.pyramid_select_shm = dao.shm(shm_path['settings']['pyramid_select']) 

    def init_vector_plots(self):
        self.time_view = CustomChartView(self.time_widget,"t [s]","amplitude", n_lines = 3)
        layout = QVBoxLayout(self.time_widget)
        layout.addWidget(self.time_view)
        self.time_view.set_legend(["pol","command","res"])
        self.reset_time_view_button.clicked.connect(self.time_view.resetZoom)

        ##------------------  FFT VIEW ------------------------##
        self.fft_view = CustomChartView(self.fft_widget,"f [Hz]","amplitude",n_lines = 4)
        layout = QVBoxLayout(self.fft_widget)
        layout.addWidget(self.fft_view)
        self.fft_view.set_log_scale()
        self.fft_view.set_legend(["pol","command","res","sensitivity"])
        self.reset_fft_view_button.clicked.connect(self.fft_view.resetZoom)

        ##------------------  MODE VIEW ------------------------##
        self.modes_amp_view = CustomChartView(self.modes_amp_widget,"mode","amplitude", plot_type = "stem")
        layout = QVBoxLayout(self.modes_amp_widget)
        layout.addWidget(self.modes_amp_view)
        self.modes_amp_view.set_legend(["res"])
        self.reset_modes_amp_view_button.clicked.connect(self.modes_amp_view.resetZoom)

    def init_images(self):
        # Generate LUT and ColorMap once
        lut = matplotlib_cmap_to_lut('viridis')
        cmap = pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut)

        # All image views and their corresponding placeholder names
        placeholders = {
            "pyr_3_widget": "pyr_3_view",
            "pyr_3_masked_widget": "pyr_3_masked_view",
        }

        for placeholder_name, attr_name in placeholders.items():
            view = create_image_view_in_placeholder(self, placeholder_name, cmap)
            setattr(self, attr_name, view)

    def update_images(self):
        self.pyr_3_view.setImage(self.pyr_3_shm.get_data(check=False,semNb=self.sem_nb), autoLevels=(self.autoscale_pyr_3_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)
        self.pyr_3_masked_view.setImage(self.pyr_3_masked_shm.get_data(check=False,semNb=self.sem_nb), autoLevels=(self.autoscale_pyr_3_masked_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)
  

    def update_fft_wiew(self):
        f =  self.f_shm.get_data(check=False, semNb=self.sem_nb)
        pol_fft = self.pol_fft_shm.get_data(check=False, semNb=self.sem_nb)
        modes_int_fft = self.modes_in_fft_shm.get_data(check=False, semNb=self.sem_nb)
        modes_out_fft = self.modes_out_fft_shm.get_data(check=False, semNb=self.sem_nb)
        n_fft = self.n_fft_spinbox.value()
        mode_n = self.mode_select_spinbox.value()
        f_opti = self.f_opti_shm.get_data(check=False, semNb=self.sem_nb)
        
        match self.controller_select_dial.value():
            case 0:
                S = self.S_int_shm.get_data(check=False, semNb=self.sem_nb)
            case 1:
                S = self.S_dd_shm.get_data(check=False, semNb=self.sem_nb)[:n_fft,mode_n]
            case 2:
                S = self.S_omgi_shm.get_data(check=False, semNb=self.sem_nb)[:n_fft,mode_n]
        val = np.interp(f_opti[int(n_fft/2)].squeeze(), f.squeeze(), pol_fft[:,mode_n])
        S *= val/S[int(n_fft/2)]
        self.fft_view.draw([(f, pol_fft[:,mode_n]), (f, modes_out_fft[:,mode_n]), (f, modes_int_fft[:,mode_n]),(f_opti[:n_fft],S)])

    def update_time_wiew(self):
        t =  self.t_shm.get_data(check=False, semNb=self.sem_nb)
        pol = self.pol_buf_shm.get_data(check=False, semNb=self.sem_nb)
        res = self.modes_in_buf_shm.get_data(check=False, semNb=self.sem_nb)
        command = self.modes_out_buf_shm.get_data(check=False, semNb=self.sem_nb) 
        mode_n = self.mode_select_spinbox.value()
        self.time_view.draw([(t, pol[:,mode_n]), (t, command[:,mode_n]), (t, res[:,mode_n])])

    def update_modes_amp_wiew(self):
        res = self.modes_in_buf_shm.get_data(check=False, semNb=self.sem_nb)
        if(self.square_res_checkbox.checkState()==Qt.CheckState.Checked):
            res = np.sqrt(np.square(res))
        n_modes = res.shape[1]
        self.modes_amp_view.draw([(np.arange(n_modes), res[-1,:])])

    def gain_changed(self,value):
        # self.gain_shm.set_data(np.array([[value]],np.float32))
        K_mat_int = self.K_mat_int_shm.get_data(check=False, semNb=self.sem_nb)
        K_mat_int[0,:] = value
        self.K_mat_int_shm.set_data(K_mat_int)
        fs = self.fs_shm.get_data(check=False, semNb=self.sem_nb)[0][0]
        delay = self.delay_shm.get_data(check=False, semNb=self.sem_nb)[0][0]
        K = ct.tf(np.array([value, 0]), np.array([1, -0.99]), 1 /fs)
        G = G_tf(delay,fs)
        S = (1+G*K)
        n_fft = self.n_fft_spinbox.value()
        f =  self.f_opti_shm.get_data(check=False, semNb=self.sem_nb)[:n_fft]
        S_resp = np.abs(freqresp(S, 2*np.pi*f))
        self.S_int_shm.set_data(S_resp.astype(np.float32))


    def closed_loop_check(self,state):
        if state == Qt.CheckState.Checked.value:
            self.closed_loop_state_flag_shm.set_data(np.array([[1]],np.uint32))
        elif state == Qt.CheckState.Unchecked.value:
            self.closed_loop_state_flag_shm.set_data(np.array([[0]],np.uint32))

    def save_slopes_check(self,state):
        if state == Qt.CheckState.Checked.value:
            self.save_slopes_state_flag_shm.set_data(np.array([[1]],np.uint32))
        elif state == Qt.CheckState.Unchecked.value:
            self.save_slopes_state_flag_shm.set_data(np.array([[0]],np.uint32))

    def order_dd_changed(self,value):
        self.dd_order_shm.set_data(np.array([[value]],np.uint32))
        self.reset_dd_controller()

    def n_modes_changed(self,value):
        self.n_modes_controlled_shm.set_data(np.array([[value]],np.uint32))

    def reset_state_mat(self):
        self.reset_flag_shm.set_data(np.ones((1,1),dtype = np.uint32))

    def record_time_changed(self,value):
        self.record_time_shm.set_data(np.array([[value]],np.float32))

    def save_flat(self):
        try:
            calib_flat = fits.getdata(os.path.join(data_dir, "dm_flat_papy.fits"))[:,np.newaxis]

        except FileNotFoundError:
            calib_flat = 0.
            print("File: dm_flat_papy.fits not found")
        calib_flat = 0.
        #flat = self.dm_shm.get_data(check=False, semNb=self.sem_nb)
        flat = (np.mean([self.dm_shm.get_data(check=False, semNb=self.sem_nb) for i in range(10)], axis=0))
        fits.writeto(os.path.join(data_dir, "flat_cl.fits"),flat, overwrite = True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fits.writeto(os.path.join(data_dir, f'flat_{timestamp}.fits'),flat, overwrite = True)
        #self.flat_dm_shm.set_data((flat+calib_flat.astype(np.float32)).astype(np.float32))
        self.flat_dm_shm.set_data((flat).astype(np.float32))

    def load_flat(self):
        try:
            calib_flat = fits.getdata(os.path.join(data_dir, "dm_flat_papy.fits"))[:,np.newaxis]

        except FileNotFoundError:
            calib_flat = 0.
            print("File: dm_flat_papy.fits not found")
        calib_flat = 0.
        try:
            flat = fits.getdata(os.path.join(data_dir, "flat_cl.fits"))
            #self.flat_dm_shm.set_data((flat+calib_flat.astype(np.float32)).astype(np.float32))
            self.flat_dm_shm.set_data((flat).astype(np.float32))
        except FileNotFoundError:
            print("File: 'flat.fits' not found")

    def reset_flat(self):
        try:
            calib_flat = fits.getdata(os.path.join(data_dir, "dm_flat_papy.fits"))[:,np.newaxis]
        except FileNotFoundError:
            calib_flat = 0.
            print("File: dm_flat_papy.fits not found")
        calib_flat = 0.
        flat = self.dm_shm.get_data(check=False, semNb=self.sem_nb)*0
        # fits.writeto(os.path.join(data_dir, "flat_cl.fits"),flat, overwrite = True)
        #self.flat_dm_shm.set_data((flat+calib_flat.astype(np.float32)).astype(np.float32))
        self.flat_dm_shm.set_data((flat).astype(np.float32))

    def save_latency(self):
        latency = self.latency_shm.get_data(check=False, semNb=self.sem_nb)
        fs = self.fs_shm.get_data(check=False, semNb=self.sem_nb)
        delay = self.delay_shm.get_data(check=False, semNb=self.sem_nb)
        t = self.t_shm.get_data(check=False, semNb=self.sem_nb)
        fits.writeto(os.path.join(data_dir, "latency.fits"), latency, overwrite = True)
        fits.writeto(os.path.join(data_dir, "fs.fits"),fs, overwrite = True)
        fits.writeto(os.path.join(data_dir, "delay.fits"),delay, overwrite = True)
        fits.writeto(os.path.join(data_dir, "t.fits"),t, overwrite = True)
        
    def load_latency(self):
        latency = fits.getdata(os.path.join(data_dir, "latency.fits"))
        fs = fits.getdata(os.path.join(data_dir, "fs.fits"))
        delay = fits.getdata(os.path.join(data_dir, "delay.fits"))
        t = fits.getdata(os.path.join(data_dir, "t.fits"))
        self.latency_shm.set_data(latency.astype(np.float32))
        self.fs_shm.set_data(fs.astype(np.float32))
        self.delay_shm.set_data(delay.astype(np.float32))
        self.t_shm.set_data(t.astype(np.float32))
        
    def update_controller_select(self, value):
        self.controller_select_shm.set_data(np.array([[value]],np.uint32))

    def reset_dd_controller(self):
        K_mat_int = self.K_mat_int_shm.get_data(check=False, semNb=self.sem_nb)
        self.K_mat_dd_shm.set_data(K_mat_int)
        self.K_mat_omgi_shm.set_data(K_mat_int)
        self.K_mat_flag_shm.set_data(np.ones((1,1),dtype = np.uint32))

    def start_record_and_close_loop(self):
        self.record_process.start_process()
        self.closed_loop_checkbox.setChecked(True)
        self.closed_loop_state_flag_shm.set_data(np.array([[1]],np.uint32))

    def update_rate_dd_changed(self,value):
        if value:
            self.optimization_dd_timer.start(int(value*1e3))
        else:
            self.optimization_dd_timer.stop()

    def update_rate_omgi_changed(self,value):
        if value:
            self.optimization_omgi_timer.start(int(value*1e3))
        else:
            self.optimization_omgi_timer.stop()  
    def n_fft_changed(self,value):
        self.n_fft_shm.set_data(np.array([[value]],np.uint32))

    def gain_margin_changed(self,value):
        self.gain_margin_shm.set_data(np.array([[value]],np.float32))

    def update_pyramid_select(self, value):
        self.pyramid_select_shm.set_data(np.array([[value]],np.uint32))

    def closeEvent(self, event):
        self.record_process.stop_process()
        self.pol_reconstructor_process.stop_process()
        self.freq_mag_estimator_process.stop_process()
        self.view_update_timer.stop()
        self.identify_latency_frequency_process.stop_process()

        print("All processes and timers stopped")
        event.accept()

def handle_sigint(signum, frame):
    """Handle Ctrl+C and close the GUI properly"""
    print("SIGINT received, closing application...")
    app.quit()  # Close the application gracefully

if __name__ == "__main__":

    # subprocess.run(["python", "init_control_shm.py"])

    # Launch the GUI
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, handle_sigint)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


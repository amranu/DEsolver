import sys
import configparser
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QGridLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QComboBox, QSizePolicy, QSlider, QFileDialog, QGroupBox, QDialog)
from PyQt5.QtCore import Qt
from equation_canvas import CombinedEquationPlaneWidget
from settings_dialog import SettingsDialog

class MainWindow(QMainWindow):
    CONFIG_FILE = 'settings.ini'

    def __init__(self, nx, ny, dx, dy, alpha, c, nu, dt, simulation_time, initial_temp, initial_disp, initial_potential, initial_source, hbar, m, rho):
        super().__init__()
        self.setWindowTitle("2D Differential Equation Solver")
        self.nx, self.ny, self.dx, self.dy = nx, ny, dx, dy
        self.alpha, self.c, self.nu, self.dt, self.simulation_time = alpha, c, nu, dt, simulation_time
        self.initial_temp, self.initial_disp, self.initial_potential, self.initial_source = initial_temp, initial_disp, initial_potential, initial_source
        self.hbar, self.m, self.rho = hbar, m, rho
        self.initial_psi_real, self.initial_psi_imag = initial_disp, 0.0  # Assuming initial psi_imag is 0 for simplicity
        self.initial_u, self.initial_v = 0.0, 0.0  # Assuming initial velocities for Navier-Stokes
        self.equation_type = 'Heat Equation'

        self.load_settings()

        self.canvas = CombinedEquationPlaneWidget(self)

        self.equation_combo = QComboBox()
        self.equation_combo.addItems(["Heat Equation", "Wave Equation", "Laplace Equation", "Poisson Equation", "Schrödinger Equation", "Navier-Stokes Equation"])
        self.equation_combo.currentTextChanged.connect(self.on_equation_change)

        self.alpha_label = QLabel("Alpha (Heat):")
        self.alpha_input = QLineEdit(str(self.alpha))

        self.c_label = QLabel("Wave Speed (c):")
        self.c_input = QLineEdit(str(self.c))

        self.nu_label = QLabel("Viscosity (nu):")
        self.nu_input = QLineEdit(str(self.nu))

        self.dx_label = QLabel("dx:")
        self.dx_input = QLineEdit(str(self.dx))

        self.dy_label = QLabel("dy:")
        self.dy_input = QLineEdit(str(self.dy))

        self.simulation_time_label = QLabel("Simulation Time:")
        self.simulation_time_input = QLineEdit(str(self.simulation_time))

        self.initial_temp_label = QLabel("Initial Temp:")
        self.initial_temp_input = QLineEdit(str(self.initial_temp))

        self.initial_disp_label = QLabel("Initial Disp:")
        self.initial_disp_input = QLineEdit(str(self.initial_disp))

        self.initial_potential_label = QLabel("Initial Potential:")
        self.initial_potential_input = QLineEdit(str(self.initial_potential))

        self.initial_source_label = QLabel("Initial Source:")
        self.initial_source_input = QLineEdit(str(self.initial_source))

        self.hbar_label = QLabel("hbar (Schrödinger):")
        self.hbar_input = QLineEdit(str(self.hbar))

        self.m_label = QLabel("Mass (Schrödinger):")
        self.m_input = QLineEdit(str(self.m))

        self.rho_label = QLabel("Density (rho):")
        self.rho_input = QLineEdit(str(self.rho))

        self.timer_label = QLabel("Time left: 0 seconds", alignment=Qt.AlignCenter)

        # Add sliders for parameters
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(1, 100)
        self.alpha_slider.setValue(int(self.alpha * 100))
        self.alpha_slider.valueChanged.connect(self.update_alpha)

        self.c_slider = QSlider(Qt.Horizontal)
        self.c_slider.setRange(1, 100)
        self.c_slider.setValue(int(self.c * 100))
        self.c_slider.valueChanged.connect(self.update_c)

        self.nu_slider = QSlider(Qt.Horizontal)
        self.nu_slider.setRange(1, 100)
        self.nu_slider.setValue(int(self.nu * 100))
        self.nu_slider.valueChanged.connect(self.update_nu)

        self.dx_slider = QSlider(Qt.Horizontal)
        self.dx_slider.setRange(1, 100)
        self.dx_slider.setValue(int(self.dx * 100))
        self.dx_slider.valueChanged.connect(self.update_dx)

        self.dy_slider = QSlider(Qt.Horizontal)
        self.dy_slider.setRange(1, 100)
        self.dy_slider.setValue(int(self.dy * 100))
        self.dy_slider.valueChanged.connect(self.update_dy)

        start_button = QPushButton("Start")
        start_button.clicked.connect(self.start_simulation)

        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self.stop_simulation)

        export_button = QPushButton("Export Data")
        export_button.clicked.connect(self.canvas.export_data)

        load_button = QPushButton("Load Data")
        load_button.clicked.connect(self.canvas.load_data)

        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.open_settings)

        # Group related controls together
        equation_group = QGroupBox("Equation Parameters")
        equation_layout = QGridLayout()
        equation_layout.addWidget(QLabel("Equation Type:"), 0, 0)
        equation_layout.addWidget(self.equation_combo, 0, 1, 1, 2)
        equation_layout.addWidget(self.alpha_label, 1, 0)
        equation_layout.addWidget(self.alpha_input, 1, 1)
        equation_layout.addWidget(self.alpha_slider, 1, 2)
        equation_layout.addWidget(self.c_label, 2, 0)
        equation_layout.addWidget(self.c_input, 2, 1)
        equation_layout.addWidget(self.c_slider, 2, 2)
        equation_layout.addWidget(self.nu_label, 3, 0)
        equation_layout.addWidget(self.nu_input, 3, 1)
        equation_layout.addWidget(self.nu_slider, 3, 2)
        equation_layout.addWidget(self.dx_label, 4, 0)
        equation_layout.addWidget(self.dx_input, 4, 1)
        equation_layout.addWidget(self.dx_slider, 4, 2)
        equation_layout.addWidget(self.dy_label, 5, 0)
        equation_layout.addWidget(self.dy_input, 5, 1)
        equation_layout.addWidget(self.dy_slider, 5, 2)
        equation_layout.addWidget(self.simulation_time_label, 6, 0)
        equation_layout.addWidget(self.simulation_time_input, 6, 1, 1, 2)
        equation_layout.addWidget(self.initial_temp_label, 7, 0)
        equation_layout.addWidget(self.initial_temp_input, 7, 1, 1, 2)
        equation_layout.addWidget(self.initial_disp_label, 8, 0)
        equation_layout.addWidget(self.initial_disp_input, 8, 1, 1, 2)
        equation_layout.addWidget(self.initial_potential_label, 9, 0)
        equation_layout.addWidget(self.initial_potential_input, 9, 1, 1, 2)
        equation_layout.addWidget(self.initial_source_label, 10, 0)
        equation_layout.addWidget(self.initial_source_input, 10, 1, 1, 2)
        equation_layout.addWidget(self.hbar_label, 11, 0)
        equation_layout.addWidget(self.hbar_input, 11, 1, 1, 2)
        equation_layout.addWidget(self.m_label, 12, 0)
        equation_layout.addWidget(self.m_input, 12, 1, 1, 2)
        equation_layout.addWidget(self.rho_label, 13, 0)
        equation_layout.addWidget(self.rho_input, 13, 1, 1, 2)
        equation_group.setLayout(equation_layout)

        param_container = QWidget()
        param_layout = QVBoxLayout()
        param_layout.addWidget(equation_group)
        param_layout.addStretch()
        param_container.setLayout(param_layout)
        param_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.timer_label)
        bottom_layout.addWidget(start_button)
        bottom_layout.addWidget(stop_button)
        bottom_layout.addWidget(export_button)
        bottom_layout.addWidget(load_button)
        bottom_layout.addWidget(settings_button)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(param_container)
        layout.addLayout(bottom_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.startTimer(16)
        self.on_equation_change('Heat Equation')

    def load_settings(self):
        config = configparser.ConfigParser()
        config.read(self.CONFIG_FILE)

        self.nx = config.getint('Grid', 'nx', fallback=50)
        self.ny = config.getint('Grid', 'ny', fallback=50)
        self.dx = config.getfloat('Grid', 'dx', fallback=1.0)
        self.dy = config.getfloat('Grid', 'dy', fallback=1.0)
        self.alpha = config.getfloat('HeatEquation', 'alpha', fallback=0.1)
        self.c = config.getfloat('WaveEquation', 'c', fallback=1.0)
        self.nu = config.getfloat('NavierStokes', 'nu', fallback=0.1)
        self.dt = config.getfloat('General', 'dt', fallback=0.01)
        self.simulation_time = config.getfloat('General', 'simulation_time', fallback=50)
        self.initial_temp = config.getfloat('InitialConditions', 'initial_temp', fallback=100.0)
        self.initial_disp = config.getfloat('InitialConditions', 'initial_disp', fallback=1.0)
        self.initial_potential = config.getfloat('InitialConditions', 'initial_potential', fallback=0.0)
        self.initial_source = config.getfloat('InitialConditions', 'initial_source', fallback=0.0)
        self.hbar = config.getfloat('SchrodingerEquation', 'hbar', fallback=1.0)
        self.m = config.getfloat('SchrodingerEquation', 'm', fallback=1.0)
        self.rho = config.getfloat('NavierStokes', 'rho', fallback=1.0)

    def save_settings(self):
        config = configparser.ConfigParser()

        config['Grid'] = {
            'nx': self.nx,
            'ny': self.ny,
            'dx': self.dx,
            'dy': self.dy
        }

        config['HeatEquation'] = {
            'alpha': self.alpha
        }

        config['WaveEquation'] = {
            'c': self.c
        }

        config['NavierStokes'] = {
            'nu': self.nu,
            'rho': self.rho
        }

        config['General'] = {
            'dt': self.dt,
            'simulation_time': self.simulation_time
        }

        config['InitialConditions'] = {
            'initial_temp': self.initial_temp,
            'initial_disp': self.initial_disp,
            'initial_potential': self.initial_potential,
            'initial_source': self.initial_source
        }

        config['SchrodingerEquation'] = {
            'hbar': self.hbar,
            'm': self.m
        }

        with open(self.CONFIG_FILE, 'w') as configfile:
            config.write(configfile)

    def closeEvent(self, event):
        self.save_settings()
        event.accept()

    def update_alpha(self, value):
        self.alpha = value / 100.0
        self.alpha_input.setText(str(self.alpha))

    def update_c(self, value):
        self.c = value / 100.0
        self.c_input.setText(str(self.c))

    def update_nu(self, value):
        self.nu = value / 100.0
        self.nu_input.setText(str(self.nu))

    def update_dx(self, value):
        self.dx = value / 100.0
        self.dx_input.setText(str(self.dx))

    def update_dy(self, value):
        self.dy = value / 100.0
        self.dy_input.setText(str(self.dy))

    def timerEvent(self, event):
        self.canvas.update_simulation()

    def on_equation_change(self, equation):
        self.equation_type = equation
        show_alpha = equation == 'Heat Equation'
        show_c = equation == 'Wave Equation'
        show_nu = equation == 'Navier-Stokes Equation'
        show_potential = equation == 'Laplace Equation' or equation == 'Poisson Equation'
        show_source = equation == 'Poisson Equation'
        show_schrodinger = equation == 'Schrödinger Equation'
        show_rho = equation == 'Navier-Stokes Equation'

        self.alpha_label.setVisible(show_alpha)
        self.alpha_input.setVisible(show_alpha)
        self.alpha_slider.setVisible(show_alpha)
        self.c_label.setVisible(show_c)
        self.c_input.setVisible(show_c)
        self.c_slider.setVisible(show_c)
        self.nu_label.setVisible(show_nu)
        self.nu_input.setVisible(show_nu)
        self.nu_slider.setVisible(show_nu)
        self.initial_temp_label.setVisible(show_alpha)
        self.initial_temp_input.setVisible(show_alpha)
        self.initial_disp_label.setVisible(show_c)
        self.initial_disp_input.setVisible(show_c)
        self.initial_potential_label.setVisible(show_potential)
        self.initial_potential_input.setVisible(show_potential)
        self.initial_source_label.setVisible(show_source)
        self.initial_source_input.setVisible(show_source)
        self.hbar_label.setVisible(show_schrodinger)
        self.hbar_input.setVisible(show_schrodinger)
        self.m_label.setVisible(show_schrodinger)
        self.m_input.setVisible(show_schrodinger)
        self.rho_label.setVisible(show_rho)
        self.rho_input.setVisible(show_rho)

        self.canvas.init_simulation_parameters()

    def start_simulation(self):
        try:
            if self.equation_type == 'Heat Equation':
                self.alpha = float(self.alpha_input.text())
                self.initial_temp = float(self.initial_temp_input.text())
            elif self.equation_type == 'Wave Equation':
                self.c = float(self.c_input.text())
                self.initial_disp = float(self.initial_disp_input.text())
            elif self.equation_type == 'Navier-Stokes Equation':
                self.nu = float(self.nu_input.text())
                self.rho = float(self.rho_input.text())
            elif self.equation_type == 'Laplace Equation':
                self.initial_potential = float(self.initial_potential_input.text())
            elif self.equation_type == 'Poisson Equation':
                self.initial_potential = float(self.initial_potential_input.text())
                self.initial_source = float(self.initial_source_input.text())
            elif self.equation_type == 'Schrödinger Equation':
                self.hbar = float(self.hbar_input.text())
                self.m = float(self.m_input.text())
                self.initial_psi_real = float(self.initial_disp_input.text())  # Using the initial_disp_input for initial psi_real
                self.initial_psi_imag = 0.0  # Assuming initial psi_imag is 0 for simplicity
            self.dx, self.dy = float(self.dx_input.text()), float(self.dy_input.text())
            self.simulation_time = float(self.simulation_time_input.text())
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return

        self.canvas.simulation_running = True
        self.canvas.elapsed_timer.restart()
        self.update_timer_label(self.simulation_time)

    def stop_simulation(self):
        self.canvas.simulation_running = False

    def update_timer_label(self, time_left):
        self.timer_label.setText(f"Time left: {time_left:.1f} seconds")

    def open_settings(self):
        settings_dialog = SettingsDialog(self)
        if settings_dialog.exec_() == QDialog.Accepted:
            self.canvas.init_simulation_parameters()
            self.canvas.update()



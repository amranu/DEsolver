from PyQt5.QtWidgets import QOpenGLWidget, QFileDialog
from PyQt5.QtCore import Qt, QElapsedTimer
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import csv
from equation_solver import *

class CombinedEquationPlaneWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.elapsed_timer = QElapsedTimer()
        self.simulation_running = self.drawing = False
        self.mouse_x = self.mouse_y = None
        self.init_simulation_parameters()

    def init_simulation_parameters(self):
        self.temperature = self.prev_disp = self.curr_disp = self.potential = self.source = None
        self.psi_real = self.psi_imag = None
        self.u = self.v = self.p = None
        if self.parent.equation_type == 'Heat Equation':
            self.temperature = initial_temperature(self.parent.nx, self.parent.ny)
        elif self.parent.equation_type == 'Wave Equation':
            self.prev_disp, self.curr_disp = initial_displacement(self.parent.nx, self.parent.ny)
        elif self.parent.equation_type == 'Laplace Equation':
            self.potential = initial_potential_distribution(self.parent.nx, self.parent.ny)
        elif self.parent.equation_type == 'Poisson Equation':
            self.potential = initial_potential_distribution(self.parent.nx, self.parent.ny)
            self.source = initial_source_distribution(self.parent.nx, self.parent.ny)
        elif self.parent.equation_type == 'Schrödinger Equation':
            self.psi_real, self.psi_imag = initial_wave_function(self.parent.nx, self.parent.ny)
            self.V = np.zeros((self.parent.nx, self.parent.ny))  # Potential field
        elif self.parent.equation_type == 'Navier-Stokes Equation':
            self.u, self.v = initial_velocity_vortex(self.parent.nx, self.parent.ny)
            self.p = initial_pressure(self.parent.nx, self.parent.ny)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glutInit()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -50.0)

        if self.parent.equation_type == 'Heat Equation' and self.temperature is not None:
            self.draw_plane(self.temperature)
        elif self.parent.equation_type == 'Wave Equation' and self.curr_disp is not None:
            self.draw_plane(self.curr_disp)
        elif self.parent.equation_type == 'Laplace Equation' and self.potential is not None:
            self.draw_plane(self.potential)
        elif self.parent.equation_type == 'Poisson Equation' and self.potential is not None:
            self.draw_plane(self.potential)
        elif self.parent.equation_type == 'Schrödinger Equation' and self.psi_real is not None:
            self.draw_plane(np.sqrt(self.psi_real**2 + self.psi_imag**2))  # Draw the magnitude of the wave function
        elif self.parent.equation_type == 'Navier-Stokes Equation' and self.u is not None and self.v is not None:
            self.draw_plane(np.sqrt(self.u**2 + self.v**2))  # Draw the magnitude of the velocity field

        self.draw_mouse_indicator()

    def draw_plane(self, data):
        max_val, min_val = np.max(data), np.min(data)
        for i in range(self.parent.nx):
            for j in range(self.parent.ny):
                val = data[i, j]
                color_intensity = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                glColor3f(color_intensity, 0, 1 - color_intensity)
                x, y = (i - self.parent.nx / 2) * self.parent.dx, (j - self.parent.ny / 2) * self.parent.dy
                glPushMatrix()
                glTranslatef(x, y, 0)
                glutSolidCube(min(self.parent.dx, self.parent.dy) * 0.8)
                glPopMatrix()

    def draw_mouse_indicator(self):
        if self.mouse_x is not None and self.mouse_y is not None:
            glColor3f(1.0, 1.0, 0.0)
            glPushMatrix()
            glTranslatef(self.mouse_x, self.mouse_y, 0.5)  # Slightly above the plane
            glutSolidSphere(min(self.parent.dx, self.parent.dy) * 0.4, 20, 20)
            glPopMatrix()

    def update_simulation(self):
        if not self.simulation_running:
            return

        current_time = self.elapsed_timer.elapsed() / 1000.0
        remaining_time = max(0, self.parent.simulation_time - current_time)
        self.parent.update_timer_label(remaining_time)

        if current_time < self.parent.simulation_time:
            if self.parent.equation_type == 'Heat Equation' and self.temperature is not None:
                self.temperature = solve_heat_equation_step(self.temperature, self.parent.alpha, self.parent.dx, self.parent.dy, self.parent.dt)
            elif self.parent.equation_type == 'Wave Equation' and self.curr_disp is not None:
                self.prev_disp, self.curr_disp = solve_wave_equation_step(self.prev_disp, self.curr_disp, self.parent.c, self.parent.dx, self.parent.dy, self.parent.dt)
            elif self.parent.equation_type == 'Laplace Equation' and self.potential is not None:
                self.potential = solve_laplace_equation_step(self.potential, self.parent.dx, self.parent.dy)
            elif self.parent.equation_type == 'Poisson Equation' and self.potential is not None:
                self.potential = solve_poisson_equation_step(self.potential, self.source, self.parent.dx, self.parent.dy)
            elif self.parent.equation_type == 'Schrödinger Equation' and self.psi_real is not None:
                self.psi_real, self.psi_imag = solve_schrodinger_equation_step(self.psi_real, self.psi_imag, self.V, self.parent.dx, self.parent.dy, self.parent.dt, self.parent.hbar, self.parent.m)
            elif self.parent.equation_type == 'Navier-Stokes Equation' and self.u is not None and self.v is not None and self.p is not None:
                self.u, self.v, self.p = solve_navier_stokes_step(self.u, self.v, self.p, self.parent.nu, self.parent.dx, self.parent.dy, self.parent.dt, self.parent.rho)
            self.update()
        else:
            self.reset_simulation()

    def reset_simulation(self):
        self.simulation_running = False
        self.init_simulation_parameters()
        self.update()
        self.parent.update_timer_label(0)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.set_initial_value(event.pos())
        self.update_cursor_position(event.pos())

    def mouseMoveEvent(self, event):
        self.update_cursor_position(event.pos())
        if self.drawing:
            self.set_initial_value(event.pos())
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
        self.update_cursor_position(event.pos())

    def set_initial_value(self, pos):
        width, height = self.width(), self.height()
        x, y = pos.x(), height - pos.y()
        grid_x = int((x / width) * self.parent.nx)
        grid_y = int((y / height) * self.parent.ny)

        if 0 <= grid_x < self.parent.nx and 0 <= grid_y < self.parent.ny:
            if self.parent.equation_type == 'Heat Equation' and self.temperature is not None:
                self.temperature[grid_x, grid_y] = self.parent.initial_temp
            elif self.parent.equation_type == 'Wave Equation' and self.curr_disp is not None:
                self.curr_disp[grid_x, grid_y] = self.parent.initial_disp
            elif self.parent.equation_type == 'Laplace Equation' and self.potential is not None:
                self.potential[grid_x, grid_y] = self.parent.initial_potential
            elif self.parent.equation_type == 'Poisson Equation' and self.source is not None:
                self.source[grid_x, grid_y] = self.parent.initial_source
            elif self.parent.equation_type == 'Schrödinger Equation' and self.psi_real is not None:
                self.psi_real[grid_x, grid_y] = self.parent.initial_psi_real
                self.psi_imag[grid_x, grid_y] = self.parent.initial_psi_imag
            elif self.parent.equation_type == 'Navier-Stokes Equation' and self.u is not None and self.v is not None:
                self.u[grid_x, grid_y] = self.parent.initial_u
                self.v[grid_x, grid_y] = self.parent.initial_v
            self.update()

    def update_cursor_position(self, pos):
        width, height = self.width(), self.height()
        x, y = pos.x(), height - pos.y()
        grid_x, grid_y = int((x / width) * self.parent.nx), int((y / height) * self.parent.ny)

        if 0 <= grid_x < self.parent.nx and 0 <= grid_y < self.parent.ny:
            self.mouse_x, self.mouse_y = (grid_x - self.parent.nx / 2) * self.parent.dx, (grid_y - self.parent.ny / 2) * self.parent.dy
            self.update()

    def export_data(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv);;All Files (*)")
        if file_name:
            data = None
            if self.parent.equation_type == 'Heat Equation':
                data = self.temperature
            elif self.parent.equation_type == 'Wave Equation':
                data = self.curr_disp
            elif self.parent.equation_type == 'Laplace Equation' or self.parent.equation_type == 'Poisson Equation':
                data = self.potential
            elif self.parent.equation_type == 'Schrödinger Equation':
                data = np.sqrt(self.psi_real**2 + self.psi_imag**2)  # Export the magnitude of the wave function
            elif self.parent.equation_type == 'Navier-Stokes Equation':
                data = np.dstack((self.u, self.v, self.p))  # Combine u, v, and p into a single array

            if data is not None:
                with open(file_name, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data)
                print(f"Data saved to {file_name}")

    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "CSV Files (*.csv);;All Files (*)")
        if file_name:
            with open(file_name, 'r') as file:
                reader = csv.reader(file)
                data = np.array([list(map(float, row)) for row in reader])

            if self.parent.equation_type == 'Heat Equation':
                self.temperature = data
            elif self.parent.equation_type == 'Wave Equation':
                self.curr_disp = data
            elif self.parent.equation_type == 'Laplace Equation' or self.parent.equation_type == 'Poisson Equation':
                self.potential = data
            elif self.parent.equation_type == 'Schrödinger Equation':
                self.psi_real = data
                self.psi_imag = np.zeros_like(data)  # Assuming the imaginary part is zero for simplicity
            elif self.parent.equation_type == 'Navier-Stokes Equation':
                self.u, self.v, self.p = np.dsplit(data, 3)  # Split the data into u, v, and p

            self.update()


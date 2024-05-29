from main_window import MainWindow
import sys
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow(
        nx=50, ny=50, dx=1.0, dy=1.0, alpha=0.1, c=1.0, nu=0.1, dt=0.01, simulation_time=50,
        initial_temp=100.0, initial_disp=1.0, initial_potential=0.0, initial_source=0.0,
        hbar=1.0, m=1.0, rho=1.0)

    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


import contextlib
import itertools
import sys
from functools import partial

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QFont, QIcon, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMessageBox
from rich import print

from sudoku import Difficulties, check_solution, generate_board


# It's a QObject that emits a signal when it's done
class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, output_filename) -> None:
        """
        This function initializes the class and sets the output filename

        Args:
          output_filename: The name of the file to write the output to.
        """
        super().__init__()
        self.output_filename = output_filename

    def run(self, output_filename):
        """
        It checks the solution of the user's code and emits a signal to the GUI to update the status of
        the problem

        Args:
          output_filename: The name of the file that the user wants to save the output to.
        """
        check_solution(output_filename)
        self.finished.emit()


# This class is a subclass of the QMainWindow class, which is a class that provides a main application
# window
class Window(QtWidgets.QMainWindow):
    def __init__(self):
        """
        It loads the UI file, sets the window size, sets the window title, sets the brush size, sets the
        brush color, sets the last point, sets the painter, sets the pointer type, sets the pen
        pressure, connects the generate action to the generate function, connects the clear action to
        the clear function, connects the toggle solution action to the toggle solution function, and
        then calls the generate function.
        """
        super().__init__()
        uic.loadUi("main.ui", self)
        width, height = 1500, 1520
        self.setFixedSize(width, height)
        self.setWindowTitle("Sudoku")

        self.drawing: bool = False
        self.erase: bool = False
        self.show_solution: bool = False
        self.brushSize: int = 10
        self._clear_size: int = 40
        self.brushColor = QtGui.QColor(QtCore.Qt.blue)
        self.lastPoint = QtCore.QPoint()
        self.painter = QtGui.QPainter()
        self.pointer_type = 0
        self.pen_pressure: float = 1
        self.difficulty: int = Difficulties.normal

        self.generateAction.triggered.connect(self.generate)
        self.generateAction.setEnabled(False)
        self.clearAction.triggered.connect(self.clear)
        self.toggleSolutionAction.triggered.connect(self.toggle_solution)

        self.actionEasy.triggered.connect(
            partial(self.set_difficulty, Difficulties.easy, self.actionEasy)
        )
        self.actionNormal.triggered.connect(
            partial(self.set_difficulty, Difficulties.normal, self.actionNormal)
        )
        self.actionHard.triggered.connect(
            partial(self.set_difficulty, Difficulties.hard, self.actionHard)
        )

        self.load_cell_positions()
        self.generate()

    def set_difficulty(self, difficulty: int, action):
        self.actionEasy.setChecked(False)
        self.actionNormal.setChecked(False)
        self.actionHard.setChecked(False)
        action.setChecked(True)
        self.difficulty = difficulty

    def solve(self, output_filename: str):
        """
        It creates a QThread, creates a Worker object, moves the Worker object to the QThread, connects
        the Worker object's run function to the QThread's started signal, connects the Worker object's
        finished signal to the QThread's quit signal, connects the Worker object's finished signal to
        the Worker object's deleteLater slot, connects the Worker object's finished signal to the
        found_solution slot, connects the QThread's finished signal to the QThread's deleteLater slot,
        and starts the QThread

        Args:
          output_filename (str): str = The name of the file to save the solution to.
        """
        self.thread = QThread()
        self.worker = Worker(output_filename=output_filename)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(partial(self.worker.run, output_filename))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.found_solution)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def found_solution(self):
        """
        If the solver has found a solution, then the toggleSolutionAction is enabled
        """
        self.toggleSolutionAction.setEnabled(True)
        self.generateAction.setEnabled(True)

    def clear(self):
        """
        It clears the board and loads the board.png image.
        """
        with contextlib.suppress(Exception):
            self.painter.end()
        self.image = QtGui.QImage("board.png")
        self.imageDraw = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32)
        self.imageDraw.fill(QtCore.Qt.transparent)
        self.painter = QtGui.QPainter(self.image)
        self.painter.setFont(QFont("Calibri", 64))
        self.load_board()
        self.update()

    def generate(self):
        """
        It generates a new board, loads it, updates the board, and saves the board as a screenshot.
        """
        self.toggleSolutionAction.setEnabled(False)
        with contextlib.suppress(Exception):
            self.painter.end()
        self.image = QtGui.QImage("board.png")
        self.imageDraw = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32)
        self.imageDraw.fill(QtCore.Qt.transparent)
        self.painter = QtGui.QPainter(self.image)
        self.painter.setFont(QFont("Calibri", 64))
        generate_board(self.difficulty)
        self.load_board()
        self.update()
        self.image.save("screenshot.png", "png")
        self.solve(output_filename="solution.png")

    def load_cell_positions(self) -> None:
        """
        It creates a list of lists of tuples, where each tuple represents the x and y coordinates of a
        cell in the grid
        """
        self.cell_positions = []
        for row in range(9):
            self.cell_positions.append([])
            for col in range(9):
                self.cell_positions[row].append((111 * (col), 111 * (row) + 20))

    def load_board(self) -> None:
        """
        It reads a text file, and then draws the numbers from the text file onto the board.
        """
        with open("board.txt", "r") as f:
            lines = f.readlines()
        lines = [line.replace("\n", "") for line in lines]
        for row, col in itertools.product(range(9), range(9)):
            line = lines[row]
            number = line[col]
            if number == "0":
                number = ""
            self.painter.setRenderHint(QPainter.TextAntialiasing, True)
            self.painter.setRenderHint(QPainter.HighQualityAntialiasing, True)
            self.painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            self.painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(QtCore.Qt.white),
                    self.brushSize,
                    QtCore.Qt.SolidLine,
                    QtCore.Qt.RoundCap,
                    QtCore.Qt.RoundJoin,
                )
            )
            self.painter.drawText(
                self.cell_positions[row][col][0],
                self.cell_positions[row][col][1],
                111,
                111,
                Qt.AlignCenter,
                number,
            )

    def toggle_solution(self) -> None:
        """
        It toggles the solution on and off
        """
        self.show_solution = not self.show_solution
        self.painter.end()
        if self.show_solution:
            self.toggleSolutionAction.setText("Hide Solution")
            self.image = QtGui.QImage("solution.png")
            self.painter = QtGui.QPainter(self.image)
        else:
            self.toggleSolutionAction.setText("Show Solution")
            self.image = QtGui.QImage("board.png")
            self.painter = QtGui.QPainter(self.image)
            self.painter.setFont(QFont("Calibri", 64))
            self.load_board()
        self.update()
        # QMessageBox.about(self, "Found", "Solution found")
        # else:
        # QMessageBox.about(self, "Wrong", "No solution")

    def mousePressEvent(self, event):
        """
        If the pointer type is 1, then the drawing is set to True, erase is set to False, and the last
        point is set to the position of the event.
        If the pointer type is 3, then erase is set to True, a pixmap is created, a painter is created,
        the painter is set to a pen, and the painter draws a rectangle.

        Args:
          event: The event that triggered the mousePressEvent
        """
        if self.pointer_type == 1 or event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.erase = False
            self.lastPoint = event.pos()
            QtWidgets.QApplication.restoreOverrideCursor()
        elif self.pointer_type == 3 or event.button() == QtCore.Qt.RightButton:
            self.erase = True
            pixmap = QtGui.QPixmap(QtCore.QSize(1, 1) * self._clear_size)
            pixmap.fill(QtCore.Qt.transparent)
            self.painter = QtGui.QPainter(pixmap)
            self.painter.setPen(QtGui.QPen(QtCore.Qt.white, 2))
            self.painter.drawRect(pixmap.rect())
            self.painter.end()
            cursor = QtGui.QCursor(pixmap)
            QtWidgets.QApplication.setOverrideCursor(cursor)

    def tabletEvent(self, tabletEvent):
        """
        It takes the tabletEvent and assigns the pointer type and pressure to the variables
        self.pointer_type and self.pen_pressure.

        Args:
          tabletEvent: The event object.
        """
        self.pointer_type = tabletEvent.pointerType()
        self.pen_pressure = tabletEvent.pressure()
        tabletEvent.accept()

    def mouseMoveEvent(self, event):
        """
        If the left mouse button is pressed and the drawing variable is true, then draw a line from the
        last point to the current point.

        Args:
          event: The event object that contains the mouse position.
        """
        print(self.pen_pressure)
        if event.buttons() and QtCore.Qt.LeftButton and self.drawing:
            self.painter = QtGui.QPainter(self.imageDraw)
            self.painter.setPen(
                QtGui.QPen(
                    self.brushColor,
                    self.brushSize * self.pen_pressure,
                    QtCore.Qt.SolidLine,
                    QtCore.Qt.RoundCap,
                    QtCore.Qt.RoundJoin,
                )
            )
            if self.erase:
                r = QtCore.QRect(QtCore.QPoint(), self._clear_size * QtCore.QSize())
                r.moveCenter(event.pos())
                self.painter.save()
                self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
                self.painter.eraseRect(r)
                self.painter.restore()
            else:
                self.painter.drawLine(self.lastPoint, event.pos())
            self.painter.end()
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """
        If the left mouse button is released, set the drawing variable to False and restore the cursor
        to its default state

        Args:
          event: The event object that contains the mouse position.
        """
        if event.button == QtCore.Qt.LeftButton:
            self.drawing = False
        QtWidgets.QApplication.restoreOverrideCursor()

    def paintEvent(self, event):
        """
        It draws the image on the canvas.

        Args:
          event: The event object that was passed to the paintEvent() method.
        """
        canvasPainter = QtGui.QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        canvasPainter.drawImage(self.rect(), self.imageDraw, self.imageDraw.rect())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())

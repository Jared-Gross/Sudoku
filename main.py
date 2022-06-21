import contextlib
import itertools
import sys
from functools import partial
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from rich import print
from PyQt5 import uic
from sudoku import check_solution, generate_board



class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, output_filename) -> None:
        super().__init__()
        self.output_filename = output_filename

    def run(self, output_filename):
        check_solution(output_filename)
        self.finished.emit()
        
class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('main.ui', self)
        width, height = 1500, 1520
        self.setFixedSize(width, height)
        self.setWindowTitle("Sudoku")

        self.drawing = False
        self.erase = False
        self.show_solution = False
        self.brushSize = 10
        self._clear_size = 40
        self.brushColor = QtGui.QColor(QtCore.Qt.blue)
        self.lastPoint = QtCore.QPoint()
        self.painter = QtGui.QPainter()
        self.pointer_type = 0
        self.pen_pressure: float = 0.0
        
        self.generateAction.triggered.connect(self.generate)
        self.clearAction.triggered.connect(self.clear)
        self.toggleSolutionAction.triggered.connect(self.toggle_solution)

        self.load_cell_positions()
        self.generate()
        
    def solve(self, output_filename: str):
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
        self.toggleSolutionAction.setEnabled(True)
        
    def clear(self):
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
        self.toggleSolutionAction.setEnabled(False)
        with contextlib.suppress(Exception):
            self.painter.end()
        self.image = QtGui.QImage("board.png")
        self.imageDraw = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32)
        self.imageDraw.fill(QtCore.Qt.transparent)
        self.painter = QtGui.QPainter(self.image)
        self.painter.setFont(QFont("Calibri", 64))
        generate_board()
        self.load_board()
        self.update()
        self.image.save('screenshot.png', 'png')
        self.solve(output_filename='solution.png')

    def load_cell_positions(self) -> None:
        self.cell_positions = []
        for row in range(9):
            self.cell_positions.append([])
            for col in range(9):
                self.cell_positions[row].append((111 * (col), 111 * (row)+20))

    def load_board(self) -> None:
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
        if self.pointer_type == 1:
            self.drawing = True
            self.erase = False
            self.lastPoint = event.pos()
            QtWidgets.QApplication.restoreOverrideCursor()
        elif self.pointer_type == 3:
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
        self.pointer_type = tabletEvent.pointerType()
        self.pen_pressure = tabletEvent.pressure()
        tabletEvent.accept()
        
    def mouseMoveEvent(self, event):
        if event.buttons() and QtCore.Qt.LeftButton and self.drawing:
            self.painter = QtGui.QPainter(self.imageDraw)
            self.painter.setPen(
                QtGui.QPen(
                    self.brushColor,
                    self.brushSize*self.pen_pressure,
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
        if event.button == QtCore.Qt.LeftButton:
            self.drawing = False
        QtWidgets.QApplication.restoreOverrideCursor()

    def paintEvent(self, event):
        canvasPainter = QtGui.QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        canvasPainter.drawImage(self.rect(), self.imageDraw, self.imageDraw.rect())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())

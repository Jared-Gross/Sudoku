import contextlib
import itertools
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import QMessageBox

from sudoku import check_solution, generate_board


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        top, left, width, height = 400, 400, 800, 800
        self.setFixedSize(width, height)
        self.setWindowTitle("Sudoku")

        mainMenu = self.menuBar()
        sudoku = mainMenu.addMenu("Sudoku")
        solveAction = QtWidgets.QAction("Solve", self)
        solveAction.triggered.connect(self.solve)

        generateAction = QtWidgets.QAction("New", self)
        sudoku.addAction(generateAction)
        sudoku.addAction(solveAction)
        generateAction.triggered.connect(self.generate)
        self.load_cell_positions()
        self.generate()

    def generate(self):
        with contextlib.suppress(Exception):
            self.painter.end()
        self.image = QtGui.QImage("board.png")
        self.imageDraw = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32)
        self.imageDraw.fill(QtCore.Qt.transparent)
        self.painter = QtGui.QPainter(self.image)
        self.painter.setFont(QFont("Calibri", 64))
        generate_board()
        self.load_board()
        self.drawing = False
        self.erase = False
        self.brushSize = 5
        self._clear_size = 20
        self.brushColor = QtGui.QColor(QtCore.Qt.black)
        self.lastPoint = QtCore.QPoint()
        self.update()

    def load_cell_positions(self) -> None:
        self.cell_positions = []
        for row in range(9):
            self.cell_positions.append([])
            for col in range(9):
                self.cell_positions[row].append((111 * (col), 111 * (row)))

    def load_board(self) -> None:
        with open("board.txt", "r") as f:
            lines = f.readlines()
        lines = [line.replace("\n", "") for line in lines]
        for row, col in itertools.product(range(9), range(9)):
            line = lines[row]
            number = line[col]
            if number == "0":
                number = ""
            self.painter.drawText(
                self.cell_positions[row][col][0],
                self.cell_positions[row][col][1],
                111,
                111,
                Qt.AlignCenter,
                number,
            )

    def solve(self) -> None:
        pixmap = QPixmap(self.grab())
        pixmap.save("screenshot.png")
        if check_solution():
            self.painter.end()
            self.image = QtGui.QImage("solution.png")
            self.painter = QtGui.QPainter(self.image)
            self.update()
            QMessageBox.about(self, "Found", "Solution found")
        else:
            QMessageBox.about(self, "Wrong", "No solution")

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.erase = False
            self.lastPoint = event.pos()
            QtWidgets.QApplication.restoreOverrideCursor()
        else:
            self.erase = True
            pixmap = QtGui.QPixmap(QtCore.QSize(1, 1) * self._clear_size)
            pixmap.fill(QtCore.Qt.transparent)
            self.painter = QtGui.QPainter(pixmap)
            self.painter.setPen(QtGui.QPen(QtCore.Qt.black, 2))
            self.painter.drawRect(pixmap.rect())
            self.painter.end()
            cursor = QtGui.QCursor(pixmap)
            QtWidgets.QApplication.setOverrideCursor(cursor)

    def mouseMoveEvent(self, event):
        if event.buttons() and QtCore.Qt.LeftButton and self.drawing:
            self.painter = QtGui.QPainter(self.imageDraw)
            self.painter.setPen(
                QtGui.QPen(
                    self.brushColor,
                    self.brushSize,
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

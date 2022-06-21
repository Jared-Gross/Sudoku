import itertools
from random import sample

import cv2
import imutils
import numpy as np
import pytesseract
from rich import print
from skimage.filters import threshold_local


class Difficulties:
    easy: int = 8
    normal: int = 6
    hard: int = 3


def order_points(pts):
    """
    It takes in a list of four points, and returns a list of four points where the points are ordered in
    a clockwise fashion

    Args:
      pts: The input array of (x, y) coordinates.

    Returns:
      The ordered coordinates of the rectangle.
    """

    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    """
    The function takes an image and a set of four points, applies a perspective transform to the image,
    and returns the transformed image

    Args:
      image: The image we want to transform.
      pts: The points that we want to transform.

    Returns:
      The warped image
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    # return the warped image
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def show_image(img, title):
    """
    It shows an image and waits for a key press

    Args:
      img: The image to be displayed
      title: The title of the window
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    """
    It finds the largest connected white area in the image, and then fills it in white, and everything
    else in black

    Args:
      inp_img: The image to be scanned
      scan_tl: The top left corner of the area to scan in search of the grid.
      scan_br: The bottom right corner of the area to scan for the grid.

    Returns:
      The largest feature in the image.
    """
    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # Only operate on light or white squares
            if (
                img.item(y, x) == 255 and x < width and y < height
            ):  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if (
                    area[0] > max_area
                ):  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    # Colour everything grey (compensates for features outside of our middle scanning range
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros(
        (height + 2, width + 2), np.uint8
    )  # Mask that is 2 pixels bigger than the image

    # Highlight the main feature
    if all(p is not None for p in seed_point):
        cv2.floodFill(img, mask, seed_point, 255)

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)

    return img


def preprocess(image, case):
    """
    It takes an image, resizes it, and then applies a perspective transform to it

    Args:
      image: The image to be preprocessed
      case: True if the image is a scanned image, False if it's a photo

    Returns:
      The preprocess function returns the following:
    """
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    # image = imutils.resize(image, height=500)

    if case == True:

        gray = cv2.GaussianBlur(image, (5, 5), 0)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
        div = np.float32(gray) / (close)
        res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
        edged = cv2.Canny(res, 75, 200)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            rect = cv2.boundingRect(c)

            cv2.rectangle(
                edged.copy(),
                (rect[0], rect[1]),
                (rect[2] + rect[0], rect[3] + rect[1]),
                (0, 0, 0),
                2,
            )
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                # print(screenCnt)
                break

        # show the contour (outline) of the piece of paper
        # print(screenCnt)
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

        # apply the four point transform to obtain a top-down
        # view of the original image
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    else:

        warped = image
    warped1 = cv2.resize(warped, (1500, 1520))
    warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warp, 11, offset=10, method="gaussian")
    warp = (warp > T).astype("uint8") * 255
    th3 = cv2.adaptiveThreshold(
        warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # show_image(warped1,"preprocessed")

    return th3, warped1, warped


def grids(img, warped2):
    """
    It draws a grid on the image

    Args:
      img: The image to be processed
      warped2: The image that we want to draw the grid on.

    Returns:
      the image with the grid lines drawn on it.
    """
    img = np.zeros((500, 500, 3), np.uint8)

    frame = img

    img = cv2.resize(frame, (1500, 1520))

    for i in range(10):
        cv2.line(
            img,
            (0, (img.shape[0] // 9) * i),
            (img.shape[1], (img.shape[0] // 9) * i),
            (255, 255, 255),
            3,
            1,
        )
        cv2.line(
            warped2,
            (0, (img.shape[0] // 9) * i),
            (img.shape[1], (img.shape[0] // 9) * i),
            (125, 0, 55),
            3,
            1,
        )

    for j in range(10):
        cv2.line(
            img,
            ((img.shape[1] // 9) * j, 0),
            ((img.shape[1] // 9) * j, img.shape[0]),
            (255, 255, 255),
            3,
            1,
        )
        cv2.line(
            warped2,
            ((img.shape[1] // 9) * j, 0),
            ((img.shape[1] // 9) * j, img.shape[0]),
            (125, 0, 55),
            3,
            1,
        )

    # show_image(warped2,"grids")
    return img


def grid_points(img, warped2):
    """
    It takes an image and a warped image as input, and returns the centroids of the grid points, the
    grid points in a 10x10 matrix, and the contours of the grid points

    Args:
      img: The original image
      warped2: The image that has been warped to a top-down view.

    Returns:
      the centroids of the grid points, the grid points in a 10x10 matrix, and the contours of the grid
    points.
    """
    img1 = img.copy()
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))

    dx = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    dx = cv2.convertScaleAbs(dx)
    c = cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
    c = cv2.morphologyEx(c, cv2.MORPH_DILATE, kernelx, iterations=1)
    cy = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    closex = cv2.morphologyEx(cy, cv2.MORPH_DILATE, kernelx, iterations=1)

    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    dy = cv2.Sobel(img, cv2.CV_16S, 0, 2)
    dy = cv2.convertScaleAbs(dy)
    c = cv2.normalize(dy, dy, 0, 255, cv2.NORM_MINMAX)
    c = cv2.morphologyEx(c, cv2.MORPH_DILATE, kernely, iterations=1)
    cy = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    closey = cv2.morphologyEx(cy, cv2.MORPH_DILATE, kernelx, iterations=1)

    res = cv2.bitwise_and(closex, closey)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((6, 6), np.uint8)

    # Perform morphology
    se = np.ones((8, 8), dtype="uint8")
    image_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)
    image_close = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, kernel)

    contour, hier = cv2.findContours(image_close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contour, key=cv2.contourArea, reverse=True)[:100]
    centroids = []
    for cnt in cnts:

        mom = cv2.moments(cnt)
        (x, y) = int(mom["m10"] / mom["m00"]), int(mom["m01"] / mom["m00"])
        cv2.circle(img1, (x, y), 4, (0, 255, 0), -1)
        cv2.circle(warped2, (x, y), 4, (0, 255, 0), -1)
        centroids.append((x, y))

    # show_image(warped2,"grid_points")

    Points = np.array(centroids, dtype=np.float32)
    c = Points.reshape((100, 2))
    c2 = c[np.argsort(c[:, 1])]

    b = np.vstack(
        [
            c2[i * 10 : (i + 1) * 10][np.argsort(c2[i * 10 : (i + 1) * 10, 0])]
            for i in range(10)
        ]
    )
    bm = b.reshape((10, 10, 2))

    return c2, bm, cnts


def image_to_num(c2):
    """
    It takes an image of a single digit, inverts it, and then uses the Tesseract OCR engine to convert
    it to a number

    Args:
      c2: the image

    Returns:
      A list of the first character of the text.
    """
    img = 255 - c2
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    )

    text = pytesseract.image_to_string(
        img, lang="eng", config="--psm 6 --oem 3"
    )  # builder=builder)
    return list(text)[0]


def get_digit(c2, bm, warped1, cnts):
    """
    It takes an image, finds the largest feature, and then returns the image with the largest feature

    Args:
      c2: the image
      bm: the list of coordinates of the boxes
      warped1: The image that we want to extract the digits from
      cnts: the contours of the image

    Returns:
      the following:
    """
    num = []
    centroidx = np.empty((9, 9))
    centroidy = np.empty((9, 9))
    global list_images
    list_images = []
    for i, j in itertools.product(range(9), range(9)):
        x1, y1 = bm[i][j]  # bm[0] row1
        x2, y2 = bm[i + 1][j + 1]

        coordx = (x1 + x2) // 2
        coordy = (y1 + y2) // 2
        centroidx[i][j] = coordx
        centroidy[i][j] = coordy
        crop = warped1[int(x1) : int(x2), int(y1) : int(y2)]
        crop = imutils.resize(crop, height=69, width=67)
        c2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        c2 = cv2.adaptiveThreshold(
            c2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((2, 2), np.uint8)
        # c2 = cv2.morphologyEx(c2, cv2.MORPH_OPEN, kernel)
        c2 = cv2.copyMakeBorder(c2, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        no = 0
        shape = c2.shape
        w = shape[1]
        h = shape[0]
        mom = cv2.moments(c2)
        (x, y) = int(mom["m10"] / mom["m00"]), int(mom["m01"] / mom["m00"])
        c2 = c2[14:70, 15:62]
        contour, hier = cv2.findContours(c2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if cnts is not None:
            cnts = sorted(contour, key=cv2.contourArea, reverse=True)[:1]

        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            #               print(aspect_ratio)
            area = cv2.contourArea(cnt)
            # print(area)
            if (
                area > 120
                and cnt.shape[0] > 15
                and aspect_ratio > 0.2
                and aspect_ratio <= 0.9
            ):
                # print("area:",area)
                c2 = find_largest_feature(c2)
                # show_image(c2,"box2")
                contour, hier = cv2.findContours(
                    c2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                cnts = sorted(contour, key=cv2.contourArea, reverse=True)[:1]
                for cnt in cnts:
                    rect = cv2.boundingRect(cnt)
                    # cv2.rectangle(c2, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (255,255,255), 2)
                    c2 = c2[rect[1] : rect[3] + rect[1], rect[0] : rect[2] + rect[0]]
                    c2 = cv2.copyMakeBorder(
                        c2, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                    list_images.append(c2)
                # show_image(c2,"box")
                no = image_to_num(c2)
        num.append(no)
    centroidx = np.transpose(centroidx)
    centroidy = np.transpose(centroidy)
    return c2, num, centroidx, centroidy


def sudoku_matrix(num):
    """
    It takes a string of 81 digits and returns a 9x9 matrix

    Args:
      num: a string of 81 digits, representing the sudoku puzzle

    Returns:
      A 9x9 matrix of the sudoku puzzle
    """
    grid = np.empty((9, 9))
    for c, (i, j) in enumerate(itertools.product(range(9), range(9))):
        try:
            grid[i][j] = int(num[c])
        except ValueError:
            if num[c] == "?":
                grid[i][j] = 2

    grid = np.transpose(grid)
    return grid


def check_col(arr, num, col) -> bool:
    """
    It returns True if the number is not in the column, and False otherwise

    Args:
      arr: The array of the sudoku board
      num: the number we're checking
      col: the column number

    Returns:
      A boolean value.
    """
    return all(num != arr[i][col] for i in range(9))


def check_row(arr, num, row) -> bool:
    """
    It returns True if the number is not in the row, and False otherwise

    Args:
      arr: The array of the sudoku board
      num: the number we're checking
      row: the row number

    Returns:
      True or False
    """
    return all(num != arr[row][i] for i in range(9))


def check_cell(arr, num, row, col) -> bool:
    """
    It checks if the number is already in the 3x3 section of the board

    Args:
      arr: The array of the sudoku board
      num: the number we're checking
      row: the row of the cell we're checking
      col: the column of the cell we're checking

    Returns:
      A boolean value.
    """
    sectopx = 3 * (row // 3)
    sectopy = 3 * (col // 3)

    return any(
        arr[i][j] == num
        for i, j in itertools.product(
            range(sectopx, sectopx + 3), range(sectopy, sectopy + 3)
        )
    )


def empty_loc(arr, l) -> bool:
    """
    It takes a 2D array and a list as input and returns True if the 2D array has an empty location, and
    if so, it fills the list with the row and column of the empty location

    Args:
      arr: The sudoku board
      l: a list of length 2, which will be used to store the row and column of the empty cell

    Returns:
      a boolean value.
    """
    for i, j in itertools.product(range(9), range(9)):
        if arr[i][j] == 0:
            l[0] = i
            l[1] = j
            return True
    return False


def sudoku(arr) -> bool:
    """
    If there is an empty location, try all numbers from 1 to 9 in that location and recursively call the
    function. If it returns True, return True. If it returns False, undo the number and try again

    Args:
      arr: The sudoku board

    Returns:
      a boolean value.
    """
    l = [0, 0]

    if not empty_loc(arr, l):
        return True

    row = l[0]
    col = l[1]

    for num in range(1, 10):
        if (
            check_row(arr, num, row)
            and check_col(arr, num, col)
            and not check_cell(arr, num, row, col)
        ):
            arr[row][col] = int(num)

            if sudoku(arr):
                return True

            # failure, unmake & try again
            arr[row][col] = 0

    return False


def overlay(arr, num, img, cx, cy, output_filename):
    """
    It takes the solved sudoku array, the array of numbers that are not to be written, the image of the
    sudoku, and the x and y coordinates of the center of each box, and writes the numbers in the solved
    sudoku array on the image of the sudoku

    Args:
      arr: The solved sudoku array
      num: This is the array that contains the numbers that are already present in the sudoku.
      img: The image of the sudoku puzzle
      cx: x-coordinates of the centers of the 81 cells
      cy: The y-coordinates of the centers of the 81 cells.
    """
    for no, (i, j) in enumerate(itertools.product(range(9), range(9))):
        if num[no] == 0:

            cv2.putText(
                img,
                str(int(arr[j][i])),
                (int(cx[i][j]) + 30, int(cy[i][j]) + 90 - (j * 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
    img = cv2.bitwise_not(img)
    # img = cv2.copyMakeBorder(img, 35, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.imwrite(output_filename, img)
    # cv2.imshow("Sudoku", img)
    # cv2.waitKey(0)


def generate_board(difficulty: int) -> None:
    """
    It generates a random sudoku board and writes it to a file
    """
    base = 3
    side = base * base

    # pattern for a baseline valid solution
    def pattern(r, c):
        return (base * (r % base) + r // base + c) % side

    def shuffle(s):
        return sample(s, len(s))

    rBase = range(base)
    rows = [g * base + r for g in shuffle(rBase) for r in shuffle(rBase)]
    cols = [g * base + c for g in shuffle(rBase) for c in shuffle(rBase)]
    nums = shuffle(range(1, base * base + 1))

    # produce board using randomized baseline pattern
    board = [[nums[pattern(r, c)] for c in cols] for r in rows]

    squares = side * side
    empties = squares * 3 // difficulty
    for p in sample(range(squares), empties):
        board[p // side][p % side] = 0

    with open("board.txt", "w") as f:
        f.write("")
    for i, line in enumerate(board):
        line = "".join([str(elem) for elem in line])
        with open("board.txt", "a") as f:
            f.writelines(line)
            if i != len(board) - 1:
                f.write("\n")


def get_digit_from_file() -> list:
    num = []
    with open("board.txt", "r") as f:
        lines = f.readlines()
    num.extend(
        int(lines[j][i].replace("\n", ""))
        for i, j in itertools.product(range(9), range(9))
    )
    return num


def check_solution(output_filename) -> bool:
    """
    It takes a screenshot of the sudoku board, preprocesses it, finds the grid points, finds the digits,
    creates a sudoku matrix, solves the sudoku, and then overlays the solution on the original image

    Returns:
      a boolean value.
    """
    case = "False"  # If transformation is required set True
    image = cv2.imread("screenshot.png")
    image = cv2.bitwise_not(image)

    _, warped1, warped = preprocess(image, case)
    warped2 = warped1.copy()
    img = grids(warped, warped2)
    c2, bm, cnts = grid_points(img, warped2)
    c2, num, cx, cy = get_digit(c2, bm, warped1, cnts)
    grid = sudoku_matrix(num)
    if sudoku(grid):
        overlay(grid, num, warped1, cx, cy, output_filename)
        return True
    else:
        return False

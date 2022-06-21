# Sudoku

Sudoku that is designed to be played on tablet such as Surface Pro. Performance is really good, on idle its 0%, when looking for solution to sudoku puzzle it goes up to 6% at most. Ram usage is around 100mb. Its just what you need for a table board game. 

![image](https://user-images.githubusercontent.com/25397800/174698150-6be61833-423a-4ef3-b901-9b3103d9fc19.png)

## Requirements

Download and install: https://www.softpedia.com/get/Programming/Other-Programming-Files/Tesseract-OCR.shtml

Make sure you fix paths in `main.py`

PyTesseract might through environmental errors at you, they give good instructions on how to fix it.

Python requirements
```
pip install -r requirements.txt
```

## Run

```
python main.py
```

## Light mode

Dark mode is permanent, to change it you need to change the `board.png` and a couple `bitwise_not` in the code, Line 639, 695. And line 117 in `main.py` change to `Qt.black`. But Darkmode is alot better 

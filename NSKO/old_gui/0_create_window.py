import tkinter as tk

window = tk.Tk()
window.title("ФММ РАЗ")
# window.geometry("400x300")

first_button = tk.Button(
    text="Hello",          # текст кнопки
    # background="#555",     # фоновый цвет кнопки
    # foreground="#ccc",     # цвет текста
    padx="20",             # отступ от границ до содержимого по горизонтали
    pady="8",              # отступ от границ до содержимого по вертикали
    font="16"              # высота шрифта
)
first_button.pack()

window.mainloop()

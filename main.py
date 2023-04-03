from tkinter import *
from speechtotext import *
from chatbot import *
from texttospeech import *
import time
import webbrowser

# GUI
root = Tk()
root.title("QNEbot")
 
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#FFFFFF"
 
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

def send():
    txt.insert(END, "\nYou : " + e.get())
    request = e.get().lower()
    ints = predict_class(request)
    response = get_response(ints)
    txt.insert(END, "\nQNEbot : " + response)
    text_to_speech(response)
    search(request, response)
    e.delete(0, END)

def record():
    time.sleep(0.8)
    request = speech_to_text(duration=3)

    txt.insert(END, "\nYou : " + request)
    ints = predict_class(request)
    response = get_response(ints)
    txt.insert(END, "\nQNEbot : " + response)
    text_to_speech(response)
    search(request, response)

def search(request, response):
    if re.search("http", response):
         webbrowser.open(response)
    elif re.search("google", response):
        item = request.replace("google", "")
        webbrowser.open(f"https://google.com/search?q={item}")
    
txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
txt.grid(row=1, column=0, columnspan=5)

txt.insert(END, "QNEbot : Hi! I am QNEbot. How can I help?")
 
v_scrollbar = Scrollbar(txt)
v_scrollbar.place(relheight=1, relx=0.974)

label1 = Label(root,pady=10, width=30, height=2).grid(row=2)
 
e = Entry(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=48)
e.grid(row=2, column=0)
 
btn_send = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY,
              command=send).grid(row=2, column=1)

btn_record = Button(root, text="Record", font=FONT_BOLD, bg=BG_GRAY,
              command=record).grid(row=2, column=2)
 
root.mainloop()
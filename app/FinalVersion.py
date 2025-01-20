import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk # type: ignore
import random
import os
import time
from playsound import playsound # type: ignore

limit= 10
img_size=1000
name = False
dataset_path = 'dataset1k'

class ImageClassificationApp:
    def __init__(self, root, image_folder_path):
        self.root = root
        self.image_folder_path = image_folder_path
        self.correct_answers = 0
        self.wrong_answers = 0
        self.images = []
        self.current_image_index = -1
        self.selection_counter = 0
        self.current_list = []
        self.guesses = []
        self.mistakes= []
        
        self.load_images()
        self.create_widgets()
        self.start_timer()
        self.display_next_image()

    def load_images(self):
        for class_folder in os.listdir(self.image_folder_path):
            class_path = os.path.join(self.image_folder_path, class_folder)
            if os.path.isdir(class_path):
                class_images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
                self.images.extend(class_images)

    def create_widgets(self):
        self.image_label = ttk.Label(self.root)
        self.image_label.pack()

        self.image_name_label = ttk.Label(self.root, text="")
        self.image_name_label.pack()

        self.completed_label = ttk.Label(self.root, text="Completed", font=("Helvetica", 57), foreground="green")
        self.result_label = ttk.Label(self.root, text="Correct: 0, Wrong: 0",font=("Helvetica", 20))
        self.selection_label = ttk.Label(self.root, text="Selection: {self.selection_counter}")
        self.class1_button = ttk.Button(self.root, text="Real", command=lambda: self.check_answer(0))
        self.class2_button = ttk.Button(self.root, text="Deepfake", command=lambda: self.check_answer(1))
        self.timer_label = ttk.Label(self.root, text="Time: 00:00",font=("Helvetica", 20))
        self.complete_list = ttk.Label(self.root, text="Complete list:")
        self.guesses_label = ttk.Label(self.root, text="Guesses:")

        
        self.completed_label.pack_forget()
        self.timer_label.pack_forget()
        self.result_label.pack_forget()
        self.complete_list.pack_forget()
        self.selection_label.pack()
        self.class1_button.pack(side=tk.LEFT, padx=10)
        self.class2_button.pack(side=tk.RIGHT, padx=10)
    
    def start_timer(self):
        self.start_time = time.time()

    def display_next_image(self):
        if self.selection_counter <= limit+1:  
            self.current_image_index = random.randint(0, len(self.images) - 1)
            image_path = self.images[self.current_image_index]
            image_name = os.path.basename(image_path)
            self.current_list.append(image_path)

            image = Image.open(image_path)
            image = image.resize((img_size, img_size))  
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo 

            if name:
                self.image_name_label.configure(text=image_name)


            self.selection_counter += 1
            self.selection_label.config(text=f"Selection: {self.selection_counter}")
            if self.selection_counter == limit+1:
                self.completed_label.pack()
                self.result_label.pack()
                self.timer_label.config(text=f"Time: {time.time() - self.start_time:.2f}")
                self.timer_label.pack()
                
                complete_frame = ttk.Frame(self.root)
                complete_frame.pack()

                for i, image_path in enumerate(self.current_list[0:10]):
                    image = Image.open(image_path)
                    image = image.resize((200, 200))  
                    photo = ImageTk.PhotoImage(image)
                    image_label = ttk.Label(complete_frame, image=photo)
                    image_label.image = photo
                    image_label.grid(row=i // 5, column=i % 5, padx=10, pady=10)

                    image_name = os.path.basename(image_path)


                self.mistakes_label = ttk.Label(self.root, text=str(len(self.mistakes)) +" mistakes at position " + str(self.mistakes),font=("Helvetica", 17))
                self.mistakes_label.pack()
                self.guesses_label.config(text="Guesses: " + str(self.guesses))
                self.guesses_label.pack()
                displayed_images = "\n".join(self.current_list[0:10])
                self.complete_list.config(text="Complete list:\n" + displayed_images)
                self.complete_list.pack()
                self.image_name_label.pack_forget()
                self.image_label.pack_forget()
                self.selection_label.pack_forget()
                self.class1_button.pack_forget()
                self.class2_button.pack_forget()
                self.root.update()
                playsound('D:\CV\\app\\finished_sound.mp3')



    def check_answer(self, chosen_class):
        if self.selection_counter <= limit+1:  
            # Controllare bene indice
            self.guesses.append(chosen_class)
            image_class = 0 if self.current_image_index < 500 else 1
            
            if chosen_class == image_class:
                self.correct_answers += 1
            else:
                self.wrong_answers += 1
                self.mistakes.append(self.selection_counter)
            
            self.result_label.config(text=f"Correct: {self.correct_answers}, Wrong: {self.wrong_answers}")

            self.display_next_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Test App")
    app = ImageClassificationApp(root, dataset_path)
    root.mainloop()

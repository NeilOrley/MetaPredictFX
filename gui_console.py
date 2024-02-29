import tkinter as tk
from program_object import execute_program_with
from program_object_without import execute_program_without
import sys

class Aplication:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepLearning Forecast")
        
        # labels and entries
        label1 = tk.Label(root, text="Input account mt5:")
        label1.pack(pady=5)

        self.value1_entry = tk.Entry(root)
        self.value1_entry.pack(pady=5)

        label2 = tk.Label(root, text="Input password:")
        label2.pack(pady=5)

        self.value2_entry = tk.Entry(root)
        self.value2_entry.pack(pady=5)

        label3 = tk.Label(root, text="Input server of mt5:")
        label3.pack(pady=5)

        self.value3_entry = tk.Entry(root)
        self.value3_entry.pack(pady=5)

        label4 = tk.Label(root, text="Input delay in days (for ticks):")
        label4.pack(pady=5)

        self.value4_entry = tk.Entry(root)
        self.value4_entry.pack(pady=5)

        label5 = tk.Label(root, text="Input timeframe 1 (1h),2,4,1d,1w:")
        label5.pack(pady=5)

        self.value5_entry = tk.Entry(root)
        self.value5_entry.pack(pady=5)

        label6 = tk.Label(root, text="Input epochs:")
        label6.pack(pady=5)

        self.value6_entry = tk.Entry(root)
        self.value6_entry.pack(pady=5)

        label7 = tk.Label(root, text="Input symbol:")
        label7.pack(pady=5)

        self.value7_entry = tk.Entry(root)
        self.value7_entry.pack(pady=5)


        label8 = tk.Label(root, text="Input path for mt5:")
        label8.pack(pady=5)

        self.value8_entry = tk.Entry(root)
        self.value8_entry.pack(pady=5)

        # Radio button to select program to execute
        self.opcion_var = tk.StringVar(value="program_object")
        radio_btn_object = tk.Radiobutton(root, text="With r2 Score, MAE & MSE", variable=self.opcion_var, value="program_object")
        radio_btn_object.pack(pady=5)
        radio_btn_object_without = tk.Radiobutton(root, text="Without", variable=self.opcion_var, value="program_object_without")
        radio_btn_object_without.pack(pady=5)

        # Botón start
        boton_execute = tk.Button(root, text="Run Program", command=self.execute_programa)
        boton_execute.pack(pady=10)

        # Botón close
        boton_quit = tk.Button(root, text="Exit", command=root.destroy)
        boton_quit.pack(pady=10)

    def write(self, text):
        # this method y called when sys.stdout.write is used
        self.salida_text.insert(tk.END, text)
    def flush(self):
        pass
    def execute_program(self):
        # Obteined value of the selected option
        selected_program = self.opcion_var.get()

        # Obteined value of inputs
        value1 = self.value1_entry.get()
        value2 = self.value2_entry.get()
        value3 = self.value3_entry.get()
        value4 = self.value4_entry.get()
        value5 = self.value5_entry.get()
        value6 = self.value6_entry.get()
        value7 = self.value7_entry.get()
        value8 = self.value8_entry.get()

                # Redirects stdout & stderr to the console
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Calls the function to execute a selected program and pass values as arguments
        if selected_program == "program_object":
            execute_program_with(value1, value2, value3, value4, value5, value6, value7, value8)
        elif selected_program == "program_object_without":
            execute_program_without(value1, value2, value3, value4, value5, value6, value7, value8)
        # Restores stdout to a predeterminate value to no have conflicts
        sys.stdout = self
        sys.stderr = self

if __name__ == "__main__":
    root = tk.Tk()
    app = Aplication(root)
    root.mainloop()

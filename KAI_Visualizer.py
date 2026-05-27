import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, func, model, min, max):
        plt.ion()

        self.x = np.linspace(min, max, 400)
        self.func = func
        self.model = model

        self.fig, self.ax = plt.subplots()

        self.actual_line, = self.ax.plot(self.x, [self.func(x) for x in self.x], label="Actual")
        self.pred_line, = self.ax.plot(self.x, [self.model.calculate([x]) for x in self.x], label="Prediction")

        self.ax.legend()
        self.ax.grid(True)

    def update(self):
        self.actual_line.set_data(self.x, [self.func(x) for x in self.x])
        predicted_y_values = []
        for x in self.x:
            self.model.calculate([x])
            predicted_y_values.append(self.model.outputs[0])

        self.pred_line.set_data(self.x, predicted_y_values)

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.show()
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, trainingDataSet, model):
        plt.ion()

        self.trainingDataSet = trainingDataSet
        self.x = []
        self.inputs = []
        self.y = []
        for trainingData in trainingDataSet:
            self.x.append(trainingData.x_index)
            self.inputs.append(trainingData.inputs)
            self.y.append(trainingData.outputs)
        self.model = model

        self.fig, self.ax = plt.subplots()

        self.actual_line, = self.ax.plot(self.x, self.y, label="Actual")
        self.pred_line, = self.ax.plot(self.x, [self.model.calculate(input) for input in self.inputs], label="Prediction")

        self.ax.legend()
        self.ax.grid(True)

    def update(self):
        self.actual_line.set_data(self.x, self.y)
        predicted_y_values = []
        for input in self.inputs:
            self.model.calculate(input)
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
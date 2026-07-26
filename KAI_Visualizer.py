import matplotlib.pyplot as plt

ACTUAL_COLOR = "blue"
PRED_COLORS = [
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
]

class Visualizer:
    def __init__(self, trainingDataSet, models):
        plt.ion()

        self.trainingDataSet = sorted(
            trainingDataSet,
            key=lambda x: x.x_index
        )
        self.output_count = len(trainingDataSet[0].outputs)
        self.models = models

        self.fig, (self.ax, self.error_ax) = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            gridspec_kw={"height_ratios": [3, 1]}
        )
        self.fig.canvas.manager.set_window_title("KAI v2.0 Neural Network Training Dashboard")

        self.x = []
        for trainingData in trainingDataSet:
            self.x.append(trainingData.x_index)

        self.actual_lines = []
        for output_index in range(self.output_count):
            actual = []
            for trainingData in trainingDataSet:
                actual.append(trainingData.outputs[output_index])
            line, = self.ax.plot(
                self.x,
                actual,
                color=ACTUAL_COLOR,
                linewidth=3,
                alpha=0.9,
                label="Actual" if output_index == 0 else None
            )
            self.actual_lines.append(line)

        self.pred_lines_list = []
        for model_index, model in enumerate(self.models):
            prediction_lines = []
            predictions = [[] for _ in range(self.output_count)]
            for trainingData in trainingDataSet:
                model.calculate(trainingData.inputs)
                for output_index, value in enumerate(model.outputs):
                    predictions[output_index].append(value)
            for output_index in range(self.output_count):
                line, = self.ax.plot(
                    self.x,
                    predictions[output_index],
                    color=PRED_COLORS[model_index % len(PRED_COLORS)],
                    alpha=0.3,
                    label=f"Prediction {model_index}" if output_index == 0 else None
                )
                prediction_lines.append(line)
            self.pred_lines_list.append(prediction_lines)

        self.ax.set_title("Prediction")
        self.ax.grid(True)
        self.ax.legend()

        self.error_history = [[] for _ in self.models]
        self.error_lines = []

        for model_index in range(len(self.models)):
            line, = self.error_ax.plot(
                [],
                [],
                color=PRED_COLORS[model_index % len(PRED_COLORS)],
                label=f"Model {model_index}"
            )

            self.error_lines.append(line)

        self.error_ax.set_title("Training Error")
        self.error_ax.set_xlabel("Generation")
        self.error_ax.set_ylabel("Average Error")
        self.error_ax.set_ylim(0, 1)
        self.error_ax.grid(True)
        self.error_ax.legend()
    
    def update(self, errors):
        for output_index in range(self.output_count):
            actual = []
            for trainingData in self.trainingDataSet:
                actual.append(trainingData.outputs[output_index])
            self.actual_lines[output_index].set_data(
                self.x,
                actual
            )

        for model_index, model in enumerate(self.models):
            predictions = [[] for _ in range(self.output_count)]
            for trainingData in self.trainingDataSet:
                model.calculate(trainingData.inputs)
                for output_index, value in enumerate(model.outputs):
                    predictions[output_index].append(value)
            for output_index in range(self.output_count):
                self.pred_lines_list[model_index][output_index].set_data(
                    self.x,
                    predictions[output_index]
                )
        self.ax.relim()
        self.ax.autoscale_view()

        for model_index, error in enumerate(errors):
            self.error_history[model_index].append(error)
            self.error_lines[model_index].set_data(
                range(len(self.error_history[model_index])),
                self.error_history[model_index]
            )
        self.error_ax.relim()
        self.error_ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def close(self, best_model_index):
        for model_index, prediction_lines in enumerate(self.pred_lines_list):
            if model_index == best_model_index:
                for line in prediction_lines:
                    line.set_alpha(1)
                    line.set_linewidth(3)
            else:
                for line in prediction_lines:
                    line.set_alpha(0.15)
        
        for model_index, error_line in enumerate(self.error_lines):
            if model_index == best_model_index:
                error_line.set_alpha(1)
                error_line.set_linewidth(1.5)
            else:
                error_line.set_alpha(0.15)
        self.ax.legend()
        self.error_ax.legend()
        self.fig.canvas.draw()
        plt.ioff()
        plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .model import Model
from .config import monster_colors, brief_monster_labels_16
from typing import Optional, Tuple, List

class Visualizer:
    
    def __error_check(self, 
                 model: Model,
                 plot_only: Optional[List[int]] = None,
                 figure_size: Optional[Tuple[int, int]] = (12, 8),
                 background_color: Optional[Tuple[float, float, float]] = (.1, .1, .1),
                 legend_color: Optional[Tuple[float, float, float]] = (.753, .753, .753),
                 line_colors: Optional[List[str]] = monster_colors,
                 line_labels: Optional[List[str]] = brief_monster_labels_16, 
                 animation_frames: Optional[int] = 1000,
    ) -> Tuple[bool, Exception]:
        
        #Check for types
        if not isinstance(model, Model):
            return (False, ValueError("`model` must be of type `Model`."))

        if not isinstance(plot_only, List):
            return (False, ValueError("`plot_only` must be of type `list`."))

        if not isinstance(figure_size, Tuple):
            return (False, ValueError("`figure_size` must be of type `tuple`."))

        if not isinstance(background_color, Tuple):
            return (False, ValueError("`background_color` must be of type `tuple`."))

        if not isinstance(legend_color, Tuple):
            return (False, ValueError("`legend_color` must be of type `Model`."))

        if not isinstance(line_colors, List):
            return (False, ValueError("`line_color` must be of type `list`."))

        if not isinstance(line_labels, List):
            return (False, ValueError("`line_labels` must be of type `list`."))

        if not isinstance(animation_frames, int):
            return (False, ValueError("`animation_frames` must be of type `int`."))

        #Check for values:
        dim = model.dimension

        #Check that plot_only contains numbers in the range (0, dim)
        check_range = range(0, dim)
        for i in plot_only:
            if i not in check_range:
                return (False, ValueError("`plot_only` must be a subset of `range(0, model.dimension)`."))

        #Check that len(line_labels) >= dim
        if len(line_labels) < dim:
            return (False, ValueError("`line_labels` must contains at least `model.dimension` elements."))

        #Check that len(line_colors) >= dim
        if len(line_colors) < dim:
            return (False, ValueError("`line_colors` must contains at least `model.dimension` elements."))

        #Check that animation frames are positive

        if animation_frames <= 0:
            return (False, ValueError("`animation_frames` must be a positive integer."))

        return (True, None)

    def __init__(self, 
                 model: Model,
                 plot_only: Optional[List[int]] = None,
                 figure_size: Optional[Tuple[int, int]] = (12, 8),
                 background_color: Optional[Tuple[float, float, float]] = (.1, .1, .1),
                 legend_color: Optional[Tuple[float, float, float]] = (.753, .753, .753),
                 line_colors: Optional[List[str]] = monster_colors,
                 line_labels: Optional[List[str]] = brief_monster_labels_16, 
                 animation_frames: Optional[int] = 1000,
                 run_checks: Optional[bool] = False
    ) -> None:

        if run_checks is True:
            report = self.__error_check(model,
                                        plot_only,
                                        figure_size,
                                        background_color,
                                        legend_color,
                                        line_colors,
                                        line_labels,
                                        animation_frames)

            if report[0] is False:
                raise report[1]

        if plot_only is None:
            self._plot = np.arange(0, model.dimension)
        else:
            self._plot = plot_only

        #Create Class properties.
        self._model: Model                                      = model
        self._figsize: Tuple[int, int]                          = figure_size
        self._background_color: Tuple[float, float, float]      = background_color
        self._legend_color: Tuple[float, float, float]          = legend_color
        self._line_colors: List[str]                            = line_colors
        self._line_labels: List[str]                            = line_labels
        self._timepoints = np.linspace(0, 1, self._model.sampling_frequency)

        #Need these for animation.
        self._ani_frames: int      = animation_frames if animation_frames > 0 else 1000
        self._frame_offset: int    = (self._model.sampling_frequency // self._ani_frames)  #Determines how many datapoints will be drawn for each frame.

    #General Plotting Method
    def _plotter(self, evolution: List, y_label: str, legend: Optional[bool] = True, label_numbers: Optional[bool] = True) -> None:
        
        fig, ax = plt.subplots(figsize = self._figsize)
        ax.set(xlim = [0, 1], xlabel = "Time", ylabel = y_label, facecolor = self._background_color)

        for i in self._plot:
            plt.plot(self._timepoints, evolution[i], color = self._line_colors[i], label= str(i) + ". " + self._line_labels[i] if label_numbers else self._line_labels[i])
        
        if legend:
            plt.legend(facecolor = self._legend_color, loc="upper left")
     
        plt.show()
        plt.close(fig=fig)

    #Plots Population Percentage Vs Time
    def percentage_plot(self, legend: Optional[bool] = True, label_numbers: Optional[bool] = True) -> None:
        self._plotter(self._model.population_evolution, y_label="Population Percentage", legend=legend, label_numbers=label_numbers)

    #Plots Population Derivative Vs Time
    def derivative_plot(self, legend: Optional[bool] = True, label_numbers: Optional[bool] = True ) -> None:
        self._plotter(self._model.derivative_evolution, y_label="Population Derivative", legend=legend, label_numbers=label_numbers)

    #Plots Fitness Vs Time
    def fitness_plot(self, legend: Optional[bool] = True, label_numbers: Optional[bool] = True ) -> None:
        self._plotter(self._model.fitness_evolution, y_label="Fitness", legend=legend, label_numbers=label_numbers)

    #Plots Avg Fitness Vs Time
    def avg_fitness_plot(self) -> None:

        fig, ax = plt.subplots(figsize = self._figsize)
        ax.set(xlim = [0, 1], xlabel = "Time", ylabel = "Average Fitness", facecolor = self._background_color)

        for i in self._plot:
            plt.plot(self._timepoints, self._model.avg_fitness_evolution, color = "#4169e1")
        
        plt.show()
        plt.close(fig=fig)

    #Update function for animation functions. 
    def __update(self, 
                 frame, 
                 yy_data: List, 
                 lines: List
    ) -> Tuple:
        for i in self._plot:
            lines[i].set_xdata(self._timepoints[:frame * self._frame_offset])
            lines[i].set_ydata(yy_data[i][:frame * self._frame_offset])
        return tuple(lines)

    #General Animator Method
    def __animator(self, 
                   evolution: List, 
                   y_label: str, 
                   y_lim: Optional[Tuple[float, float]] = None, 
                   legend: Optional[bool] = True, 
                   label_numbers: Optional[bool] = True
    ) -> None:

        fig, ax = plt.subplots(figsize=self._figsize)
        
        ax.set(xlim=(0, 1), xlabel="Time", ylabel=y_label, facecolor=self._background_color)

        if y_lim is not None:
            ax.set_ylim(y_lim)

        yy_data = [evolution[i] for i in self._plot]
        lines = [ax.plot(self._timepoints[0], yy_data[i][0], color=self._line_colors[i], label= str(i) + ". " + self._line_labels[i] if label_numbers else self._line_labels[i])[0] for i in self._plot]
        
        if legend:
            #ax.legend(loc="upper center", fancybox=True, bbox_to_anchor=(-0.165, 1.15), facecolor = self._legend_color, ncol = 4)
            ax.legend(loc="upper center", fancybox=True, bbox_to_anchor = (0.5, 1.15), facecolor = self._legend_color, ncol = 4)

        ani = animation.FuncAnimation(fig=fig, func=self.__update, fargs=(yy_data, lines), frames=self._ani_frames, interval=1, blit=True, repeat=False)

        plt.show()
        plt.close(fig=fig)

    #Animates Population Percentage Vs Time
    def percentage_animation(self, extr: Optional[bool] = False, legend: Optional[bool] = True, label_numbers: Optional[bool] = True) -> None:
        if not extr:
            self.__animator(self._model.population_evolution, y_lim = (0, 1), y_label="Population Percentage", legend=legend, label_numbers=label_numbers)
        else:
            low, high = self._model.population_percentage_extrema
            self.__animator(self._model.population_evolution, y_lim = (1.15 * low, 1.07 * high), y_label="Population Percentage", legend=legend, label_numbers=label_numbers)

    #Animates Population Derivative Vs Time
    def derivative_animation(self, extr: Optional[bool] = True, legend: Optional[bool] = True, label_numbers: Optional[bool] = True) -> None:
        if not extr:
            self.__animator(self._model.derivative_evolution, y_label="Population Derivative", legend=legend, label_numbers=label_numbers)
        else:
            low, high = self._model.population_derivative_extrema
            self.__animator(self._model.derivative_evolution, y_lim = (1.15 * low, 1.07 * high), y_label="Population Derivative", legend=legend, label_numbers=label_numbers)
    
    #Animates Population Fitness Vs Time
    def fitness_animation(self, extr: Optional[bool] = True, legend: Optional[bool] = True, label_numbers: Optional[bool] = True) -> None:
        if not extr:
            self.__animator(self._model.fitness_evolution, y_label="Fitness", legend=legend, label_numbers=label_numbers)
        else:
            low, high = self._model.fitness_extrema
            self.__animator(self._model.fitness_evolution, y_lim = (low, high), y_label="Fitness", legend=legend, label_numbers=label_numbers)

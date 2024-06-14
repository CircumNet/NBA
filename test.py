import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox

# Create a simple plot (you can customize this)
fig, ax = plt.subplots()
t = np.arange(-2.0, 2.0, 0.001)
l, = ax.plot(t, np.zeros_like(t), lw=2)

def submit(expression):
    """
    Update the plotted function based on the new math expression.
    The expression should use "t" as the independent variable.
    Example: "t ** 2"
    """
    ydata = eval(expression, {'np': np}, {'t': t})
    l.set_ydata(ydata)
    ax.relim()
    ax.autoscale_view()
    plt.draw()

# Add a text box for user input
fig.subplots_adjust(bottom=0.2)
axbox = fig.add_axes([0.1, 0.05, 0.8, 0.075])
text_box = TextBox(axbox, "Evaluate", textalignment="center")
text_box.on_submit(submit)
text_box.set_val("t ** 2")  # Initial expression

plt.show()

"""One place for every message to the people around the robot.

The director calls say(state, text) and does not know where it goes. Today the
only output is the laptop screen. An LCD, LEDs or a buzzer on a second board are
added later as one more output function, without touching the director:

    def lcd(state, text): ...                  # e.g. red light while state == "ROBOT_MOVING"
    announcer = Announcer(console, lcd)

state is a short word chosen by the director ("YOUR_TURN", "COUNTDOWN", ...);
text is the sentence for a person.
"""


def console(state, text):
    print(f"[{state}] {text}", flush=True)


class Announcer:
    def __init__(self, *outputs):
        self.outputs = outputs or (console,)
        self.last = None

    def say(self, state, text=""):
        """Send a message to every output. The same message twice in a row is sent once."""
        if (state, text) == self.last:
            return
        self.last = (state, text)
        for output in self.outputs:
            try:
                output(state, text)
            except Exception as err:  # a broken display must never stop the robot mid-move
                print(f"[ANNOUNCE] output failed: {err}", flush=True)

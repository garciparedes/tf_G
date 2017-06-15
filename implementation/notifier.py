class Notifier:
    def __init__(self):
        self._listeners = set()

    def attach(self, observer):
        self._listeners.add(observer)

    def detach(self, listener):
        self._listeners.discard(listener)

    def _notify(self):
        for observer in self._listeners:
            observer.update()

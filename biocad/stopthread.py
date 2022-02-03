from threading import Thread


class StopThread(Exception):
    pass


class StoppableThread(Thread):
    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)

    def run(self):
        try:
            super(StoppableThread, self).run()
        except StopThread:
            return

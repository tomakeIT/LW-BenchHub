from collections import deque


class ContactQueue:
    def __init__(self):
        self.queue = deque()

    def is_empty(self):
        return len(self.queue) == 0

    def add(self, contact_view):
        self.queue.append(contact_view)

    def pop(self):
        if self.is_empty():
            return None
        contact_view = self.queue.popleft()
        self.queue.append(contact_view)
        return contact_view

    def clear(self):
        self.queue.clear()

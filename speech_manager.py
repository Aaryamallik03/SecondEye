import threading
import time
import subprocess
import queue


class SpeechManager:

    def __init__(self, cooldown=1.5):

        self.cooldown = cooldown
        self.running = True

        self.last_spoken = {}
        self.lock = threading.Lock()

        self.speech_queue = queue.PriorityQueue()

        self.worker = threading.Thread(
            target=self._speech_worker,
            daemon=True
        )

        self.worker.start()

    # =========================
    # SPEECH WORKER
    # =========================

    def _speech_worker(self):

        while self.running:

            try:

                priority, timestamp, text = (
                    self.speech_queue.get(
                        timeout=0.1
                    )
                )

            except queue.Empty:

                continue

            if text is None:

                self.speech_queue.task_done()
                break

            try:

                print("VOICE:", text)

                # Escape apostrophes for PowerShell
                safe_text = text.replace(
                    "'",
                    "''"
                )

                subprocess.run(
                    [
                        "powershell",
                        "-Command",
                        (
                            "Add-Type -AssemblyName "
                            "System.Speech; "
                            "$s = New-Object "
                            "System.Speech.Synthesis."
                            "SpeechSynthesizer; "
                            "$s.Rate = 1; "
                            f"$s.Speak('{safe_text}')"
                        )
                    ],
                    creationflags=subprocess.CREATE_NO_WINDOW
                )

            except Exception as e:

                print(
                    "Speech error:",
                    e
                )

            finally:

                self.speech_queue.task_done()

    # =========================
    # QUEUE SPEECH
    # =========================

    def _queue_speech(
        self,
        text,
        event_key,
        priority=0
    ):

        now = time.time()

        with self.lock:

            last_time = self.last_spoken.get(
                event_key,
                0
            )

            if (
                now - last_time
                < self.cooldown
            ):

                return

            self.last_spoken[
                event_key
            ] = now

        self.speech_queue.put(
            (
                priority,
                now,
                text
            )
        )

    # =========================
    # PERSON ANNOUNCEMENT
    # =========================

    def announce_person(
        self,
        identity,
        position
    ):

        if identity == "Unknown":

            if position == "left":

                text = (
                    "Unknown person on your left."
                )

            elif position == "right":

                text = (
                    "Unknown person on your right."
                )

            else:

                text = (
                    "Unknown person in front of you."
                )

        else:

            if position == "left":

                text = (
                    f"{identity} is on your left."
                )

            elif position == "right":

                text = (
                    f"{identity} is on your right."
                )

            else:

                text = (
                    f"{identity} is in front of you."
                )

        event_key = (
            f"person:{identity}:{position}"
        )

        self._queue_speech(
            text,
            event_key,
            priority=0
        )

    # =========================
    # NEW PERSON
    # =========================

    def announce_new_person(
        self,
        identity,
        position
    ):

        if identity == "Unknown":

            text = "Unknown person detected."

        else:

            text = (
                f"{identity} detected."
            )

        event_key = (
            f"new:{identity}:{position}"
        )

        self._queue_speech(
            text,
            event_key,
            priority=1
        )

    # =========================
    # OCR / GENERAL SPEECH
    # =========================

    def announce_text(
        self,
        text
    ):

        event_key = (
            f"text:{text}"
        )

        self._queue_speech(
            text,
            event_key,
            priority=0
        )

    # =========================
    # STOP
    # =========================

    def stop(self):

        self.running = False

        try:

            self.speech_queue.put(
                (
                    -1,
                    time.time(),
                    None
                )
            )

        except Exception:

            pass
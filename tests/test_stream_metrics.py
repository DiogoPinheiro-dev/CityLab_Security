import ast
import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.benchmark_stream import summarize


class StreamMetricsTest(unittest.TestCase):
    def run_stream(self, waits=(10,), processing_seconds=.2):
        tree = ast.parse(Path('Server/main.py').read_text(encoding='utf-8'))
        handler = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == 'websocket_reconhecimento')
        handler.decorator_list = []
        handler.args.args[0].annotation = None
        clock = [0.0]
        recorded = []
        responses = []
        resets = []

        class Disconnect(Exception):
            pass

        class Socket:
            async def accept(self):
                pass

            async def receive_bytes(self):
                if len(responses) == len(waits):
                    raise Disconnect()
                clock[0] += waits[len(responses)]
                return b'jpeg'

            async def send_json(self, payload):
                responses.append(dict(payload['metrics']))
                clock[0] += 0.5

        def process(frame):
            clock[0] += processing_seconds
            return {'metrics': {'total_ms': 999, 'effective_fps': 999}}

        async def log(*args):
            pass

        namespace = dict(time=SimpleNamespace(perf_counter=lambda: clock[0]),
                         recognizer=SimpleNamespace(process_frame=process,
                             reset_gesture_history=lambda: resets.append(True)),
                         np=SimpleNamespace(frombuffer=lambda *args: None, uint8=None),
                         cv2=SimpleNamespace(imdecode=lambda *args: object(), IMREAD_COLOR=1),
                         event_logger=SimpleNamespace(log_face_events=log, log_gesture_events=log),
                         system_monitor=SimpleNamespace(resource_snapshot=lambda: {},
                             record_frame_metrics=lambda metrics: recorded.append(dict(metrics)),
                             maybe_log_snapshot=lambda: None),
                         DEBUG_PIPELINE=False, ENABLE_PERFORMANCE_METRICS=True,
                         GESTURE_IDLE_RESET_SECONDS=5,
                         WebSocketDisconnect=Disconnect, asyncio=asyncio)
        exec(compile(ast.Module(body=[handler], type_ignores=[]), '<handler>', 'exec'), namespace)
        asyncio.run(namespace['websocket_reconhecimento'](Socket()))
        return responses, recorded, resets

    def test_wait_excluded_and_send_included(self):
        responses, recorded, resets = self.run_stream()
        self.assertAlmostEqual(responses[0]['receive_wait_ms'], 10000)
        self.assertAlmostEqual(responses[0]['response_ready_ms'], 200)
        self.assertNotIn('total_ms', responses[0])
        self.assertAlmostEqual(recorded[0]['total_ms'], 700)
        self.assertAlmostEqual(recorded[0]['send_ms'], 500)
        self.assertEqual(len(resets), 1)

    def test_slow_inference_is_not_a_pause_but_idle_is(self):
        responses, _, resets = self.run_stream(waits=(0, 0, 6, 0), processing_seconds=17)
        self.assertEqual(len(responses), 4)
        self.assertEqual(len(resets), 2)

    def test_summary(self):
        self.assertEqual(summarize([200, 100, 400, 300]), {'count': 4, 'median': 250, 'p95': 400})
        self.assertIsNone(summarize([])['p95'])


if __name__ == '__main__':
    unittest.main()

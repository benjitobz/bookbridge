import base64
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

import requests

from src.api.api_clients import KoSyncClient


class _CwaKoSyncHandler(BaseHTTPRequestHandler):
    expected_auth = "Basic " + base64.b64encode(b"reader:cwa-password").decode("ascii")
    last_put = None

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == self.expected_auth

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/kosync/healthcheck":
            self._json_response(404, {})
            return
        if not self._authorized():
            self._json_response(401, {"error": 2001, "message": "Unauthorized"})
            return
        if self.path == "/kosync/syncs/progress/test-connection":
            self._json_response(200, {})
            return
        if self.path == "/kosync/syncs/progress/doc-1":
            self._json_response(200, {
                "document": "doc-1",
                "percentage": 0.42,
                "progress": "/body/DocFragment[1]/body/p[1]/text().0",
            })
            return
        if self.path == "/kosync/syncs/progress/doc-null":
            self._json_response(200, {
                "document": "doc-null",
                "percentage": None,
                "progress": None,
            })
            return
        self._json_response(404, {})

    def do_PUT(self) -> None:
        if not self._authorized():
            self._json_response(401, {"error": 2001, "message": "Unauthorized"})
            return
        if self.path != "/kosync/syncs/progress":
            self._json_response(404, {})
            return
        length = int(self.headers.get("Content-Length", "0"))
        type(self).last_put = json.loads(self.rfile.read(length))
        self._json_response(200, {"document": "doc-1", "timestamp": 1700000000})

    def log_message(self, format: str, *args) -> None:
        pass


class TestKoSyncClientBasicAuth(unittest.TestCase):
    def setUp(self) -> None:
        _CwaKoSyncHandler.last_put = None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _CwaKoSyncHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = KoSyncClient(credentials={
            "KOSYNC_ENABLED": "true",
            "KOSYNC_SERVER": f"http://{host}:{port}/kosync",
            "KOSYNC_USER": "reader",
            "KOSYNC_KEY": "cwa-password",
            "KOSYNC_AUTH_METHOD": "basic",
        })

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_cwa_basic_auth_connection_read_and_write(self) -> None:
        self.assertTrue(
            self.client.check_connection(),
            "CWA credentials should not fail with KoSync connection Response: 401",
        )

        percentage, progress, metadata = self.client.get_progress_with_metadata("doc-1")
        self.assertEqual(percentage, 0.42)
        self.assertEqual(progress, "/body/DocFragment[1]/body/p[1]/text().0")
        self.assertEqual(metadata["document"], "doc-1")

        self.assertTrue(self.client.update_progress("doc-1", 0.5, progress))
        self.assertEqual(_CwaKoSyncHandler.last_put["document"], "doc-1")
        self.assertEqual(_CwaKoSyncHandler.last_put["percentage"], 0.5)

    def test_null_percentage_is_treated_as_no_progress(self) -> None:
        with self.assertNoLogs("src.api.api_clients", level="ERROR"):
            percentage, progress, metadata = (
                self.client.get_progress_with_metadata("doc-null")
            )

        self.assertIsNone(percentage)
        self.assertIsNone(progress)
        self.assertEqual(metadata["document"], "doc-null")

    def test_local_startup_connection_refusal_is_not_an_error(self) -> None:
        self.client._creds["KOSYNC_SERVER"] = "http://127.0.0.1:5758"
        with patch.object(
            self.client.session,
            "get",
            side_effect=requests.exceptions.ConnectionError("connection refused"),
        ), self.assertNoLogs("src.api.api_clients", level="ERROR"):
            result = self.client.get_progress_with_metadata("startup-doc")

        self.assertEqual(result, (None, None, {}))


class TestKoSyncClientBuiltinDefault(unittest.TestCase):
    def test_empty_server_writes_to_builtin_port(self) -> None:
        for port, expected_port in (("", 5757), ("5758", 5758)):
            with self.subTest(port=port), patch.dict(
                "os.environ",
                {
                    "KOSYNC_ENABLED": "true",
                    "KOSYNC_SERVER": "",
                    "KOSYNC_PORT": port,
                },
            ):
                client = KoSyncClient(credentials={
                    "KOSYNC_USER": "reader",
                    "KOSYNC_KEY": "password",
                })
                with patch.object(
                    client.session, "put", return_value=Mock(status_code=200)
                ) as put:
                    self.assertTrue(client.is_configured())
                    self.assertTrue(client.update_progress("doc-1", 0.3, "/body/p.0"))

                self.assertEqual(
                    put.call_args.args[0],
                    f"http://127.0.0.1:{expected_port}/syncs/progress",
                )


if __name__ == "__main__":
    unittest.main()

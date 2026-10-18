"""Independent generated-only native ASGI checks; no sockets/model/engine."""
import asyncio
from pathlib import Path
import sys

import httpx
import pytest

PREPARED = Path(__file__).resolve().parents[2] / "training/private/qwen38-vllm-integration-a10/prepared"
sys.path.insert(0, str(PREPARED / "tests"))
from test_qwen38_vllm_provider import packet, synthetic_ready
from test_qwen38_vllm_boundary import setup_boundary, generated_payload, headers
from serving.qwen38_identity import CandidateError


def test_websocket_never_reaches_unwrapped_native_application():
    async def run():
        _, boundary, engine = setup_boundary()
        async def forbidden(*args):
            raise AssertionError("Non-HTTP native route delegated before auth")
        boundary.app = forbidden
        sent = []
        async def receive(): return {"type": "websocket.connect"}
        async def send(event): sent.append(event)
        await boundary({"type": "websocket", "path": "/v1/realtime", "headers": []}, receive, send)
        assert sent == [{"type": "websocket.close", "code": 1008}]
        assert not engine.calls
    asyncio.run(run())


def test_duplicate_authorization_rejects_before_body_or_model():
    async def run():
        _, boundary, engine = setup_boundary()
        async def receive(): raise AssertionError("Duplicate auth must precede body")
        sent = []
        async def send(event): sent.append(event)
        await boundary({"type": "http", "path": "/v1/chat/completions", "method": "POST",
            "headers": [(b"authorization", b"Bearer generated-private-key")] * 2}, receive, send)
        assert sent[0]["status"] == 401 and not engine.calls
    asyncio.run(run())


@pytest.mark.parametrize("raw", [b"{", b"[]", b'{"n":1,"n":2}'])
def test_authenticated_malformed_json_is_bounded_input_error(raw):
    async def run():
        config, boundary, engine = setup_boundary()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=boundary), base_url="http://native") as client:
            response = await client.post("/v1/chat/completions", headers=headers(config, boundary), content=raw)
        assert response.status_code == 400
        assert not engine.calls and not boundary.invalidated
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["empty", "wrong", "other_mode"])
def test_actual_sampling_schema_must_match_the_http_bound_canonical_schema(packet, mode):
    async def run():
        config, boundary, engine = setup_boundary(actual_grammar=mode)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=boundary), base_url="http://native") as client:
            response = await client.post("/v1/chat/completions", headers=headers(config, boundary),
                json=generated_payload(packet, config))
        assert response.status_code == 502 and not engine.calls and boundary.invalidated
    asyncio.run(run())


def test_finished_output_with_unverified_drain_does_not_admit_the_next_request(packet):
    async def run():
        config, boundary, engine = setup_boundary()
        engine.get_num_unfinished_requests = lambda: 1
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=boundary), base_url="http://native") as client:
            first = await client.post("/v1/chat/completions", headers=headers(config, boundary),
                json=generated_payload(packet, config))
            second = await client.post("/v1/chat/completions", headers=headers(config, boundary),
                json=generated_payload(packet, config))
        assert (first.status_code, second.status_code) == (502, 503)
        assert len(engine.calls) == 1 and engine.aborts and boundary.invalidated
    asyncio.run(run())


def test_six_generated_same_engine_calls_are_required_before_either_role_is_ready(tmp_path):
    from deploy.qwen38_vllm_pilot.request import prepare
    async def run():
        config, boundary, engine = setup_boundary()
        proof = synthetic_ready(config, "base_diagnostic", engine=boundary.instance)
        boundary.proofs.clear()
        case = prepare(Path.cwd(), tmp_path)
        boundary.stage_generated_acceptance(case, loaded_engine=proof["loaded_engine"],
            native_grammar=proof["native_grammar"])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=boundary, client=("127.0.0.1", 1234)),
                                     base_url="http://native") as client:
            for index in range(6):
                role = "base_diagnostic" if index % 2 == 0 else "fit_candidate"
                request_headers = {**headers(config, boundary, role),
                    "X-Forte-Native-Generated-Gate": boundary.bootstrap_token}
                response = await client.post("/v1/chat/completions", headers=request_headers,
                    json=boundary.generated_payloads[role])
                assert response.status_code == 200
                ready = await client.get("/readyz", headers=headers(config, boundary, role))
                assert ready.status_code == 503
                if index == 4:
                    with pytest.raises(CandidateError):
                        boundary.finish_generated_acceptance()
            gate = boundary.finish_generated_acceptance()
            for role in config["roles"]:
                ready = await client.get("/readyz", headers=headers(config, boundary, role))
                assert ready.status_code == 200
                assert ready.json()["engine_instance_id"] == boundary.instance
        assert gate["attempts"] == len(gate["calls"]) == len(engine.calls) == 6
        assert gate["successful_calls_per_role"] == {"base_diagnostic": 3, "fit_candidate": 3}
        assert boundary.bootstrap_token is None and not engine.aborts
    asyncio.run(run())

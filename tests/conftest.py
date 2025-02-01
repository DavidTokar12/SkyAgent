from __future__ import annotations

import json
import os

import pytest
import vcr


allowed_headers = {"Content-Encoding", "Content-Type"}


def remove_sensitive_headers(request):
    """Remove all headers from the request before saving."""
    request.headers.clear()
    return request


def remove_sensitive_response_headers(response):
    """Remove all response headers except 'Content-Encoding' and 'Content-Type'."""
    allowed_headers = {"Content-Encoding", "Content-Type"}
    response["headers"] = {
        key: value
        for key, value in response["headers"].items()
        if key in allowed_headers
    }
    return response


my_vcr = vcr.VCR(
    cassette_library_dir=os.path.join(os.path.dirname(__file__), "cassettes"),
    record_mode="once",
    filter_headers=["authorization"],
    before_record=remove_sensitive_headers,
    before_record_response=remove_sensitive_response_headers,
)


@pytest.fixture(autouse=True)
def set_dummy_api_keys(monkeypatch):
    # If we're running on GitHub Actions, set dummy API keys.
    # GitHub Actions usually sets the environment variable GITHUB_ACTIONS to "true".
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_openai_token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_anthropic_token")


@pytest.fixture
def vcr_fixture():
    return my_vcr


@pytest.fixture(scope="session")
def test_inputs():
    test_inputs_path = os.path.join(os.path.dirname(__file__), "test_inputs.json")
    with open(test_inputs_path) as f:
        return json.load(f)


def get_cassette_name(api_type: str, input_name: str) -> str:
    return f"{api_type}_{input_name}.json"


@pytest.fixture
def base_input(test_inputs):
    input_name = "base_input"
    input_data = test_inputs[input_name]
    return input_name, input_data

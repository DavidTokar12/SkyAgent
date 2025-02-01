from __future__ import annotations

from conftest import get_cassette_name

from skyagent.open_ai.open_ai_agent import OpenAIAgent


def test_base_openai(vcr_fixture, base_input):
    """Use the saved VCR cassette in CI/CD."""
    input_name, input_data = base_input
    cassette_name = get_cassette_name("openai", input_name)

    with vcr_fixture.use_cassette(cassette_name):

        agent = OpenAIAgent(
            name="Test",
            model=input_data["models"]["open_ai"],
            system_prompt=input_data["system_prompt"],
        )

        result = agent.call_agent(
            query=input_data["query"],
        )

        assert result

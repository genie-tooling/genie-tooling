import logging

import pytest
from genie_tooling.guardrails.impl.keyword_blocklist import (
    KeywordBlocklistGuardrailPlugin,
)

GUARDRAIL_LOGGER_NAME = "genie_tooling.guardrails.impl.keyword_blocklist"


@pytest.fixture
async def keyword_guardrail() -> KeywordBlocklistGuardrailPlugin:
    guardrail = KeywordBlocklistGuardrailPlugin()
    # Default setup: empty blocklist, case_sensitive=False, action_on_match="block"
    await guardrail.setup()
    return guardrail


@pytest.mark.asyncio
async def test_setup_default_config(keyword_guardrail: KeywordBlocklistGuardrailPlugin, caplog: pytest.LogCaptureFixture):
    guardrail = await keyword_guardrail # Fixture already calls setup
    with caplog.at_level(logging.INFO, logger=GUARDRAIL_LOGGER_NAME):
        # Re-setup to capture its specific log, or check logs from fixture if possible
        # For simplicity, let's assume the fixture's setup log is what we'd expect
        # or we can re-initialize and setup here.
        temp_guardrail = KeywordBlocklistGuardrailPlugin()
        await temp_guardrail.setup() # Default config

    assert temp_guardrail._blocklist == set()
    assert temp_guardrail._case_sensitive is False
    assert temp_guardrail._action_on_match == "block"
    assert f"{temp_guardrail.plugin_id}: Initialized with 0 keywords. Case sensitive: False. Action on match: block." in caplog.text


@pytest.mark.asyncio
async def test_setup_with_blocklist_and_case_sensitivity(caplog: pytest.LogCaptureFixture):
    guardrail = KeywordBlocklistGuardrailPlugin()
    config = {
        "blocklist": ["Danger", "SECRET", "forbiddenWord"],
        "case_sensitive": True,
        "action_on_match": "warn"
    }
    with caplog.at_level(logging.INFO, logger=GUARDRAIL_LOGGER_NAME):
        await guardrail.setup(config)

    assert guardrail._blocklist == {"Danger", "SECRET", "forbiddenWord"}
    assert guardrail._case_sensitive is True
    assert guardrail._action_on_match == "warn"
    assert f"{guardrail.plugin_id}: Initialized with 3 keywords. Case sensitive: True. Action on match: warn." in caplog.text


@pytest.mark.asyncio
async def test_setup_blocklist_case_insensitive(caplog: pytest.LogCaptureFixture):
    guardrail = KeywordBlocklistGuardrailPlugin()
    config = {"blocklist": ["Danger", "SECRET"], "case_sensitive": False}
    with caplog.at_level(logging.INFO, logger=GUARDRAIL_LOGGER_NAME):
        await guardrail.setup(config)
    assert guardrail._blocklist == {"danger", "secret"}


@pytest.mark.asyncio
async def test_setup_invalid_action_on_match(caplog: pytest.LogCaptureFixture):
    guardrail = KeywordBlocklistGuardrailPlugin()
    with caplog.at_level(logging.WARNING, logger=GUARDRAIL_LOGGER_NAME):
        await guardrail.setup(config={"action_on_match": "invalid_action"})
    assert guardrail._action_on_match == "block" # Should default to block
    assert f"{guardrail.plugin_id}: Invalid action_on_match 'invalid_action'. Defaulting to 'block'." in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("text_input, expected_keyword", [
    ("This contains danger word", "danger"),
    ("SECRET information here", "secret"),
    ("A normal sentence", None),
    ("this is dangerous", "danger"), # Case-insensitive match
])
async def test_check_text_case_insensitive(text_input: str, expected_keyword: str):
    guardrail = KeywordBlocklistGuardrailPlugin()
    await guardrail.setup(config={"blocklist": ["Danger", "SECRET"], "case_sensitive": False})
    assert guardrail._check_text(text_input) == expected_keyword


@pytest.mark.asyncio
@pytest.mark.parametrize("text_input, expected_keyword", [
    ("This contains Danger word", "Danger"),
    ("SECRET information here", "SECRET"),
    ("secret should not match", None),
    ("A normal sentence", None),
])
async def test_check_text_case_sensitive(text_input: str, expected_keyword: str):
    guardrail = KeywordBlocklistGuardrailPlugin()
    await guardrail.setup(config={"blocklist": ["Danger", "SECRET"], "case_sensitive": True})
    assert guardrail._check_text(text_input) == expected_keyword


@pytest.mark.asyncio
async def test_check_input_string_match(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["badword"], "action_on_match": "block"})
    violation = await guardrail.check_input("This is a badword here.")
    assert violation["action"] == "block"
    assert violation["reason"] == "Blocked input keyword: 'badword'"
    assert violation["guardrail_id"] == guardrail.plugin_id


@pytest.mark.asyncio
async def test_check_input_chat_message_match(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["unsafe"], "action_on_match": "warn"})
    chat_message = {"role": "user", "content": "Is this unsafe content?"}
    violation = await guardrail.check_input(chat_message)
    assert violation["action"] == "warn"
    assert violation["reason"] == "Blocked input keyword: 'unsafe'"


@pytest.mark.asyncio
async def test_check_input_list_of_chat_messages_match(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["alert"]})
    messages = [
        {"role": "user", "content": "First message is fine."},
        {"role": "assistant", "content": "This is an alert!"}
    ]
    violation = await guardrail.check_input(messages)
    assert violation["action"] == "block"
    assert violation["reason"] == "Blocked input keyword: 'alert'"


@pytest.mark.asyncio
async def test_check_input_no_match(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["sensitive"]})
    violation_str = await guardrail.check_input("This is a normal sentence.")
    assert violation_str["action"] == "allow"
    violation_list = await guardrail.check_input([{"role": "user", "content": "Safe message."}])
    assert violation_list["action"] == "allow"


@pytest.mark.asyncio
async def test_check_input_unrecognized_format(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["test"]})
    violation = await guardrail.check_input(12345) # Integer input
    assert violation["action"] == "allow"
    assert violation["reason"] == "Input data format not recognized for keyword check."


@pytest.mark.asyncio
async def test_check_input_empty_blocklist(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail # Default setup has empty blocklist
    violation = await guardrail.check_input("Any input text.")
    assert violation["action"] == "allow"
    assert violation["reason"] == "Input passed keyword check." # Because _check_text returns None


@pytest.mark.asyncio
async def test_check_output_string_match(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["private"]})
    violation = await guardrail.check_output("This output contains private data.")
    assert violation["action"] == "block"
    assert violation["reason"] == "Blocked output keyword: 'private'"


@pytest.mark.asyncio
async def test_check_output_dict_content_match(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["confidential"]})

    output_data_text = {"text": "This is confidential output."}
    violation_text = await guardrail.check_output(output_data_text)
    assert violation_text["action"] == "block"
    assert violation_text["reason"] == "Blocked output keyword: 'confidential'"

    output_data_content = {"content": "More confidential stuff."}
    violation_content = await guardrail.check_output(output_data_content)
    assert violation_content["action"] == "block"
    assert violation_content["reason"] == "Blocked output keyword: 'confidential'"

    output_data_message = {"message": {"content": "A confidential message."}}
    violation_message = await guardrail.check_output(output_data_message)
    assert violation_message["action"] == "block"
    assert violation_message["reason"] == "Blocked output keyword: 'confidential'"


@pytest.mark.asyncio
async def test_check_output_dict_fallback_json_dump_match(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["secret_code"]})
    output_data = {"other_field": "Contains secret_code value."}
    violation = await guardrail.check_output(output_data)
    assert violation["action"] == "block"
    assert violation["reason"] == "Blocked output keyword: 'secret_code'"


@pytest.mark.asyncio
async def test_check_output_no_match(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["restricted"]})
    violation_str = await guardrail.check_output("This is public information.")
    assert violation_str["action"] == "allow"
    violation_dict = await guardrail.check_output({"data": "safe data"})
    assert violation_dict["action"] == "allow"


@pytest.mark.asyncio
async def test_check_output_unrecognized_format(keyword_guardrail: KeywordBlocklistGuardrailPlugin):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["test"]})
    violation = await guardrail.check_output(123.45) # Float output
    assert violation["action"] == "allow"
    assert violation["reason"] == "Output data format not recognized for keyword check."


@pytest.mark.asyncio
async def test_teardown_clears_blocklist(keyword_guardrail: KeywordBlocklistGuardrailPlugin, caplog: pytest.LogCaptureFixture):
    guardrail = await keyword_guardrail
    await guardrail.setup(config={"blocklist": ["temp_word"]})
    assert len(guardrail._blocklist) == 1

    with caplog.at_level(logging.DEBUG, logger=GUARDRAIL_LOGGER_NAME):
        await guardrail.teardown()

    assert len(guardrail._blocklist) == 0
    assert f"{guardrail.plugin_id}: Teardown complete, blocklist cleared." in caplog.text
